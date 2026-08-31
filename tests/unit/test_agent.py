"""Bounded agent tests with injected router, RAG pipeline, and DeepSeek HTTP."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi.testclient import TestClient

from customer_service_rag import api
from customer_service_rag.agent_orchestrator import (
    AgentAnswerResponse,
    AgentToolTrace,
    run_agent_orchestrator,
)
from customer_service_rag.agent_router import (
    AgentAction,
    AgentRoute,
    AgentRouterError,
    route_agent,
)
from customer_service_rag.schemas import AgentAnswerRequest, AnswerResponse, Platform


def make_answer(platform: str, request_id: str) -> AnswerResponse:
    return AnswerResponse(
        request_id=request_id,
        status="ready_for_grounding",
        reason="Evidence gate passed",
        entry_platform=platform,
        requested_platform=platform,
        intent="refund_after_sales",
        intent_confidence=0.95,
        intent_reason="退款咨询",
        evidence_gate={
            "passed": True,
            "min_rerank_score": 0.75,
            "checked_candidates": 1,
            "top_rerank_score": 6.0,
        },
        evidence=[
            {
                "citation_id": "E1",
                "chunk_id": f"chunk-{platform}",
                "source_id": f"source-{platform}",
                "platform": platform,
                "headings": ["退款规则"],
                "text": "退款申请提交后，平台按照流程处理。",
                "retrieve_score": 0.81,
                "rerank_score": 6.0,
            }
        ],
        answer=f"{platform} 原始回答。[E1]",
        used_citations=["E1"],
    )


class FakeDeepSeekResponse:
    is_error = False

    def __init__(self, content):
        self.content = content

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class AgentRouterTests(unittest.TestCase):
    def test_real_entry_uses_existing_deepseek_http_shape_and_validates_json(self):
        fake = FakeDeepSeekResponse(
            '{"action":"single_platform","platform":"temu","clarification":null}'
        )
        with mock.patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_INTENT_MODEL": "shared-model",
            },
        ), mock.patch(
            "customer_service_rag.agent_router.httpx.post", return_value=fake
        ) as post:
            route = route_agent("退款流程是什么", Platform.TEMU)

        self.assertEqual(route.action, AgentAction.SINGLE_PLATFORM)
        self.assertEqual(route.platform, Platform.TEMU)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "shared-model")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["stream"], False)
        self.assertIn('只输出一个原始 JSON 对象', payload["messages"][0]["content"])

    def test_invalid_json_is_safe_router_error(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), mock.patch(
            "customer_service_rag.agent_router.httpx.post",
            return_value=FakeDeepSeekResponse("not-json"),
        ):
            with self.assertRaises(AgentRouterError):
                route_agent("退款流程是什么")

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            AgentRoute(action="single_platform", platform="amazon")
        with self.assertRaises(ValueError):
            AgentRoute(action="clarify", clarification="   ")


class AgentOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.request = AgentAnswerRequest(query="退款流程是什么")

    def test_single_platform_calls_pipeline_once(self):
        router = mock.Mock(
            return_value=AgentRoute(
                action=AgentAction.SINGLE_PLATFORM,
                platform=Platform.ALIEXPRESS,
            )
        )
        pipeline = mock.Mock(return_value=make_answer("aliexpress", "req-1"))
        result = run_agent_orchestrator(
            self.request,
            router=router,
            pipeline=pipeline,
            request_id_factory=lambda: "agent-req",
        )
        self.assertEqual(result.action, AgentAction.SINGLE_PLATFORM)
        self.assertEqual(result.answer.answer, "aliexpress 原始回答。[E1]")
        self.assertEqual(len(result.tool_trace), 1)
        pipeline.assert_called_once()
        self.assertEqual(pipeline.call_args.args[0].entry_platform, Platform.ALIEXPRESS)

    def test_entry_platform_takes_precedence_over_router_platform(self):
        request = AgentAnswerRequest(
            query="AliExpress 退款流程是什么",
            entry_platform=Platform.TEMU,
        )
        router = mock.Mock(
            return_value=AgentRoute(
                action=AgentAction.SINGLE_PLATFORM,
                platform=Platform.ALIEXPRESS,
            )
        )
        pipeline = mock.Mock(return_value=make_answer("temu", "req-temu"))

        result = run_agent_orchestrator(request, router=router, pipeline=pipeline)

        passed_request = pipeline.call_args.args[0]
        self.assertEqual(passed_request.entry_platform, Platform.TEMU)
        self.assertEqual(result.tool_trace[0].platform, Platform.TEMU)

    def test_compare_is_sequential_and_returns_two_raw_answers(self):
        router = mock.Mock(return_value=AgentRoute(action="compare_platforms"))
        pipeline = mock.Mock(
            side_effect=[
                make_answer("aliexpress", "req-a"),
                make_answer("temu", "req-t"),
            ]
        )
        result = run_agent_orchestrator(self.request, router=router, pipeline=pipeline)
        self.assertEqual(
            [call.args[0].entry_platform for call in pipeline.call_args_list],
            [Platform.ALIEXPRESS, Platform.TEMU],
        )
        self.assertEqual([item.answer for item in result.answers], [
            "aliexpress 原始回答。[E1]",
            "temu 原始回答。[E1]",
        ])
        self.assertEqual([trace.step for trace in result.tool_trace], [1, 2])
        self.assertEqual(pipeline.call_count, 2)

    def test_clarify_and_reject_do_not_call_rag(self):
        pipeline = mock.Mock()
        for route in (
            AgentRoute(action="clarify", clarification="请说明平台和售后问题。"),
            AgentRoute(action="reject"),
        ):
            with self.subTest(action=route.action):
                result = run_agent_orchestrator(
                    self.request,
                    router=mock.Mock(return_value=route),
                    pipeline=pipeline,
                )
                self.assertEqual(result.action, route.action)
                self.assertEqual(result.tool_trace, [])
        pipeline.assert_not_called()

    def test_invalid_router_parameters_stop_before_tool(self):
        pipeline = mock.Mock()
        with self.assertRaises(ValueError):
            run_agent_orchestrator(
                self.request,
                router=mock.Mock(
                    return_value={"action": "single_platform", "platform": "amazon"}
                ),
                pipeline=pipeline,
            )
        pipeline.assert_not_called()


class AgentApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.router = mock.Mock()
        self.pipeline = mock.Mock()
        api.app.dependency_overrides[api.get_agent_router] = lambda: self.router
        api.app.dependency_overrides[api.get_agent_pipeline_runner] = lambda: self.pipeline

    def tearDown(self):
        api.app.dependency_overrides.clear()

    def test_agent_error_mapping_is_safe(self):
        self.router.side_effect = ValueError("model output contains secret")
        response = self.client.post(
            "/v1/agent/answer",
            json={"query": "天气怎么样"},
        )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)

    def test_agent_runtime_error_maps_to_503(self):
        self.router.side_effect = RuntimeError("upstream body with secret")
        response = self.client.post(
            "/v1/agent/answer",
            json={"query": "天气怎么样"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)

    def test_compare_api_returns_two_answers_and_trace(self):
        self.router.return_value = AgentRoute(action="compare_platforms")
        self.pipeline.side_effect = [
            make_answer("aliexpress", "req-a"),
            make_answer("temu", "req-t"),
        ]
        response = self.client.post(
            "/v1/agent/answer",
            json={"query": "比较两个平台的退款流程"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["action"], "compare_platforms")
        self.assertEqual(len(body["answers"]), 2)
        self.assertEqual(body["answers"][0]["used_citations"], ["E1"])
        self.assertEqual(
            [item["platform"] for item in body["tool_trace"]],
            ["aliexpress", "temu"],
        )
        self.assertEqual(self.pipeline.call_count, 2)

    def test_reject_api_has_zero_rag_calls(self):
        self.router.return_value = AgentRoute(action="reject")
        response = self.client.post(
            "/v1/agent/answer",
            json={"query": "天气怎么样"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "reject")
        self.assertEqual(response.json()["tool_trace"], [])
        self.pipeline.assert_not_called()

    def test_clarify_api_has_zero_rag_calls(self):
        self.router.return_value = AgentRoute(
            action="clarify", clarification="请补充平台和具体问题。"
        )
        response = self.client.post(
            "/v1/agent/answer",
            json={"query": "这个订单怎么处理"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "clarify")
        self.assertEqual(response.json()["tool_trace"], [])
        self.pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
