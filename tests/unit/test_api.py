"""api.py 的 TestClient 单元测试（全 Mock，无真实外部服务、无文件 I/O）。"""

import io
import os
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi.testclient import TestClient

from customer_service_rag import api
from customer_service_rag.platform_gate import UNRELATED_FALLBACK

SENSITIVE_TEXT = "qdrant refused; key=sk-secret; upstream body"


def make_evidence():
    return {
        "citation_id": "E1",
        "chunk_id": "chunk-E1",
        "source_id": "source-001",
        "platform": "aliexpress",
        "headings": ["退款规则"],
        "text": "退款申请提交后，平台按照流程处理。",
        "retrieve_score": 0.81,
        "rerank_score": 6.0,
    }


def ready_response(request_id):
    return dict(
        request_id=request_id,
        status="ready_for_grounding",
        reason="Evidence gate passed",
        entry_platform="aliexpress",
        requested_platform="aliexpress",
        intent="refund_after_sales",
        intent_confidence=0.95,
        intent_reason="退款咨询",
        evidence_gate={
            "passed": True,
            "min_rerank_score": 0.75,
            "checked_candidates": 2,
            "top_rerank_score": 6.0,
        },
        evidence=[make_evidence()],
        answer="根据规则处理。[E1]",
        used_citations=["E1"],
    )


def blocked_response(status, answer=None, citations=(), requested="aliexpress"):
    return dict(
        request_id="will-be-replaced",
        status=status,
        reason="Evidence gate blocked",
        entry_platform="aliexpress",
        requested_platform=requested,
        intent=None,
        intent_confidence=None,
        intent_reason=None,
        evidence_gate=None,
        evidence=[],
        answer=answer,
        used_citations=list(citations),
    )


def runner_returning(builder):
    """构造使用 API 注入的 request_id 的 runner Mock。"""

    def runner(request, *, request_id_factory=None):
        return builder(request_id_factory())

    return mock.Mock(side_effect=runner)


class ApiTestBase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.runner = mock.Mock()
        api.app.dependency_overrides[api.get_pipeline_runner] = (
            lambda: self.runner
        )

    def tearDown(self):
        # 要求 17：所有测试结束后清理 dependency_overrides。
        api.app.dependency_overrides.clear()

    def post_answer(self, payload):
        return self.client.post("/v1/answer", json=payload)


class HealthTests(ApiTestBase):
    def test_health_returns_200_and_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_health_does_not_call_runner(self):
        self.client.get("/health")
        self.runner.assert_not_called()


class ReadyAnswerTests(ApiTestBase):
    def test_ready_returns_200_with_full_response(self):
        self.runner = runner_returning(ready_response)
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ready_for_grounding")
        self.assertEqual(body["answer"], "根据规则处理。[E1]")
        self.assertEqual(body["used_citations"], ["E1"])
        self.assertEqual(body["evidence"][0]["citation_id"], "E1")
        self.assertEqual(body["entry_platform"], "aliexpress")
        uuid.UUID(body["request_id"])

    def test_entry_platform_missing_reaches_runner(self):
        self.runner = runner_returning(ready_response)
        response = self.post_answer({"query": "退款流程是什么"})
        self.assertEqual(response.status_code, 200)
        self.runner.assert_called_once()
        request = self.runner.call_args.args[0]
        self.assertIsNone(request.entry_platform)

    def test_query_stripped_before_runner(self):
        self.runner = runner_returning(ready_response)
        api.app.dependency_overrides[api.get_pipeline_runner] = (
            lambda: self.runner
        )
        self.post_answer({"query": "  退款流程是什么  "})
        request = self.runner.call_args.args[0]
        self.assertEqual(request.query, "退款流程是什么")


class BlockedAnswerTests(ApiTestBase):
    def test_conflict_returns_200_with_null_answer(self):
        self.runner = runner_returning(
            lambda rid: blocked_response("blocked_platform_conflict", requested=None)
        )
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "blocked_platform_conflict")
        self.assertIsNone(body["answer"])
        self.assertEqual(body["used_citations"], [])

    def test_unrelated_returns_200_with_fallback(self):
        self.runner = runner_returning(
            lambda rid: blocked_response(
                "blocked_unrelated_question", answer=UNRELATED_FALLBACK
            )
        )
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], UNRELATED_FALLBACK)
        self.assertEqual(body["used_citations"], [])


class ValidationErrorTests(ApiTestBase):
    def test_invalid_platform_returns_422_without_runner_call(self):
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "amazon"}
        )
        self.assertEqual(response.status_code, 422)
        self.runner.assert_not_called()

    def test_empty_and_whitespace_query_return_422_without_runner_call(self):
        for query in ("", "   "):
            with self.subTest(query=query):
                response = self.post_answer({"query": query})
                self.assertEqual(response.status_code, 422)
        self.runner.assert_not_called()


class ErrorMappingTests(ApiTestBase):
    def assert_error_response(self, response, status_code, error_code):
        self.assertEqual(response.status_code, status_code)
        body = response.json()
        self.assertEqual(body["error_code"], error_code)
        self.assertNotIn(SENSITIVE_TEXT, response.text)
        uuid.UUID(body["request_id"])
        return body

    def test_runtime_error_maps_to_503(self):
        self.runner.side_effect = RuntimeError(SENSITIVE_TEXT)
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        body = self.assert_error_response(response, 503, "service_unavailable")
        self.assertEqual(body["reason"], api.SERVICE_UNAVAILABLE_REASON)

    def test_httpx_request_error_maps_to_503(self):
        self.runner.side_effect = httpx.ConnectTimeout(SENSITIVE_TEXT)
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        body = self.assert_error_response(response, 503, "service_unavailable")
        self.assertEqual(body["reason"], api.SERVICE_UNAVAILABLE_REASON)

    def test_value_error_maps_to_502(self):
        self.runner.side_effect = ValueError(SENSITIVE_TEXT)
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        body = self.assert_error_response(response, 502, "invalid_upstream_response")
        self.assertEqual(body["reason"], api.INVALID_UPSTREAM_REASON)

    def test_unknown_exception_maps_to_500(self):
        self.runner.side_effect = KeyError("unexpected")
        with TestClient(api.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/answer",
                json={"query": "退款流程是什么", "entry_platform": "aliexpress"},
            )
        self.assertEqual(response.status_code, 500)


class PurityTests(ApiTestBase):
    def test_api_calls_create_no_output_files(self):
        self.runner = runner_returning(ready_response)
        api.app.dependency_overrides[api.get_pipeline_runner] = (
            lambda: self.runner
        )
        with tempfile.TemporaryDirectory() as tmp:
            saved_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    response = self.post_answer(
                        {"query": "退款流程是什么", "entry_platform": "aliexpress"}
                    )
                self.assertEqual(buffer.getvalue(), "")
                self.assertEqual(response.status_code, 200)
                self.assertFalse((Path(tmp) / "output").exists())
            finally:
                os.chdir(saved_cwd)


class OpenApiContractTests(unittest.TestCase):
    def test_answer_endpoint_declares_status_codes_and_error_schema(self):
        schema = api.app.openapi()
        responses = schema["paths"]["/v1/answer"]["post"]["responses"]
        for code in ("200", "422", "502", "503"):
            self.assertIn(code, responses)
        self.assertIn("ApiErrorResponse", schema["components"]["schemas"])
        for code in ("502", "503"):
            ref = responses[code]["content"]["application/json"]["schema"]["$ref"]
            self.assertTrue(
                ref.endswith("/ApiErrorResponse"),
                f"{code} 应引用 ApiErrorResponse，实际 {ref!r}",
            )
        self.assertIn(
            "AnswerResponse", schema["components"]["schemas"]
        )
        health_ok = schema["paths"]["/health"]["get"]["responses"]["200"]
        self.assertIn("HealthResponse", schema["components"]["schemas"])
        self.assertIn("application/json", health_ok["content"])


if __name__ == "__main__":
    unittest.main()
