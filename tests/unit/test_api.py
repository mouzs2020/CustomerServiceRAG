"""api.py 的 TestClient 单元测试（全 Mock，无真实外部服务、无文件 I/O）。"""

import io
import json
import logging
import os
import re
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
from customer_service_rag.schemas import ReadinessResponse

SENSITIVE_TEXT = "qdrant refused; key=sk-secret; upstream body"

READY_CHECKS = ReadinessResponse(
    status="ready",
    checks={
        "deepseek_api_key": True,
        "embedding_manifest": True,
        "qdrant_collection": True,
        "dimension_match": True,
    },
)
NOT_READY_CHECKS = ReadinessResponse(
    status="not_ready",
    checks={
        "deepseek_api_key": False,
        "embedding_manifest": False,
        "qdrant_collection": False,
        "dimension_match": False,
    },
)


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


def _checker_returning(result):
    """构造 readiness 依赖的 override：FastAPI 调用 override 后，
    注入的必须是可调用的 checker，由 checker 再返回结果。"""
    def checker():
        return result
    return lambda: checker


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


class ReadyEndpointTests(ApiTestBase):
    def override_checker(self, result):
        api.app.dependency_overrides[api.get_readiness_checker] = (
            _checker_returning(result)
        )

    def test_ready_returns_200(self):
        self.override_checker(READY_CHECKS)
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ready")
        self.assertTrue(all(body["checks"].values()))

    def test_not_ready_returns_503(self):
        self.override_checker(NOT_READY_CHECKS)
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")

    def test_ready_does_not_call_runner(self):
        self.override_checker(READY_CHECKS)
        self.client.get("/ready")
        self.runner.assert_not_called()

    def test_openapi_declares_ready_with_503_schema(self):
        schema = api.app.openapi()
        responses = schema["paths"]["/ready"]["get"]["responses"]
        self.assertIn("200", responses)
        self.assertIn("503", responses)
        ref = responses["503"]["content"]["application/json"]["schema"]["$ref"]
        self.assertTrue(ref.endswith("/ReadinessResponse"))


class RequestIdHeaderTests(ApiTestBase):
    def get_request_id(self, response):
        return response.headers["X-Request-ID"]

    def test_headers_on_health_ready_answer_422_502_503(self):
        # 200（health）
        self.assertIn("X-Request-ID", self.client.get("/health").headers)
        # 200（ready）
        api.app.dependency_overrides[api.get_readiness_checker] = (
            _checker_returning(READY_CHECKS)
        )
        self.assertIn("X-Request-ID", self.client.get("/ready").headers)
        # 200（answer）
        self.runner = runner_returning(ready_response)
        ok = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        self.assertIn("X-Request-ID", ok.headers)
        # 422
        self.assertIn(
            "X-Request-ID",
            self.post_answer({"query": "", "entry_platform": "amazon"}).headers,
        )
        # 503
        self.runner = mock.Mock(side_effect=RuntimeError(SENSITIVE_TEXT))
        error_503 = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        self.assertIn("X-Request-ID", error_503.headers)
        # 502
        self.runner = mock.Mock(side_effect=ValueError(SENSITIVE_TEXT))
        error_502 = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        self.assertIn("X-Request-ID", error_502.headers)
        # 头部均为合法 UUID，且逐次不同
        ids = [self.get_request_id(r) for r in (ok, error_503, error_502)]
        for request_id in ids:
            uuid.UUID(request_id)
        self.assertEqual(len(set(ids)), len(ids))

    def test_answer_request_id_matches_header(self):
        self.runner = runner_returning(ready_response)
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        self.assertEqual(
            response.json()["request_id"],
            self.get_request_id(response),
        )

    def test_error_request_id_matches_header(self):
        self.runner = mock.Mock(side_effect=RuntimeError(SENSITIVE_TEXT))
        response = self.post_answer(
            {"query": "退款流程是什么", "entry_platform": "aliexpress"}
        )
        self.assertEqual(
            response.json()["request_id"],
            self.get_request_id(response),
        )

    def test_request_ids_differ_across_requests(self):
        first = self.get_request_id(self.client.get("/health"))
        second = self.get_request_id(self.client.get("/health"))
        self.assertNotEqual(first, second)
        uuid.UUID(first)
        uuid.UUID(second)


class StructuredLogTests(ApiTestBase):
    REQUIRED_FIELDS = {
        "request_id",
        "method",
        "path",
        "status_code",
        "duration_ms",
    }

    def capture_log(self, action):
        with mock.patch.object(api.logger, "info") as log_mock:
            action()
        self.assertTrue(log_mock.called)
        line = log_mock.call_args.args[0]
        return line, json.loads(line)

    def test_log_is_valid_json_with_exact_fields(self):
        line, data = self.capture_log(lambda: self.client.get("/health"))
        self.assertEqual(set(data), self.REQUIRED_FIELDS)
        self.assertEqual(data["method"], "GET")
        self.assertEqual(data["path"], "/health")
        self.assertEqual(data["status_code"], 200)
        self.assertGreaterEqual(data["duration_ms"], 0)
        uuid.UUID(data["request_id"])
        self.assertNotIn("\n", line)

    def test_log_excludes_sensitive_content(self):
        marker_query = "SECRET_QUERY_退款"
        self.runner = mock.Mock(side_effect=RuntimeError(SENSITIVE_TEXT))
        _, data = self.capture_log(
            lambda: self.post_answer(
                {"query": marker_query, "entry_platform": "aliexpress"}
            )
        )
        line = json.dumps(data, ensure_ascii=False)
        for marker in (marker_query, "sk-secret", "qdrant refused", "Evidence"):
            self.assertNotIn(marker, line)

    def test_unknown_exception_logs_500_and_returns_500(self):
        self.runner = mock.Mock(side_effect=KeyError("boom-marker"))

        def action():
            with TestClient(api.app, raise_server_exceptions=False) as client:
                return client.post(
                    "/v1/answer",
                    json={"query": "退款流程是什么", "entry_platform": "aliexpress"},
                )

        response_holder = {}
        with mock.patch.object(api.logger, "info") as log_mock:
            response_holder["response"] = action()
        self.assertTrue(log_mock.called)
        data = json.loads(log_mock.call_args.args[0])
        self.assertEqual(set(data), self.REQUIRED_FIELDS)
        self.assertEqual(data["status_code"], 500)
        self.assertNotIn("boom-marker", json.dumps(data))
        response = response_holder["response"]
        self.assertEqual(response.status_code, 500)


class LoggerConfigTests(ApiTestBase):
    REQUIRED_FIELDS = {
        "request_id",
        "method",
        "path",
        "status_code",
        "duration_ms",
    }

    def test_logger_deterministic_config(self):
        self.assertTrue(api.logger.isEnabledFor(logging.INFO))
        self.assertTrue(
            any(
                isinstance(handler, logging.StreamHandler)
                for handler in api.logger.handlers
            )
        )
        self.assertFalse(api.logger.propagate)

    def test_each_request_logs_exactly_one_structured_line(self):
        with mock.patch.object(api.logger, "info") as log_mock:
            self.client.get("/health")
            self.client.get("/health")
        self.assertEqual(log_mock.call_count, 2)
        request_ids = []
        for call in log_mock.call_args_list:
            data = json.loads(call.args[0])
            self.assertEqual(set(data), self.REQUIRED_FIELDS)
            request_ids.append(data["request_id"])
        self.assertNotEqual(request_ids[0], request_ids[1])


def _extract_js_function(js: str, name: str) -> str:
    """按花括号配平提取 app.js 顶层函数源码，供前端契约断言使用。"""
    start = js.index(f"function {name}(")
    open_brace = js.index("{", start)
    depth = 0
    for index in range(open_brace, len(js)):
        if js[index] == "{":
            depth += 1
        elif js[index] == "}":
            depth -= 1
            if depth == 0:
                return js[start : index + 1]
    raise AssertionError(f"function {name} is not closed in app.js")


class StaticPageTests(unittest.TestCase):
    """本地 Web 演示界面：静态页面托管与安全约束（不触碰 RAG 管线）。"""

    def setUp(self):
        self.client = TestClient(api.app)
        self.runner = mock.Mock()
        self.readiness_checker = mock.Mock()
        api.app.dependency_overrides[api.get_pipeline_runner] = (
            lambda: self.runner
        )
        api.app.dependency_overrides[api.get_readiness_checker] = (
            lambda: self.readiness_checker
        )

    def tearDown(self):
        api.app.dependency_overrides.clear()

    def get(self, path):
        return self.client.get(path)

    def test_index_returns_200_with_html(self):
        response = self.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/html")
        )
        html = response.text
        self.assertIn("跨境售后知识助手", html)

    def test_index_contains_platform_question_and_send_controls(self):
        html = self.get("/").text
        # 平台分段控件（必须明确选择的单选组）
        self.assertIn('name="entry-platform"', html)
        self.assertIn('value="aliexpress"', html)
        self.assertIn('value="temu"', html)
        self.assertIn('value="agent"', html)
        self.assertIn("AliExpress", html)
        self.assertIn("Temu", html)
        self.assertIn("Agent 自动路由", html)
        # 问题输入与发送控件
        self.assertIn('id="question-input"', html)
        self.assertIn('id="send-button"', html)
        self.assertIn("发送问题", html)
        # 引用 /static 下的样式与脚本
        self.assertIn("/static/app.css", html)
        self.assertIn("/static/app.js", html)

    def test_css_and_js_assets_return_200(self):
        for path in ("/static/app.css", "/static/app.js"):
            with self.subTest(path=path):
                response = self.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(len(response.content) > 0)

    def test_static_assets_serve_expected_content(self):
        css = self.get("/static/app.css").text
        self.assertIn(".send-button", css)
        js = self.get("/static/app.js").text
        self.assertIn("/v1/answer", js)
        self.assertIn("/v1/agent/answer", js)
        self.assertIn("entry_platform", js)

    def test_static_page_access_does_not_trigger_rag_or_readiness(self):
        for path in ("/", "/static/app.css", "/static/app.js"):
            self.get(path)
        self.runner.assert_not_called()
        self.readiness_checker.assert_not_called()

    def test_index_carries_request_id_header(self):
        response = self.get("/")
        self.assertIn("X-Request-ID", response.headers)
        uuid.UUID(response.headers["X-Request-ID"])

    def test_frontend_uses_textcontent_only(self):
        js = (api.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        for forbidden in (
            "innerHTML",
            "outerHTML",
            "insertAdjacentHTML",
            "document.write",
        ):
            self.assertNotIn(forbidden, js)
        self.assertIn("textContent", js)

    def test_service_ready_gates_send_button(self):
        js = (api.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        # serviceReady 初始为 false，仅 /ready 返回 ready 时为 true。
        self.assertIn("var serviceReady = false;", js)
        render = _extract_js_function(js, "renderServiceStatus")
        self.assertIn('serviceReady = status === "ready";', render)
        # 发送按钮可用性必须同时包含非 loading / serviceReady / 输入合法。
        enabled = _extract_js_function(js, "isSendEnabled")
        self.assertIn("!loading", enabled)
        self.assertIn("serviceReady", enabled)
        self.assertIn("isInputValid", enabled)
        for name in ("refreshSendState", "setLoadingState"):
            body = _extract_js_function(js, name)
            self.assertIn("isSendEnabled()", body)
        # /ready 请求失败按 unknown 处理，serviceReady 保持 false。
        self.assertIn('renderServiceStatus("unknown")', js)

    def test_platform_controls_disabled_while_loading(self):
        js = (api.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        body = _extract_js_function(js, "setLoadingState")
        # loading 期间：问题输入与两个 entry-platform radio 均被禁用。
        self.assertIn("els.question.disabled = next;", body)
        self.assertIn("els.platformRadios.forEach", body)
        self.assertIn("radio.disabled = next;", body)
        self.assertIn('input[name="entry-platform"]', js)
        # 请求结束后按钮状态交回 isSendEnabled（serviceReady + 输入合法性）。
        self.assertIn("els.send.disabled = !isSendEnabled();", body)

    def test_answer_timeout_at_least_180000(self):
        js = (api.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        match = re.search(r"var ANSWER_TIMEOUT_MS = (\d+);", js)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(int(match.group(1)), 180000)

    def test_agent_answer_timeout_is_360000(self):
        js = (api.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        match = re.search(r"var AGENT_ANSWER_TIMEOUT_MS = (\d+);", js)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 360000)

    def test_status_text_uses_chinese_labels(self):
        js = (api.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        render = _extract_js_function(js, "renderServiceStatus")
        self.assertIn("服务就绪", render)
        self.assertIn("服务未就绪", render)
        self.assertIn("状态检测失败", render)
        self.assertNotIn('setText(els.statusText, "ready")', render)
        self.assertNotIn('setText(els.statusText, "not_ready")', render)

    def test_openapi_contract_unchanged_by_static_routes(self):
        schema = api.app.openapi()
        self.assertNotIn("/", schema["paths"])
        for path in ("/health", "/ready", "/v1/answer"):
            self.assertIn(path, schema["paths"])


if __name__ == "__main__":
    unittest.main()
