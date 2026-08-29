"""orchestrator.run_answer_pipeline 的单元测试（全 Mock，纯内存）。"""

import io
import json
import os
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_service_rag import prepare_evidence_qdrant as peq
from customer_service_rag.orchestrator import run_answer_pipeline
from customer_service_rag.platform_gate import UNRELATED_FALLBACK, UNCERTAIN_FALLBACK
from customer_service_rag.schemas import AnswerRequest, Intent, Platform

GOOD_INTENT = {
    "intent": "refund_after_sales",
    "confidence": 0.95,
    "reason": "退款咨询",
}


def make_evidence(citation_id):
    return {
        "citation_id": citation_id,
        "chunk_id": f"chunk-{citation_id}",
        "source_id": "source-001",
        "platform": "aliexpress",
        "headings": ["退款规则", "第三章 审核流程"],
        "text": "退款申请提交后，平台按照流程处理。",
        "retrieve_score": 0.81,
        "rerank_score": 6.0 if citation_id == "E1" else 3.0,
    }


def make_ready_bundle(entry="aliexpress", requested="aliexpress"):
    """用生产 build_evidence_bundle 构造完整兼容的 ready 证据包。"""
    return peq.build_evidence_bundle(
        query="退款流程是什么",
        status="ready_for_grounding",
        reason="Evidence gate passed; answer stage must still verify",
        requested_platform=requested,
        entry_platform=entry,
        intent_result=dict(GOOD_INTENT),
        evidence_gate={
            "passed": True,
            "min_rerank_score": 0.75,
            "checked_candidates": 2,
            "top_rerank_score": 6.0,
        },
        evidence=[make_evidence("E1"), make_evidence("E2")],
    )


def make_blocked_bundle(
    status,
    *,
    entry="aliexpress",
    requested="aliexpress",
    intent_result=None,
    evidence_gate=None,
):
    return peq.build_evidence_bundle(
        query="退款流程是什么",
        status=status,
        reason="Evidence gate blocked",
        requested_platform=requested,
        entry_platform=entry,
        intent_result=intent_result,
        evidence_gate=evidence_gate,
    )


def fake_generate_result(answer="根据规则处理。[E1]", citations=("E1",)):
    return {
        "model": "deepseek-v4-flash",
        "query": "退款流程是什么",
        "answer": answer,
        "used_citations": list(citations),
    }


def make_request(entry=Platform.ALIEXPRESS):
    return AnswerRequest(query="退款流程是什么", entry_platform=entry)


class ReadyPathTests(unittest.TestCase):
    def test_prepare_called_with_query_and_platform_string(self):
        for entry, expected in (
            (Platform.ALIEXPRESS, "aliexpress"),
            (Platform.TEMU, "temu"),
        ):
            with self.subTest(entry=expected):
                prepare = mock.Mock(return_value=make_ready_bundle(entry, expected))
                run_answer_pipeline(
                    make_request(entry),
                    prepare=prepare,
                    generate=mock.Mock(return_value=fake_generate_result()),
                )
                prepare.assert_called_once_with("退款流程是什么", expected)

    def test_entry_none_passes_none_to_prepare(self):
        prepare = mock.Mock(return_value=make_ready_bundle(entry=None, requested=None))
        run_answer_pipeline(
            AnswerRequest(query="退款流程是什么"),
            prepare=prepare,
            generate=mock.Mock(return_value=fake_generate_result()),
        )
        prepare.assert_called_once_with("退款流程是什么", None)

    def test_ready_passes_same_bundle_object_to_generate(self):
        bundle = make_ready_bundle()
        prepare = mock.Mock(return_value=bundle)
        generate = mock.Mock(return_value=fake_generate_result())
        run_answer_pipeline(make_request(), prepare=prepare, generate=generate)
        generate.assert_called_once()
        self.assertIs(generate.call_args.args[0], bundle)

    def test_ready_response_contains_answer_citations_and_evidence(self):
        response = run_answer_pipeline(
            make_request(),
            prepare=mock.Mock(return_value=make_ready_bundle()),
            generate=mock.Mock(return_value=fake_generate_result()),
        )
        self.assertEqual(response.status, "ready_for_grounding")
        self.assertEqual(response.answer, "根据规则处理。[E1]")
        self.assertEqual(response.used_citations, ["E1"])
        self.assertEqual(response.entry_platform, Platform.ALIEXPRESS)
        self.assertEqual(response.requested_platform, Platform.ALIEXPRESS)
        self.assertEqual(response.intent, Intent.REFUND_AFTER_SALES)
        self.assertEqual(response.intent_confidence, 0.95)
        self.assertEqual(response.reason, "Evidence gate passed; answer stage must still verify")
        self.assertEqual(len(response.evidence), 2)
        self.assertEqual(response.evidence[0].citation_id, "E1")
        self.assertTrue(response.evidence_gate.passed)


class BlockedPathTests(unittest.TestCase):
    def assert_generate_not_called(self, bundle):
        prepare = mock.Mock(return_value=bundle)
        generate = mock.Mock()
        response = run_answer_pipeline(
            make_request(), prepare=prepare, generate=generate
        )
        generate.assert_not_called()
        self.assertIsNone(response.answer)
        self.assertEqual(response.used_citations, [])
        return response

    def test_missing_platform_does_not_call_generate(self):
        response = self.assert_generate_not_called(
            make_blocked_bundle("blocked_missing_platform", entry=None, requested=None)
        )
        self.assertEqual(response.status, "blocked_missing_platform")
        self.assertIsNone(response.entry_platform)
        self.assertIsNone(response.requested_platform)

    def test_platform_conflict_does_not_call_generate(self):
        response = self.assert_generate_not_called(
            make_blocked_bundle("blocked_platform_conflict", requested=None)
        )
        self.assertEqual(response.status, "blocked_platform_conflict")
        self.assertIsNone(response.requested_platform)

    def test_low_relevance_does_not_call_generate(self):
        bundle = make_blocked_bundle(
            "blocked_low_relevance",
            intent_result=dict(GOOD_INTENT),
            evidence_gate={
                "passed": False,
                "min_rerank_score": 0.75,
                "checked_candidates": 1,
                "top_rerank_score": 0.5,
            },
        )
        response = self.assert_generate_not_called(bundle)
        self.assertEqual(response.status, "blocked_low_relevance")
        self.assertFalse(response.evidence_gate.passed)
        self.assertEqual(response.evidence, [])
        self.assertEqual(response.intent, Intent.REFUND_AFTER_SALES)

    def test_unrelated_calls_generate_and_returns_fallback(self):
        bundle = make_blocked_bundle(
            "blocked_unrelated_question",
            intent_result={
                "intent": "unrelated",
                "confidence": 0.98,
                "reason": "闲聊",
            },
        )
        prepare = mock.Mock(return_value=bundle)
        generate = mock.Mock(
            return_value=fake_generate_result(answer=UNRELATED_FALLBACK, citations=())
        )
        response = run_answer_pipeline(make_request(), prepare=prepare, generate=generate)
        generate.assert_called_once()
        self.assertEqual(response.status, "blocked_unrelated_question")
        self.assertEqual(response.answer, UNRELATED_FALLBACK)
        self.assertEqual(response.used_citations, [])

    def test_uncertain_calls_generate_and_returns_fallback(self):
        bundle = make_blocked_bundle(
            "blocked_intent_uncertain",
            intent_result={
                "intent": "uncertain",
                "confidence": 0.55,
                "reason": "信息不足",
            },
        )
        prepare = mock.Mock(return_value=bundle)
        generate = mock.Mock(
            return_value=fake_generate_result(answer=UNCERTAIN_FALLBACK, citations=())
        )
        response = run_answer_pipeline(make_request(), prepare=prepare, generate=generate)
        generate.assert_called_once()
        self.assertEqual(response.status, "blocked_intent_uncertain")
        self.assertEqual(response.answer, UNCERTAIN_FALLBACK)
        self.assertEqual(response.used_citations, [])


class ErrorPropagationTests(unittest.TestCase):
    def test_prepare_infra_error_propagates_and_generate_not_called(self):
        prepare = mock.Mock(side_effect=RuntimeError("qdrant connection refused"))
        generate = mock.Mock()
        with self.assertRaises(RuntimeError) as ctx:
            run_answer_pipeline(make_request(), prepare=prepare, generate=generate)
        self.assertEqual(str(ctx.exception), "qdrant connection refused")
        generate.assert_not_called()

    def test_generate_error_propagates(self):
        prepare = mock.Mock(return_value=make_ready_bundle())
        generate = mock.Mock(side_effect=RuntimeError("deepseek 500"))
        with self.assertRaises(RuntimeError) as ctx:
            run_answer_pipeline(make_request(), prepare=prepare, generate=generate)
        self.assertEqual(str(ctx.exception), "deepseek 500")


class RequestIdTests(unittest.TestCase):
    def run_with(self, **kwargs):
        return run_answer_pipeline(
            make_request(),
            prepare=mock.Mock(return_value=make_ready_bundle()),
            generate=mock.Mock(return_value=fake_generate_result()),
            **kwargs,
        )

    def test_fixed_factory_id_appears_in_response(self):
        response = self.run_with(request_id_factory=lambda: "fixed-req-001")
        self.assertEqual(response.request_id, "fixed-req-001")

    def test_default_request_id_is_valid_uuid(self):
        response = self.run_with()
        uuid.UUID(response.request_id)  # 解析失败会抛 ValueError


class PurityAndSerializationTests(unittest.TestCase):
    def test_no_output_files_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    response = run_answer_pipeline(
                        make_request(),
                        prepare=mock.Mock(return_value=make_ready_bundle()),
                        generate=mock.Mock(return_value=fake_generate_result()),
                    )
                self.assertEqual(buffer.getvalue(), "")
                self.assertFalse((Path(tmp) / "output").exists())
                self.assertEqual(response.status, "ready_for_grounding")
            finally:
                os.chdir(saved_cwd)

    def test_response_model_dump_json(self):
        response = run_answer_pipeline(
            make_request(),
            prepare=mock.Mock(return_value=make_ready_bundle()),
            generate=mock.Mock(return_value=fake_generate_result()),
        )
        data = json.loads(response.model_dump_json())
        self.assertEqual(data["status"], "ready_for_grounding")
        self.assertEqual(data["entry_platform"], "aliexpress")
        self.assertEqual(data["intent"], "refund_after_sales")
        self.assertEqual(data["evidence"][0]["citation_id"], "E1")
        self.assertEqual(data["answer"], "根据规则处理。[E1]")
        self.assertEqual(data["used_citations"], ["E1"])


if __name__ == "__main__":
    unittest.main()
