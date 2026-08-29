"""prepare_evidence 纯内存函数与 main() CLI 行为的单元测试。"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_service_rag import prepare_evidence_qdrant as peq
from customer_service_rag.intent_classifier import IntentClassifierError

GOOD_INTENT = {
    "intent": "refund_after_sales",
    "confidence": 0.95,
    "reason": "退款咨询",
}


def make_candidate(platform="aliexpress", rerank=6.0):
    return {
        "record": {
            "chunk_id": f"{platform}-chunk-001",
            "source_id": f"{platform}-source-001",
            "platform": platform,
            "headings": ["退款规则"],
            "text": "示例证据文本。",
        },
        "retrieve_rank": 1,
        "retrieve_score": 0.8,
        "rerank_score": rerank,
    }


class PlatformGateShortCircuitTests(unittest.TestCase):
    def run_prepare(self, query, entry, classify=None, retrieve=None):
        classify = classify or mock.Mock(name="classify")
        retrieve = retrieve or mock.Mock(name="retrieve")
        bundle = peq.prepare_evidence(
            query, entry, classify=classify, retrieve=retrieve
        )
        return bundle, classify, retrieve

    def test_missing_platform_skips_classify_and_retrieve(self):
        bundle, classify, retrieve = self.run_prepare("退款流程是什么", None)
        self.assertEqual(bundle["status"], "blocked_missing_platform")
        classify.assert_not_called()
        retrieve.assert_not_called()

    def test_platform_conflict_skips_classify_and_retrieve(self):
        bundle, classify, retrieve = self.run_prepare(
            "Temu退款流程是什么", "aliexpress"
        )
        self.assertEqual(bundle["status"], "blocked_platform_conflict")
        classify.assert_not_called()
        retrieve.assert_not_called()


class IntentStageTests(unittest.TestCase):
    def test_classifier_error_maps_to_blocked_status(self):
        classify = mock.Mock(
            side_effect=IntentClassifierError("Classifier output is not valid JSON")
        )
        retrieve = mock.Mock()
        bundle = peq.prepare_evidence(
            "退款流程是什么", "aliexpress", classify=classify, retrieve=retrieve
        )
        self.assertEqual(bundle["status"], "blocked_intent_classifier_error")
        self.assertIn("Intent classifier failed", bundle["reason"])
        retrieve.assert_not_called()

    def test_threshold_config_error_keeps_intent_fields_null(self):
        """decide_after_intent 抛错时，意图三字段必须保持 None（旧实现兼容）。"""
        classify = mock.Mock(return_value=dict(GOOD_INTENT))
        retrieve = mock.Mock()
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "INTENT_CONFIDENCE_THRESHOLD"
        }
        with mock.patch.dict(
            os.environ, {**env, "INTENT_CONFIDENCE_THRESHOLD": "abc"}, clear=True
        ):
            bundle = peq.prepare_evidence(
                "退款流程是什么", "aliexpress",
                classify=classify, retrieve=retrieve,
            )
        self.assertEqual(bundle["status"], "blocked_intent_classifier_error")
        self.assertIn("Invalid INTENT_CONFIDENCE_THRESHOLD", bundle["reason"])
        self.assertIsNone(bundle["intent"])
        self.assertIsNone(bundle["intent_confidence"])
        self.assertIsNone(bundle["intent_reason"])
        retrieve.assert_not_called()

    def test_unrelated_does_not_call_retrieve(self):
        classify = mock.Mock(
            return_value={
                "intent": "unrelated",
                "confidence": 0.98,
                "reason": "闲聊",
            }
        )
        retrieve = mock.Mock()
        bundle = peq.prepare_evidence(
            "公司代码审核怎么做", "aliexpress", classify=classify, retrieve=retrieve
        )
        self.assertEqual(bundle["status"], "blocked_unrelated_question")
        retrieve.assert_not_called()

    def test_uncertain_and_low_confidence_do_not_call_retrieve(self):
        for intent_result in (
            {"intent": "uncertain", "confidence": 0.55, "reason": "信息不足"},
            {"intent": "refund_after_sales", "confidence": 0.40, "reason": "不太确定"},
        ):
            with self.subTest(intent=intent_result["intent"]):
                classify = mock.Mock(return_value=dict(intent_result))
                retrieve = mock.Mock()
                bundle = peq.prepare_evidence(
                    "退款流程是什么",
                    "aliexpress",
                    classify=classify,
                    retrieve=retrieve,
                )
                self.assertEqual(bundle["status"], "blocked_intent_uncertain")
                retrieve.assert_not_called()


class EvidenceStageTests(unittest.TestCase):
    def prepare_ready(self, candidates, platform="aliexpress"):
        classify = mock.Mock(return_value=dict(GOOD_INTENT))
        retrieve = mock.Mock(return_value=list(candidates))
        bundle = peq.prepare_evidence(
            "退款流程是什么", platform, classify=classify, retrieve=retrieve
        )
        return bundle, retrieve

    def test_normal_request_passes_query_and_platform_to_retrieve(self):
        _, retrieve = self.prepare_ready([])
        retrieve.assert_called_once_with("退款流程是什么", "aliexpress")

    def test_valid_candidates_pass_gate_and_ready(self):
        bundle, _ = self.prepare_ready(
            [make_candidate("aliexpress", 6.0), make_candidate("aliexpress", 3.0)]
        )
        self.assertEqual(bundle["status"], "ready_for_grounding")
        self.assertTrue(bundle["evidence_gate"]["passed"])
        self.assertEqual(
            [item["citation_id"] for item in bundle["evidence"]], ["E1", "E2"]
        )
        for item in bundle["evidence"]:
            self.assertEqual(
                set(item),
                {
                    "citation_id",
                    "chunk_id",
                    "source_id",
                    "platform",
                    "headings",
                    "text",
                    "retrieve_score",
                    "rerank_score",
                },
            )
        self.assertEqual(bundle["intent"], "refund_after_sales")

    def test_empty_candidates_blocked_no_matching_source(self):
        bundle, retrieve = self.prepare_ready([])
        self.assertEqual(bundle["status"], "blocked_no_matching_source")
        retrieve.assert_called_once()
        self.assertEqual(bundle["evidence"], [])

    def test_infra_exception_propagates_unchanged(self):
        classify = mock.Mock(return_value=dict(GOOD_INTENT))
        retrieve = mock.Mock(side_effect=RuntimeError("qdrant connection refused"))
        with self.assertRaises(RuntimeError) as ctx:
            peq.prepare_evidence(
                "退款流程是什么", "aliexpress", classify=classify, retrieve=retrieve
            )
        self.assertEqual(str(ctx.exception), "qdrant connection refused")


class PurityAndCliTests(unittest.TestCase):
    def test_prepare_evidence_writes_no_output_file(self):
        classify = mock.Mock(return_value=dict(GOOD_INTENT))
        retrieve = mock.Mock(return_value=[make_candidate()])
        with tempfile.TemporaryDirectory() as tmp:
            saved_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    bundle = peq.prepare_evidence(
                        "退款流程是什么", "aliexpress",
                        classify=classify, retrieve=retrieve,
                    )
                self.assertEqual(buffer.getvalue(), "")
                self.assertEqual(bundle["status"], "ready_for_grounding")
                self.assertFalse(
                    (Path(tmp) / "output" / "evidence_bundle_qdrant.json").exists()
                )
            finally:
                os.chdir(saved_cwd)

    def test_main_keeps_original_file_output_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_cwd = Path.cwd()
            os.chdir(tmp)
            (Path(tmp) / "output").mkdir()  # 原 CLI 行为假定 output/ 已存在
            try:
                with (
                    mock.patch.object(peq, "classify_intent", return_value=dict(GOOD_INTENT)),
                    mock.patch.object(
                        peq, "retrieve_and_rank", return_value=[make_candidate()]
                    ),
                    mock.patch.object(
                        sys, "argv",
                        ["prepare_evidence_qdrant.py", "退款流程是什么", "--user-platform", "aliexpress"],
                    ),
                ):
                    noise = io.StringIO()
                    with redirect_stdout(noise):
                        peq.main()
                out_file = Path(tmp) / "output" / "evidence_bundle_qdrant.json"
                self.assertTrue(out_file.exists())
                bundle = json.loads(out_file.read_text(encoding="utf-8"))
                self.assertEqual(bundle["status"], "ready_for_grounding")
                self.assertEqual(bundle["query"], "退款流程是什么")
                self.assertEqual(bundle["requested_platform"], "aliexpress")
                self.assertEqual(len(bundle["evidence"]), 1)
            finally:
                os.chdir(saved_cwd)


if __name__ == "__main__":
    unittest.main()
