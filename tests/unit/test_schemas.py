"""schemas.py 单元测试：请求校验、响应序列化、与现有证据包兼容。"""

import json
import unittest

from pydantic import ValidationError

from customer_service_rag.schemas import (
    AnswerRequest,
    AnswerResponse,
    BundleStatus,
    EvidenceGateInfo,
    EvidenceItem,
    Platform,
)


def make_evidence_item(**overrides):
    """字段与 output/evidence_bundle.json 的 evidence[0] 一一对应。"""
    item = {
        "citation_id": "E1",
        "chunk_id": "chunk-001::merged-0010-0014",
        "source_id": "source-001",
        "platform": "aliexpress",
        "headings": ["速卖通平台商品退款与售后处理规则", "第三章 退款审核流程"],
        "text": "退款申请提交后，平台按照以下流程处理。",
        "retrieve_score": 0.8144617080688477,
        "rerank_score": 6.720974445343018,
    }
    item.update(overrides)
    return item


def make_gate_info(**overrides):
    info = {
        "passed": True,
        "min_rerank_score": 0.75,
        "checked_candidates": 3,
        "top_rerank_score": 6.720974445343018,
    }
    info.update(overrides)
    return info


def make_response(**overrides):
    payload = {
        "request_id": "req-0001",
        "status": "ready_for_grounding",
        "reason": "Evidence gate passed; answer stage must still verify",
        "entry_platform": "aliexpress",
        "requested_platform": "aliexpress",
        "intent": "refund_after_sales",
        "intent_confidence": 0.93,
        "intent_reason": "退款流程咨询",
        "evidence_gate": make_gate_info(),
        "evidence": [make_evidence_item()],
        "answer": "根据平台规则……（E1）",
        "used_citations": ["E1"],
    }
    payload.update(overrides)
    return payload


class TestAnswerRequest(unittest.TestCase):
    def test_valid_aliexpress_request(self):
        request = AnswerRequest(
            query="速卖通退款流程是什么", entry_platform="aliexpress"
        )
        self.assertEqual(request.entry_platform, Platform.ALIEXPRESS)

    def test_entry_platform_missing_allowed(self):
        request = AnswerRequest(query="退款流程是什么")
        self.assertIsNone(request.entry_platform)

    def test_temu_request_accepted(self):
        request = AnswerRequest(query="temu 退货政策", entry_platform="temu")
        self.assertEqual(request.entry_platform, Platform.TEMU)

    def test_unsupported_platform_rejected(self):
        with self.assertRaises(ValidationError):
            AnswerRequest(query="退款流程", entry_platform="amazon")

    def test_empty_and_whitespace_query_rejected(self):
        with self.assertRaises(ValidationError):
            AnswerRequest(query="")
        with self.assertRaises(ValidationError):
            AnswerRequest(query="   ")

    def test_query_is_stripped(self):
        request = AnswerRequest(query="  退款流程是什么  ")
        self.assertEqual(request.query, "退款流程是什么")

    def test_surrounding_whitespace_not_counted_in_max_length(self):
        raw = " " + ("退" * 2000) + " "
        request = AnswerRequest(query=raw)
        self.assertEqual(request.query, "退" * 2000)
        self.assertEqual(len(request.query), 2000)

    def test_non_string_query_rejected_without_attribute_error(self):
        with self.assertRaises(ValidationError):
            AnswerRequest(query=12345)

    def test_query_too_long_rejected(self):
        with self.assertRaises(ValidationError):
            AnswerRequest(query="退" * 2001)


class TestAnswerResponse(unittest.TestCase):
    def test_serializes_to_json(self):
        response = AnswerResponse(**make_response())
        data = json.loads(response.model_dump_json())
        self.assertEqual(data["status"], "ready_for_grounding")
        self.assertEqual(data["requested_platform"], "aliexpress")
        self.assertEqual(data["evidence"][0]["citation_id"], "E1")
        self.assertEqual(data["used_citations"], ["E1"])

    def test_evidence_field_compatible_with_current_bundle(self):
        raw = make_evidence_item()
        item = EvidenceItem(**raw)
        self.assertEqual(item.model_dump(), raw)
        gate = EvidenceGateInfo(**make_gate_info())
        self.assertTrue(gate.passed)

    def test_blocked_response_with_null_gate_and_empty_evidence(self):
        response = AnswerResponse(
            **make_response(
                status="blocked_missing_platform",
                entry_platform=None,
                requested_platform=None,
                intent=None,
                intent_confidence=None,
                intent_reason=None,
                evidence_gate=None,
                evidence=[],
                answer=None,
                used_citations=[],
            )
        )
        self.assertEqual(response.status, BundleStatus.BLOCKED_MISSING_PLATFORM)
        self.assertEqual(response.evidence, [])

    def test_invalid_status_rejected(self):
        with self.assertRaises(ValidationError):
            AnswerResponse(**make_response(status="blocked_not_a_real_status"))


if __name__ == "__main__":
    unittest.main()
