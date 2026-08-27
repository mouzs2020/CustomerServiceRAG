"""管线级 Mock 测试：确定性门控 → 意图分类 → 检索 → 证据门。

覆盖项目规范的 15 个验收场景及补充边界：
- DeepSeek / Embedding / Qdrant / Reranker 全部 Mock，
  运行不需要 API Key、不触网、不加载模型、不写 output 文件。
- 所有 blocked 状态都断言后续组件未被调用。
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import intent_classifier
import prepare_evidence_qdrant as peq
from intent_classifier import IntentClassifierError


ALIEXPRESS_QUERY_OK = ["退款流程是什么", "--user-platform", "aliexpress"]

GOOD_INTENT = {"intent": "refund_after_sales", "confidence": 0.95, "reason": "退款咨询"}
UNRELATED_INTENT = {"intent": "unrelated", "confidence": 0.98, "reason": "与售后无关"}
LOW_CONFIDENT_REFUND = {
    "intent": "refund_after_sales",
    "confidence": 0.40,
    "reason": "不太确定",
}
UNCERTAIN_INTENT = {"intent": "uncertain", "confidence": 0.55, "reason": "信息不足"}


def make_candidate(platform: str, rerank_score: float, chunk_id: str) -> dict:
    """构造一个 retrieve_and_rank 形状的候选。"""
    return {
        "record": {
            "chunk_id": chunk_id,
            "source_id": f"{chunk_id}:src-hash",
            "document_id": chunk_id,
            "platform": platform,
            "headings": ["标题"],
            "text": "示例证据文本",
            "contextualized_text": "示例上下文文本",
        },
        "retrieve_rank": 1,
        "retrieve_score": 0.80,
        "rerank_score": rerank_score,
    }


class PipelineMockTests(unittest.TestCase):
    """peq.main() 全链路 Mock 测试。"""

    def run_main(
        self,
        argv: list[str],
        *,
        intent_return: dict | None = None,
        intent_side_effect: object = None,
        candidates: list[dict] | None = None,
    ) -> tuple[mock.Mock, mock.Mock, dict]:
        full_argv = ["prepare_evidence_qdrant.py", *argv]
        with (
            mock.patch.object(peq, "classify_intent") as classify_mock,
            mock.patch.object(peq, "retrieve_and_rank") as retrieve_mock,
            # WindowsPath 方法只读，patch 模块级 OUTPUT_PATH 本身。
            mock.patch.object(peq, "OUTPUT_PATH") as output_path_mock,
            mock.patch.object(sys, "argv", full_argv),
        ):
            classify_mock.return_value = (
                intent_return if intent_return is not None else GOOD_INTENT
            )
            classify_mock.side_effect = intent_side_effect
            retrieve_mock.return_value = candidates or []
            peq.main()

        bundle = json.loads(output_path_mock.write_text.call_args.args[0])
        return classify_mock, retrieve_mock, bundle

    # ---- 场景 1~3：售后问题放行到检索 ----

    def test_01_aliexpress_refund_flow_allowed_to_retrieve(self):
        candidates = [
            make_candidate("aliexpress", 6.0, "a1"),
            make_candidate("aliexpress", 4.0, "a2"),
        ]
        classify_mock, retrieve_mock, bundle = self.run_main(
            ALIEXPRESS_QUERY_OK, candidates=candidates
        )
        classify_mock.assert_called_once_with("退款流程是什么")
        retrieve_mock.assert_called_once_with("退款流程是什么", "aliexpress")
        self.assertEqual(bundle["status"], "ready_for_grounding")
        self.assertTrue(bundle["evidence_gate"]["passed"])
        self.assertEqual(bundle["intent"], "refund_after_sales")
        self.assertEqual(
            [item["platform"] for item in bundle["evidence"]],
            ["aliexpress", "aliexpress"],
        )

    def test_02_temu_package_lost_allowed_to_retrieve(self):
        argv = ["我的包裹一直没到怎么办", "--user-platform", "temu"]
        candidates = [make_candidate("temu", 5.0, "t1")]
        _, retrieve_mock, bundle = self.run_main(argv, candidates=candidates)
        retrieve_mock.assert_called_once_with("我的包裹一直没到怎么办", "temu")
        self.assertEqual(bundle["status"], "ready_for_grounding")
        self.assertEqual(bundle["requested_platform"], "temu")

    def test_03_temu_broken_item_allowed_to_retrieve(self):
        argv = ["收到的东西坏了怎么办", "--user-platform", "temu"]
        candidates = [make_candidate("temu", 5.0, "t2")]
        _, _, bundle = self.run_main(argv, candidates=candidates)
        self.assertEqual(bundle["status"], "ready_for_grounding")
        self.assertEqual(
            bundle["evidence"][0]["platform"], "temu"
        )

    # ---- 场景 4~6：意图 unrelated，检索不被调用 ----

    def test_04_code_review_unrelated(self):
        argv = ["公司代码审核怎么做", "--user-platform", "aliexpress"]
        _, retrieve_mock, bundle = self.run_main(
            argv, intent_return=UNRELATED_INTENT
        )
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_unrelated_question")
        self.assertEqual(bundle["evidence"], [])
        self.assertEqual(bundle["intent"], "unrelated")

    def test_05_work_injury_compensation_unrelated(self):
        argv = ["工伤赔付规则是什么", "--user-platform", "temu"]
        _, retrieve_mock, bundle = self.run_main(
            argv, intent_return=UNRELATED_INTENT
        )
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_unrelated_question")

    def test_06_recruit_dispute_unrelated(self):
        argv = ["招聘纠纷如何处理", "--user-platform", "aliexpress"]
        _, retrieve_mock, bundle = self.run_main(
            argv, intent_return=UNRELATED_INTENT
        )
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_unrelated_question")

    # ---- 场景 7~8：门控 blocked 跳过意图分类与检索 ----

    def test_07_platform_conflict_skips_intent_and_retrieval(self):
        argv = ["Temu退款规则", "--user-platform", "aliexpress"]
        classify_mock, retrieve_mock, bundle = self.run_main(argv)
        classify_mock.assert_not_called()
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_platform_conflict")
        self.assertIsNone(bundle["intent"])
        self.assertEqual(bundle["evidence"], [])

    def test_08_missing_platform_skips_intent_and_retrieval(self):
        classify_mock, retrieve_mock, bundle = self.run_main(["退款规则"])
        classify_mock.assert_not_called()
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_missing_platform")

    # ---- 场景 9：uncertain 拦截在检索之前 ----

    def test_09_uncertain_intent_blocked(self):
        _, retrieve_mock, bundle = self.run_main(
            ALIEXPRESS_QUERY_OK, intent_return=UNCERTAIN_INTENT
        )
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_intent_uncertain")
        self.assertEqual(bundle["evidence"], [])

    def test_09b_low_confidence_refund_blocked_as_uncertain(self):
        _, retrieve_mock, bundle = self.run_main(
            ALIEXPRESS_QUERY_OK, intent_return=LOW_CONFIDENT_REFUND
        )
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_intent_uncertain")

    # ---- 场景 10~11：分类器错误拦截在检索之前 ----

    def test_10_invalid_json_maps_to_classifier_error(self):
        _, retrieve_mock, bundle = self.run_main(
            ALIEXPRESS_QUERY_OK,
            intent_side_effect=IntentClassifierError(
                "Classifier output is not valid JSON"
            ),
        )
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_intent_classifier_error")
        self.assertIsNone(bundle["intent"])
        self.assertEqual(bundle["evidence"], [])

    def test_11_timeout_maps_to_classifier_error(self):
        _, retrieve_mock, bundle = self.run_main(
            ALIEXPRESS_QUERY_OK,
            intent_side_effect=IntentClassifierError(
                "Intent classifier request failed: timed out"
            ),
        )
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_intent_classifier_error")

    # ---- 场景 12~13：证据门 blocked ----

    def test_12_top_rerank_score_below_threshold_blocked(self):
        candidates = [make_candidate("aliexpress", 0.50, "a-low")]
        _, retrieve_mock, bundle = self.run_main(
            ALIEXPRESS_QUERY_OK, candidates=candidates
        )
        retrieve_mock.assert_called_once()  # 检索发生了，但证据不可靠。
        self.assertEqual(bundle["status"], "blocked_low_relevance")
        self.assertEqual(bundle["evidence"], [])
        self.assertFalse(bundle["evidence_gate"]["passed"])

    def test_12b_nonfinite_scores_blocked_as_low_relevance(self):
        candidates = [make_candidate("aliexpress", float("nan"), "a-nan")]
        _, _, bundle = self.run_main(ALIEXPRESS_QUERY_OK, candidates=candidates)
        self.assertEqual(bundle["status"], "blocked_low_relevance")
        self.assertEqual(bundle["evidence"], [])

    def test_13_other_platform_evidence_mismatch(self):
        candidates = [make_candidate("temu", 6.0, "t-cross")]
        _, retrieve_mock, bundle = self.run_main(
            ALIEXPRESS_QUERY_OK, candidates=candidates
        )
        retrieve_mock.assert_called_once()
        self.assertEqual(bundle["status"], "blocked_platform_evidence_mismatch")
        self.assertEqual(bundle["evidence"], [])

    # ---- 场景 14~15：正常路径证据平台纯净 ----

    def test_14_normal_aliexpress_evidence_all_aliexpress(self):
        candidates = [
            make_candidate("aliexpress", 6.0, "a-hi"),
            make_candidate("aliexpress", 3.0, "a-mid"),
            make_candidate("aliexpress", 0.10, "a-noise"),
        ]
        _, _, bundle = self.run_main(ALIEXPRESS_QUERY_OK, candidates=candidates)
        self.assertEqual(bundle["status"], "ready_for_grounding")
        self.assertTrue(bundle["evidence"])
        self.assertTrue(
            all(item["platform"] == "aliexpress" for item in bundle["evidence"])
        )
        # 低于阈值的候选不应进入证据包。
        chunk_ids = [item["chunk_id"] for item in bundle["evidence"]]
        self.assertNotIn("a-noise", chunk_ids)

    def test_15_normal_temu_evidence_all_temu(self):
        argv = ["怎么申请退货退款", "--user-platform", "temu"]
        candidates = [
            make_candidate("temu", 6.0, "t1"),
            make_candidate("temu", 5.5, "t2"),
        ]
        _, _, bundle = self.run_main(argv, candidates=candidates)
        self.assertEqual(bundle["status"], "ready_for_grounding")
        self.assertLessEqual(len(bundle["evidence"]), peq.TOP_K_RERANK)
        self.assertTrue(
            all(item["platform"] == "temu" for item in bundle["evidence"])
        )

    # ---- 补充边界：其余门控 blocked 全链路跳过 ----

    def test_extra_invalid_entry_blocks_all_downstream(self):
        classify_mock, retrieve_mock, bundle = self.run_main(
            ["退款规则", "--user-platform", "suning"]
        )
        classify_mock.assert_not_called()
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_invalid_entry_platform")

    def test_extra_multiple_platforms_blocks_all_downstream(self):
        classify_mock, retrieve_mock, bundle = self.run_main(
            ["速卖通和Temu的退款规则一样吗"]
        )
        classify_mock.assert_not_called()
        retrieve_mock.assert_not_called()
        self.assertEqual(bundle["status"], "blocked_multiple_platforms")


class IntentClassifierUnitTests(unittest.TestCase):
    """intent_classifier 自身行为；httpx 全程 Mock，不触网。"""

    def classify_with_content(self, content: str) -> tuple[dict, mock.Mock]:
        response = mock.Mock()
        response.is_error = False
        response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        with (
            mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
            mock.patch.object(
                intent_classifier.httpx, "post", return_value=response
            ) as post_mock,
        ):
            result = intent_classifier.classify_intent("退款流程是什么")
        return result, post_mock

    def test_valid_plain_json(self):
        result, _ = self.classify_with_content(
            '{"intent": "refund_after_sales", "confidence": 0.9, "reason": "ok"}'
        )
        self.assertEqual(result["intent"], "refund_after_sales")
        self.assertEqual(result["confidence"], 0.9)

    def test_valid_fenced_json(self):
        fenced = (
            '```json\n'
            '{"intent": "unrelated", "confidence": 0.99, "reason": "闲聊"}\n'
            '```'
        )
        result, _ = self.classify_with_content(fenced)
        self.assertEqual(result["intent"], "unrelated")

    def test_timeout_raises_classifier_error(self):
        with (
            mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
            mock.patch.object(
                intent_classifier.httpx,
                "post",
                side_effect=intent_classifier.httpx.TimeoutException("timed out"),
            ),
        ):
            with self.assertRaises(IntentClassifierError):
                intent_classifier.classify_intent("退款流程是什么")

    def test_http_error_raises_classifier_error(self):
        response = mock.Mock()
        response.is_error = True
        response.status_code = 500
        response.text = "server exploded"
        with (
            mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
            mock.patch.object(
                intent_classifier.httpx, "post", return_value=response
            ),
        ):
            with self.assertRaises(IntentClassifierError):
                intent_classifier.classify_intent("退款流程是什么")

    def test_missing_api_key_raises_without_network(self):
        # 未 patch httpx.post：若实现先触网会直接失败，
        # 断言实现必须先检查配置。
        env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(IntentClassifierError):
                intent_classifier.classify_intent("退款流程是什么")

    def test_request_uses_env_model_key_and_timeout(self):
        result, post_mock = self._classify_with_env_model()
        self.assertEqual(result["intent"], "refund_after_sales")
        kwargs = post_mock.call_args.kwargs
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "test-intent-model")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["timeout"], intent_classifier.INTENT_TIMEOUT_SECONDS)
        self.assertGreater(kwargs["timeout"], 0)
        headers = kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer fake-key-for-test")

    def _classify_with_env_model(self) -> tuple[dict, mock.Mock]:
        response = mock.Mock()
        response.is_error = False
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"intent": "refund_after_sales", '
                            '"confidence": 0.88, "reason": "r"}'
                        )
                    }
                }
            ]
        }
        with (
            mock.patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "fake-key-for-test",
                    "DEEPSEEK_INTENT_MODEL": "test-intent-model",
                },
            ),
            mock.patch.object(
                intent_classifier.httpx, "post", return_value=response
            ) as post_mock,
        ):
            result = intent_classifier.classify_intent("退款流程是什么")
        return result, post_mock

    def test_validate_intent_payload_rejects_bad_outputs(self):
        validate = intent_classifier.validate_intent_payload
        cases = [
            ({"confidence": 0.5, "reason": "r"}, "missing intent"),
            ({}, "missing all fields"),
            ({"intent": "chit_chat", "confidence": 0.5, "reason": "r"}, "bad enum"),
            ({"intent": "unrelated", "confidence": "high", "reason": "r"}, "str confidence"),
            ({"intent": "unrelated", "confidence": True, "reason": "r"}, "bool confidence"),
            ({"intent": "unrelated", "confidence": 1.5, "reason": "r"}, "out of range"),
            ({"intent": "unrelated", "confidence": -0.1, "reason": "r"}, "negative"),
            ({"intent": "unrelated", "confidence": 0.5}, "missing reason"),
            ("not a dict", "non-object"),
            ([1, 2], "list payload"),
        ]
        for payload, label in cases:
            with self.subTest(case=label):
                with self.assertRaises(IntentClassifierError):
                    validate(payload)

    def test_validate_intent_payload_accepts_boundaries(self):
        validate = intent_classifier.validate_intent_payload
        zero = validate({"intent": "uncertain", "confidence": 0, "reason": " r "})
        self.assertEqual(zero, {"intent": "uncertain", "confidence": 0.0, "reason": "r"})
        one = validate({"intent": "unrelated", "confidence": 1, "reason": ""})
        self.assertEqual(one["confidence"], 1.0)


class EvidenceGateUnitTests(unittest.TestCase):
    """evaluate_evidence 独立校验逻辑。"""

    def run_evaluate(self, requested_platform, candidates):
        with mock.patch.dict(os.environ, {"MIN_RERANK_SCORE": "0.75"}):
            return peq.evaluate_evidence(requested_platform, candidates)

    def test_empty_candidates_no_matching_source(self):
        status, reason, evidence, gate = self.run_evaluate("aliexpress", [])
        self.assertEqual(status, "blocked_no_matching_source")
        self.assertEqual(evidence, [])
        self.assertFalse(gate["passed"])

    def test_top_k_limit_enforced(self):
        candidates = [
            make_candidate("aliexpress", 9.0 - index, f"a{index}")
            for index in range(5)
        ]
        status, _, evidence, gate = self.run_evaluate("aliexpress", candidates)
        self.assertEqual(status, "ready_for_grounding")
        self.assertLessEqual(len(evidence), peq.TOP_K_RERANK)
        self.assertEqual(gate["checked_candidates"], 5)
        # 按分数降序取前 TOP_K_RERANK 条。
        self.assertEqual(evidence[0]["chunk_id"], "a0")
        self.assertEqual(evidence[0]["citation_id"], "E1")

    def test_min_score_env_override(self):
        candidates = [make_candidate("aliexpress", 0.60, "a-env")]
        # 高于覆盖后的阈值 -> 通过。
        with mock.patch.dict(os.environ, {"MIN_RERANK_SCORE": "0.5"}):
            status, _, evidence, _ = peq.evaluate_evidence(
                "aliexpress", candidates
            )
        self.assertEqual(status, "ready_for_grounding")
        self.assertEqual(evidence[0]["chunk_id"], "a-env")
        # 默认阈值之下 -> 低相关拦截。
        with mock.patch.dict(os.environ, {"MIN_RERANK_SCORE": "0.75"}):
            status_default, _, evidence_default, _ = peq.evaluate_evidence(
                "aliexpress", candidates
            )
        self.assertEqual(status_default, "blocked_low_relevance")
        self.assertEqual(evidence_default, [])

    def test_threshold_boundary_is_inclusive(self):
        candidate = make_candidate("aliexpress", 0.75, "a-edge")
        status, _, evidence, _ = self.run_evaluate("aliexpress", [candidate])
        self.assertEqual(status, "ready_for_grounding")
        self.assertEqual(evidence[0]["chunk_id"], "a-edge")


class AnswerScriptRoutingTests(unittest.TestCase):
    """answer_with_citations_qdrant 的固定话术路由（静态源码断言）。

    说明：该脚本是“读证据包 -> 执行”的脚本式模块，直接 import 会
    触发顶层流程并抛出 SystemExit，因此这里只做源码级路由断言，
    避免任何副作用。
    """

    SOURCE_PATH = Path(__file__).resolve().parent / "answer_with_citations_qdrant.py"

    def get_source(self) -> str:
        return self.SOURCE_PATH.read_text(encoding="utf-8")

    def test_uncertain_status_has_friendly_answer(self):
        source = self.get_source()
        # 导入两个固定话术并用字典映射两个友好 blocked 状态。
        self.assertIn(
            "from platform_gate import UNRELATED_FALLBACK, UNCERTAIN_FALLBACK",
            source,
        )
        self.assertIn('"blocked_unrelated_question": UNRELATED_FALLBACK', source)
        self.assertIn('"blocked_intent_uncertain": UNCERTAIN_FALLBACK', source)

    def test_other_blocked_statuses_are_not_friendly(self):
        source = self.get_source()
        for status in (
            "blocked_low_relevance",
            "blocked_platform_evidence_mismatch",
            "blocked_no_matching_source",
            "blocked_intent_classifier_error",
        ):
            with self.subTest(status=status):
                # 这些状态必须走 “Evidence gate blocked” 的报错路径，
                # 而不是出现在友好话术映射里。
                self.assertNotIn(f'"{status}"', source.split("FRIENDLY_BLOCKED_ANSWERS = {")[1].split("}")[0])


if __name__ == "__main__":
    unittest.main()
