"""确定性平台门控（platform_gate.resolve_platform）单元测试。

历史背景：业务相关性关键词判断（STORE_DOMAIN_KEYWORDS /
blocked_unrelated_question）已迁移到 DeepSeek 意图分类器
（intent_classifier.py + test_platform_pipeline.py），
因此只要能确定唯一平台，任意问题都会 ``platform_resolved`` 放行。

运行方式：
    python -m unittest test_platform_gate -v
"""

import unittest

from acceptance_platform import CASES
from platform_gate import detect_platforms_in_query, resolve_platform


class ResolvePlatformTests(unittest.TestCase):
    """resolve_platform 纯逻辑测试（无网络、无模型）。"""

    def test_invalid_entry_platform_blocked(self):
        for value in ("suning", "amazon", "ALIEXPRESS_APP"):
            with self.subTest(value=value):
                result = resolve_platform("退款规则", value)
                self.assertEqual(
                    result["status"], "blocked_invalid_entry_platform"
                )
                self.assertIsNone(result["requested_platform"])
                self.assertIsNone(result["entry_platform"])

    def test_invalid_entry_platform_not_degraded_to_query_detection(self):
        # 非法进入平台 + 问题带平台名：不得退化为问题识别。
        result = resolve_platform("速卖通退款规则", "suning")
        self.assertEqual(result["status"], "blocked_invalid_entry_platform")

    def test_blank_entry_is_empty_value(self):
        # 纯空白进入平台视为空值（不视为非法）。
        result = resolve_platform("退款规则", "   ")
        self.assertEqual(result["status"], "blocked_missing_platform")

    def test_single_platform_questions_pass_to_intent_classification(self):
        # 旧的关键词拦截（unrelated / 泛化词）已由意图分类器接管，
        # 只要能确定唯一平台，本层一律放行。
        cases = [
            ("你是什么模型", "aliexpress", "aliexpress"),
            ("今天天气怎么样", "temu", "temu"),
            ("Temu是什么模型", "", "temu"),
            ("速卖通老板是谁", "", "aliexpress"),
            ("招聘流程是什么", "aliexpress", "aliexpress"),
            ("足球比赛规则是什么", "temu", "temu"),
            ("Python运行条件是什么", "aliexpress", "aliexpress"),
        ]
        for query, entry, expected_platform in cases:
            with self.subTest(query=query, entry=entry or "<empty>"):
                result = resolve_platform(query, entry)
                self.assertEqual(result["status"], "platform_resolved")
                self.assertEqual(
                    result["requested_platform"], expected_platform
                )

    def test_detect_platforms_in_query(self):
        # 平台名称识别能力保持不变。
        self.assertEqual(
            detect_platforms_in_query("速卖通和Temu"), {"aliexpress", "temu"}
        )
        self.assertEqual(
            detect_platforms_in_query("AliExpress"), {"aliexpress"}
        )
        self.assertEqual(detect_platforms_in_query("退款规则"), set())

    def test_acceptance_cases_still_pass(self):
        for name, user_platform, query, expected_status, expected_platform in CASES:
            with self.subTest(case=name):
                result = resolve_platform(query, user_platform)
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(result["requested_platform"], expected_platform)


if __name__ == "__main__":
    unittest.main()
