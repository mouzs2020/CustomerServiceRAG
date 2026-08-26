"""平台门控与证据包管线测试。

覆盖：
1. 平台名称只用于识别平台，不单独构成“退款售后相关”。
2. 非法 user_platform 返回 blocked_invalid_entry_platform。
3. 原有 11 个验收案例回归（从 acceptance_platform 复用）。
4. 所有 blocked 状态都不会调用 retrieve_and_rank（Mock）。
5. 对照：platform_resolved 时 retrieve_and_rank 会被调用。

运行方式：
    python -m unittest test_platform_gate -v
"""

import json
import sys
import unittest
from unittest import mock

import prepare_evidence_qdrant as peq
from acceptance_platform import CASES
from platform_gate import resolve_platform


class ResolvePlatformTests(unittest.TestCase):
    """resolve_platform 纯逻辑测试（不加载任何模型）。"""

    def test_platform_name_alone_is_not_store_related(self):
        cases = [
            ("你是什么模型", "aliexpress", "blocked_unrelated_question"),
            ("今天天气怎么样", "temu", "blocked_unrelated_question"),
            ("Temu是什么模型", "", "blocked_unrelated_question"),
            ("速卖通老板是谁", "", "blocked_unrelated_question"),
        ]
        for query, entry, expected in cases:
            with self.subTest(query=query, entry=entry or "<empty>"):
                self.assertEqual(
                    resolve_platform(query, entry)["status"], expected
                )

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

    def test_acceptance_cases_still_pass(self):
        for name, user_platform, query, expected_status, expected_platform in CASES:
            with self.subTest(case=name):
                result = resolve_platform(query, user_platform)
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(result["requested_platform"], expected_platform)


class RetrieveNotCalledOnBlockedTests(unittest.TestCase):
    """所有 blocked 状态都不得调用 retrieve_and_rank。"""

    BLOCKED_CASES = [
        # (argv, 期望 status)
        (
            ["你是什么模型", "--user-platform", "aliexpress"],
            "blocked_unrelated_question",
        ),
        (
            ["今天天气怎么样", "--user-platform", "temu"],
            "blocked_unrelated_question",
        ),
        (["Temu是什么模型"], "blocked_unrelated_question"),
        (["速卖通老板是谁"], "blocked_unrelated_question"),
        (["退款规则"], "blocked_missing_platform"),
        (
            ["退款规则", "--user-platform", "suning"],
            "blocked_invalid_entry_platform",
        ),
        (
            ["速卖通和Temu的退款规则一样吗"],
            "blocked_multiple_platforms",
        ),
        (
            ["Temu退款规则", "--user-platform", "aliexpress"],
            "blocked_platform_conflict",
        ),
    ]

    def run_main(self, argv):
        full_argv = ["prepare_evidence_qdrant.py", *argv]
        with (
            mock.patch.object(peq, "retrieve_and_rank") as retrieve_mock,
            # WindowsPath 的方法只读，patch 模块级 OUTPUT_PATH 本身。
            mock.patch.object(peq, "OUTPUT_PATH") as output_path_mock,
            mock.patch.object(sys, "argv", full_argv),
        ):
            peq.main()
        return retrieve_mock, output_path_mock

    def test_blocked_statuses_skip_retrieve(self):
        for argv, expected_status in self.BLOCKED_CASES:
            with self.subTest(argv=argv):
                retrieve_mock, output_path_mock = self.run_main(argv)
                retrieve_mock.assert_not_called()
                bundle = json.loads(
                    output_path_mock.write_text.call_args.args[0]
                )
                self.assertEqual(bundle["status"], expected_status)
                self.assertEqual(bundle["evidence"], [])

    def test_resolved_status_calls_retrieve(self):
        # 对照：platform_resolved 时 retrieve_and_rank 必须被调用，
        # 证明测试不是“永远不调用”。
        retrieve_mock, _ = self.run_main(
            ["退款规则", "--user-platform", "aliexpress"]
        )
        retrieve_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
