"""answer_with_citations_qdrant 重构后的单元测试（无网络、无真实模型）。"""

import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODULE_NAME = "customer_service_rag.answer_with_citations_qdrant"
answer = importlib.import_module(MODULE_NAME)


def make_bundle(status="ready_for_grounding", n=2):
    evidence = [
        {
            "citation_id": f"E{index + 1}",
            "chunk_id": f"c{index + 1}",
            "source_id": f"s{index + 1}",
            "platform": "aliexpress",
            "headings": ["退款规则"],
            "text": f"证据文本{index + 1}",
        }
        for index in range(n)
    ]
    return {"status": status, "query": "退款流程是什么", "evidence": evidence}


def fake_response(content, status_code=200):
    response = mock.Mock()
    response.status_code = status_code
    response.text = "upstream error body"
    response.is_error = status_code >= 400
    response.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return response


class ImportSideEffectTests(unittest.TestCase):
    def test_import_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved_cwd = Path.cwd()
            os.chdir(tmp)  # 无 output/：任何顶层文件读取都会报错
            try:
                sys.modules.pop(MODULE_NAME, None)
                buffer = io.StringIO()
                with mock.patch.object(
                    httpx, "post", side_effect=AssertionError("network")
                ):
                    with redirect_stdout(buffer):
                        try:
                            module = importlib.import_module(MODULE_NAME)
                        except SystemExit as exc:
                            self.fail(f"import raised SystemExit: {exc}")
                self.assertEqual(buffer.getvalue(), "")
                self.assertTrue(callable(module.generate_answer))
                self.assertTrue(callable(module.main))
                self.assertFalse(
                    (Path(tmp) / "output" / "answer_qdrant.json").exists()
                )
            finally:
                os.chdir(saved_cwd)
                sys.modules.pop(MODULE_NAME, None)


class GenerateAnswerTests(unittest.TestCase):
    def test_friendly_blocked_does_not_call_post(self):
        post = mock.Mock()
        result = answer.generate_answer(
            make_bundle("blocked_intent_uncertain"), post=post
        )
        self.assertEqual(result["model"], "fallback")
        self.assertEqual(
            result["answer"],
            answer.FRIENDLY_BLOCKED_ANSWERS["blocked_intent_uncertain"],
        )
        self.assertEqual(result["used_citations"], [])
        post.assert_not_called()

    def test_plain_blocked_does_not_call_post(self):
        post = mock.Mock()
        with self.assertRaises(ValueError):
            answer.generate_answer(
                make_bundle("blocked_low_relevance"), post=post
            )
        post.assert_not_called()

    def test_ready_uses_injected_post_with_test_key(self):
        post = mock.Mock(return_value=fake_response("按规则审核。[E1]"))
        with mock.patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-key"}
        ):
            result = answer.generate_answer(make_bundle(), post=post)
        self.assertEqual(result["used_citations"], ["E1"])
        self.assertEqual(result["query"], "退款流程是什么")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-key",
        )

    def test_missing_api_key_rejected_even_with_injected_post(self):
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "DEEPSEEK_API_KEY"
        }
        post = mock.Mock(return_value=fake_response("按规则审核。[E1]"))
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                answer.generate_answer(make_bundle(), post=post)
        post.assert_not_called()

    def test_unknown_citation_e9_rejected(self):
        post = mock.Mock(return_value=fake_response("这与规则无关。[E9]"))
        with mock.patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-key"}
        ):
            with self.assertRaises(ValueError):
                answer.generate_answer(make_bundle(), post=post)
        post.assert_called_once()

    def test_answer_without_citation_rejected(self):
        post = mock.Mock(return_value=fake_response("可以直接退款，无需依据。"))
        with mock.patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-key"}
        ):
            with self.assertRaises(ValueError):
                answer.generate_answer(make_bundle(), post=post)

    def test_fallback_answer_without_citation_allowed(self):
        post = mock.Mock(return_value=fake_response(answer.FALLBACK))
        with mock.patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-key"}
        ):
            result = answer.generate_answer(make_bundle(), post=post)
        self.assertEqual(result["used_citations"], [])
        self.assertEqual(result["answer"], answer.FALLBACK)


class PromptContractTests(unittest.TestCase):
    """Prompt 契约：证据部分覆盖时必须回答已支持部分，禁止整体拒答。"""

    TEMU_QUERY = "Temu对符合条件的小额订单如何退款？"
    TEMU_E1_TEXT = (
        "对于符合条件的小额订单：平台可能直接批准退款。"
        "无需消费者：1. 寄回商品；"
    )

    def make_temu_bundle(self):
        return {
            "status": "ready_for_grounding",
            "query": self.TEMU_QUERY,
            "evidence": [
                {
                    "citation_id": "E1",
                    "chunk_id": "c1",
                    "source_id": "s1",
                    "platform": "temu",
                    "headings": ["退款政策"],
                    "text": self.TEMU_E1_TEXT,
                }
            ],
        }

    def call_with_post(self, bundle, content):
        post = mock.Mock(return_value=fake_response(content))
        with mock.patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-key"}
        ):
            result = answer.generate_answer(bundle, post=post)
        return result, post

    def test_temu_partial_evidence_answer_is_accepted(self):
        expected = (
            "对于符合条件的小额订单，平台可能直接批准退款，"
            "无需消费者寄回商品。[E1]"
        )
        result, post = self.call_with_post(
            self.make_temu_bundle(), expected
        )
        self.assertEqual(result["answer"], expected)
        self.assertEqual(result["used_citations"], ["E1"])
        self.assertNotEqual(result["model"], "fallback")
        post.assert_called_once()

    def test_system_prompt_contains_partial_answer_contract(self):
        _, post = self.call_with_post(
            self.make_temu_bundle(), "按规则直接退款。[E1]"
        )
        system_prompt = post.call_args.kwargs["json"]["messages"][0][
            "content"
        ]
        self.assertIn(
            "只要能直接支持问题的部分答案，就必须回答已支持的部分",
            system_prompt,
        )
        self.assertIn(
            "不得因为证据没有提供问题的全部细节",
            system_prompt,
        )
        self.assertIn("完全不能支持问题的任何答案", system_prompt)
        self.assertIn(answer.FALLBACK, system_prompt)

    def test_user_prompt_contains_temu_query_and_evidence(self):
        _, post = self.call_with_post(
            self.make_temu_bundle(), "按规则直接退款。[E1]"
        )
        user_prompt = post.call_args.kwargs["json"]["messages"][1][
            "content"
        ]
        self.assertIn(self.TEMU_QUERY, user_prompt)
        self.assertIn(self.TEMU_E1_TEXT, user_prompt)
        self.assertIn("[E1]", user_prompt)

    def test_citation_validation_still_enforced(self):
        with self.assertRaises(ValueError):
            self.call_with_post(
                self.make_temu_bundle(), "直接退款。[E9]"
            )


class MainTests(unittest.TestCase):
    def run_main_in_tmp(self, bundle):
        tmp = tempfile.TemporaryDirectory()
        out_dir = Path(tmp.name) / "output"
        out_dir.mkdir()
        (out_dir / "evidence_bundle_qdrant.json").write_text(
            json.dumps(bundle, ensure_ascii=False), encoding="utf-8"
        )
        saved_cwd = Path.cwd()
        os.chdir(tmp.name)
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                try:
                    answer.main()
                    exit_code = None
                except SystemExit as exc:
                    exit_code = exc.code
            data = json.loads(
                (out_dir / "answer_qdrant.json").read_text(encoding="utf-8")
            )
            return data, exit_code
        finally:
            os.chdir(saved_cwd)
            tmp.cleanup()

    def test_cli_main_friendly_reads_and_writes_files(self):
        with mock.patch.object(
            httpx, "post", side_effect=AssertionError("network")
        ):
            data, exit_code = self.run_main_in_tmp(
                make_bundle("blocked_intent_uncertain")
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(data["model"], "fallback")
        self.assertEqual(data["used_citations"], [])

    def test_cli_main_ready_uses_mocked_deepseek(self):
        with mock.patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-key"}
        ):
            with mock.patch.object(
                httpx,
                "post",
                return_value=fake_response("按规则审核。[E1]"),
            ) as post:
                data, exit_code = self.run_main_in_tmp(make_bundle())
        self.assertIsNone(exit_code)
        self.assertEqual(data["used_citations"], ["E1"])
        self.assertNotEqual(data["model"], "fallback")
        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
