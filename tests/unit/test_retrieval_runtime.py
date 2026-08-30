"""retrieval_runtime 线程安全惰性缓存的单元测试（不加载真实模型）。

通过 loader 注入替代真实 sentence_transformers / transformers，
全部用例离线、纯内存、无网络。
"""

import sys
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_service_rag import retrieval_runtime


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        retrieval_runtime.clear_model_caches()

    def tearDown(self):
        retrieval_runtime.clear_model_caches()

    def test_same_model_id_loaded_once(self):
        calls = []

        def loader(model_id):
            calls.append(model_id)
            return f"model:{model_id}"

        first = retrieval_runtime.get_embedding_model("m1", loader=loader)
        second = retrieval_runtime.get_embedding_model("m1", loader=loader)

        self.assertEqual(first, second)
        self.assertEqual(calls, ["m1"])

    def test_different_model_id_not_reused(self):
        calls = []

        def loader(model_id):
            calls.append(model_id)
            return f"model:{model_id}"

        retrieval_runtime.get_embedding_model("m1", loader=loader)
        retrieval_runtime.get_embedding_model("m2", loader=loader)

        self.assertEqual(calls, ["m1", "m2"])

    def test_load_failure_not_cached_next_call_retries(self):
        calls = []

        def loader(model_id):
            calls.append(model_id)
            if len(calls) == 1:
                raise RuntimeError("load failed")
            return "ok-model"

        with self.assertRaises(RuntimeError):
            retrieval_runtime.get_embedding_model("m1", loader=loader)

        model = retrieval_runtime.get_embedding_model("m1", loader=loader)
        self.assertEqual(model, "ok-model")
        self.assertEqual(calls, ["m1", "m1"])

    def test_concurrent_first_call_loads_once(self):
        calls = []
        calls_lock = threading.Lock()

        def loader(model_id):
            with calls_lock:
                calls.append(model_id)
            time.sleep(0.2)  # 放大并发竞争窗口
            return f"model:{model_id}"

        results = []
        results_lock = threading.Lock()

        def worker():
            model = retrieval_runtime.get_embedding_model(
                "m-cc", loader=loader
            )
            with results_lock:
                results.append(model)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(len(results), 4)
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(calls, ["m-cc"])

    def test_clear_forces_reload(self):
        calls = []

        def loader(model_id):
            calls.append(model_id)
            return f"model:{model_id}"

        retrieval_runtime.get_embedding_model("m1", loader=loader)
        retrieval_runtime.clear_model_caches()
        retrieval_runtime.get_embedding_model("m1", loader=loader)

        self.assertEqual(calls, ["m1", "m1"])


class RerankerCacheTests(unittest.TestCase):
    def setUp(self):
        retrieval_runtime.clear_model_caches()

    def tearDown(self):
        retrieval_runtime.clear_model_caches()

    def test_pair_loaded_once_and_reused(self):
        calls = []

        def loader(reranker_id):
            calls.append(reranker_id)
            return (f"tokenizer:{reranker_id}", f"model:{reranker_id}")

        first = retrieval_runtime.get_reranker("rer-1", loader=loader)
        second = retrieval_runtime.get_reranker("rer-1", loader=loader)

        self.assertEqual(first, second)
        self.assertEqual(calls, ["rer-1"])

    def test_load_failure_not_cached_next_call_retries(self):
        calls = []

        def loader(reranker_id):
            calls.append(reranker_id)
            if len(calls) == 1:
                raise RuntimeError("reranker load failed")
            return ("tok", "model")

        with self.assertRaises(RuntimeError):
            retrieval_runtime.get_reranker("rer-1", loader=loader)

        pair = retrieval_runtime.get_reranker("rer-1", loader=loader)
        self.assertEqual(pair, ("tok", "model"))
        self.assertEqual(calls, ["rer-1", "rer-1"])

    def test_clear_forces_reload(self):
        calls = []

        def loader(reranker_id):
            calls.append(reranker_id)
            return ("tok", "model")

        retrieval_runtime.get_reranker("rer-1", loader=loader)
        retrieval_runtime.clear_model_caches()
        retrieval_runtime.get_reranker("rer-1", loader=loader)

        self.assertEqual(calls, ["rer-1", "rer-1"])


if __name__ == "__main__":
    unittest.main()
