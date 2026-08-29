"""readiness.check_readiness 的单元测试（全静态、无 Qdrant 实例、无网络）。"""

import json
import tempfile
import unittest
from pathlib import Path

from customer_service_rag.readiness import COLLECTION_NAME, check_readiness

KEY_OK = {"DEEPSEEK_API_KEY": "test-key"}
MODEL_ID = "BAAI/bge-small-zh-v1.5"


def make_env(tmp, **kwargs) -> dict:
    """在 tmp/output 下写入 manifest 与 qdrant meta，返回检查入参。"""
    dim = kwargs.get("dim", 512)
    chunk_count = kwargs.get("chunk_count", 3)
    chunk_ids = kwargs.get("chunk_ids")
    model_id = kwargs.get("model_id", MODEL_ID)
    distance = kwargs.get("distance", "Cosine")
    qdrant_size = kwargs.get("qdrant_size", 512)
    manifest_raw = kwargs.get("manifest_raw")
    meta_raw = kwargs.get("meta_raw")
    write_manifest = kwargs.get("write_manifest", True)
    write_meta = kwargs.get("write_meta", True)

    out_dir = tmp / "output"
    (out_dir / "qdrant_storage").mkdir(parents=True, exist_ok=True)
    if write_manifest:
        if manifest_raw is None:
            ids = (
                chunk_ids
                if chunk_ids is not None
                else [f"c{i}" for i in range(chunk_count)]
            )
            manifest_raw = json.dumps(
                {
                    "model_id": model_id,
                    "embedding_dimension": dim,
                    "chunk_count": chunk_count,
                    "chunk_ids": ids,
                }
            )
        (out_dir / "embedding_manifest.json").write_text(
            manifest_raw, encoding="utf-8"
        )
    if write_meta:
        if meta_raw is None:
            meta_raw = json.dumps(
                {
                    "collections": {
                        COLLECTION_NAME: {
                            "vectors": {"size": qdrant_size, "distance": distance}
                        }
                    }
                }
            )
        (out_dir / "qdrant_storage" / "meta.json").write_text(
            meta_raw, encoding="utf-8"
        )
    return kwargs


def check(tmp: Path, env=None):
    return check_readiness(
        base_dir=tmp, environ=KEY_OK if env is None else env
    )


class ReadinessTests(unittest.TestCase):
    def run_case(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            make_env(tmp, **kwargs)
            return check(tmp, kwargs.get("environ"))

    def test_all_normal_is_ready(self):
        result = self.run_case()
        self.assertEqual(result.status, "ready")
        self.assertEqual(
            set(result.checks),
            {"deepseek_api_key", "embedding_manifest",
             "qdrant_collection", "dimension_match"},
        )
        self.assertTrue(all(result.checks.values()))

    def test_missing_or_blank_key_not_ready(self):
        for env in ({}, {"DEEPSEEK_API_KEY": "   "}):
            with self.subTest(env=env):
                result = self.run_case(environ=env)
                self.assertEqual(result.status, "not_ready")
                self.assertFalse(result.checks["deepseek_api_key"])

    def test_manifest_missing_or_invalid_json_or_bad_types(self):
        cases = [
            {"write_manifest": False},
            {"manifest_raw": "{oops"},
            {"model_id": 123},
            {"model_id": ""},
            {"model_id": "   "},
            {"dim": "512"},
            {"chunk_count": 0},
            {"chunk_ids": "not-a-list"},
        ]
        for case in cases:
            with self.subTest(case=case):
                result = self.run_case(**case)
                self.assertEqual(result.status, "not_ready")
                self.assertFalse(result.checks["embedding_manifest"])
                self.assertFalse(result.checks["dimension_match"])

    def test_chunk_ids_count_mismatch_not_ready(self):
        result = self.run_case(chunk_count=3, chunk_ids=["c0", "c1"])
        self.assertEqual(result.status, "not_ready")
        self.assertFalse(result.checks["embedding_manifest"])

    def test_qdrant_meta_missing_or_collection_missing(self):
        for case in (
            {"write_meta": False},
            {"meta_raw": json.dumps({"collections": {"other": {"vectors": {
                "size": 512, "distance": "Cosine"}}}})},
        ):
            with self.subTest(case=case):
                result = self.run_case(**case)
                self.assertEqual(result.status, "not_ready")
                self.assertFalse(result.checks["qdrant_collection"])

    def test_dimension_mismatch_not_ready(self):
        result = self.run_case(qdrant_size=768)
        self.assertEqual(result.status, "not_ready")
        self.assertTrue(result.checks["qdrant_collection"])
        self.assertFalse(result.checks["dimension_match"])

    def test_distance_not_cosine_not_ready(self):
        result = self.run_case(distance="Euclid")
        self.assertEqual(result.status, "not_ready")
        self.assertFalse(result.checks["dimension_match"])

    def test_bool_is_not_valid_int(self):
        for field in ("embedding_dimension", "chunk_count"):
            with self.subTest(field=field):
                manifest = json.dumps(
                    {
                        "model_id": MODEL_ID,
                        "embedding_dimension": True if field == "embedding_dimension" else 512,
                        "chunk_count": True if field == "chunk_count" else 3,
                        "chunk_ids": ["c0", "c1", "c2"],
                    }
                )
                result = self.run_case(manifest_raw=manifest)
                self.assertEqual(result.status, "not_ready")
                self.assertFalse(result.checks["embedding_manifest"])

    def test_result_leaks_nothing(self):
        secret = "sk-test-key-123"
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            make_env(tmp, model_id="SECRET-MODEL-CONTENT")
            result = check(tmp, {"DEEPSEEK_API_KEY": secret})
            payload = result.model_dump_json()
            self.assertNotIn(secret, payload)
            self.assertNotIn(str(tmp), payload)
            self.assertNotIn("SECRET-MODEL-CONTENT", payload)
            self.assertNotIn("embedding_manifest.json", payload)
            self.assertEqual(set(json.loads(payload)), {"status", "checks"})
            for name, value in result.checks.items():
                self.assertIsInstance(value, bool)


if __name__ == "__main__":
    unittest.main()
