"""本地静态 Readiness 检查。

只读取本地配置文件并做结构校验：
不实例化 QdrantClient，不发起网络请求，不加载任何模型。
结果只包含检查项布尔值，不泄漏 API Key、文件内容或绝对路径。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from customer_service_rag.schemas import ReadinessResponse

COLLECTION_NAME = "rag_rules_bge_small_zh_v1_5"
MANIFEST_RELATIVE = Path("output") / "embedding_manifest.json"
QDRANT_META_RELATIVE = Path("output") / "qdrant_storage" / "meta.json"


def _load_json(path: Path) -> object | None:
    """读取 JSON；文件缺失或内容非法返回 None（fail closed）。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_positive_int(value: object) -> bool:
    """正整数；bool 是 int 的子类，必须显式排除。"""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _manifest_valid(manifest: object) -> bool:
    """embedding_manifest 结构校验；任何字段异常都返回 False。"""
    if not isinstance(manifest, dict):
        return False
    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    if not _is_positive_int(manifest.get("embedding_dimension")):
        return False
    if not _is_positive_int(manifest.get("chunk_count")):
        return False
    chunk_ids = manifest.get("chunk_ids")
    if not isinstance(chunk_ids, list):
        return False
    return len(chunk_ids) == manifest["chunk_count"]


def _collection_vectors(meta: object) -> dict | None:
    """从 qdrant meta.json 提取目标 collection 的 vectors 配置。"""
    if not isinstance(meta, dict):
        return None
    collections = meta.get("collections")
    if not isinstance(collections, dict):
        return None
    collection = collections.get(COLLECTION_NAME)
    if not isinstance(collection, dict):
        return None
    vectors = collection.get("vectors")
    if not isinstance(vectors, dict):
        return None
    return vectors


def check_readiness(
    *,
    base_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ReadinessResponse:
    """只做本地静态检查；全部通过才是 ready，否则 not_ready。"""
    if base_dir is None:
        base_dir = Path.cwd()
    if environ is None:
        environ = os.environ

    api_key = environ.get("DEEPSEEK_API_KEY") or ""
    deepseek_api_key = bool(api_key.strip())

    manifest = _load_json(base_dir / MANIFEST_RELATIVE)
    embedding_manifest = _manifest_valid(manifest)

    vectors = _collection_vectors(
        _load_json(base_dir / QDRANT_META_RELATIVE)
    )
    qdrant_collection = vectors is not None

    dimension_match = bool(
        embedding_manifest
        and vectors is not None
        and _is_positive_int(vectors.get("size"))
        and vectors["size"] == manifest["embedding_dimension"]  # type: ignore[index]
        and vectors.get("distance") == "Cosine"
    )

    checks = {
        "deepseek_api_key": deepseek_api_key,
        "embedding_manifest": embedding_manifest,
        "qdrant_collection": qdrant_collection,
        "dimension_match": dimension_match,
    }
    status = "ready" if all(checks.values()) else "not_ready"
    return ReadinessResponse(status=status, checks=checks)
