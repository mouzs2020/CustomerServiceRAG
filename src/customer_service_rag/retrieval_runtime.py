"""检索模型进程内缓存：线程安全的惰性加载。

职责边界：
- 仅缓存 Embedding 模型（按 ``model_id`` 键）与 Reranker
  Tokenizer/Model（按 ``reranker_id`` 键）；
- 不缓存 QdrantClient，不改变 Qdrant 生命周期；
- import 本模块零副作用：不加载任何模型、不读取文件、不访问网络；
- 模型构造失败时不写入缓存（不缓存损坏或不完整的对象），
  下次调用会自动重新尝试加载；
- ``clear_model_caches()`` 仅供测试使用，用于避免测试之间状态污染，
  生产代码不得调用。

并发语义：同 key 的并发首次调用使用双检锁（double-checked
locking），只有第一个到达的线程执行加载，其余线程复用结果。
"""

import threading

# 单把锁同时保护两个缓存表：模型加载是低频重操作，锁竞争可忽略；
# 拆成两把锁只会增加复杂度，没有实际收益。
_CACHE_LOCK = threading.Lock()

_EMBEDDING_MODELS: dict[object, object] = {}
_RERANKERS: dict[object, tuple[object, object]] = {}


def _default_load_embedding_model(model_id: object) -> object:
    """默认 Embedding 加载器：惰性导入，与原实现参数一致。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device="cpu")


def _default_load_reranker(reranker_id: object) -> tuple[object, object]:
    """默认 Reranker 加载器：惰性导入，先 Tokenizer 后 Model。"""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(reranker_id)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_id)
    return tokenizer, model


def get_embedding_model(model_id: object, *, loader=None) -> object:
    """返回按 ``model_id`` 缓存的 Embedding 模型（线程安全惰性加载）。

    ``loader`` 仅供测试注入，未传入时使用默认加载器。
    """
    load = loader if loader is not None else _default_load_embedding_model

    model = _EMBEDDING_MODELS.get(model_id)
    if model is not None:
        return model

    with _CACHE_LOCK:
        model = _EMBEDDING_MODELS.get(model_id)
        if model is not None:
            return model
        # 加载放在锁内：同 model_id 的并发首调只执行一次；
        # 只有构造成功才写入缓存，失败时缓存保持为空。
        loaded = load(model_id)
        _EMBEDDING_MODELS[model_id] = loaded
        return loaded


def get_reranker(
    reranker_id: object,
    *,
    loader=None,
) -> tuple[object, object]:
    """返回按 ``reranker_id`` 缓存的 (tokenizer, model)。

    Tokenizer 与 Model 作为整体缓存：任一加载失败都不会留下
    不完整的缓存条目。``loader`` 仅供测试注入。
    """
    load = loader if loader is not None else _default_load_reranker

    cached = _RERANKERS.get(reranker_id)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _RERANKERS.get(reranker_id)
        if cached is not None:
            return cached
        loaded = load(reranker_id)
        _RERANKERS[reranker_id] = loaded
        return loaded


def clear_model_caches() -> None:
    """仅供测试：清空进程内全部模型缓存，下次调用将重新加载。"""
    with _CACHE_LOCK:
        _EMBEDDING_MODELS.clear()
        _RERANKERS.clear()
