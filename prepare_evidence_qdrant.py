"""平台门控 + DeepSeek 意图分类 + Qdrant 检索重排 + Evidence Gate。

调用顺序（每一层通过才会进入下一层）：
1. 确定性平台门控（platform_gate.resolve_platform）；
2. DeepSeek 意图分类（intent_classifier.classify_intent）；
3. 检索重排（Qdrant 平台过滤 + BGE Reranker，本文件不改动其内部逻辑）；
4. Evidence Gate 二次校验（evaluate_evidence）。

任何 blocked 状态都会写出 ``evidence: []`` 的证据包，
并且不会触碰后续组件。
"""

import argparse
import json
import math
import os
from pathlib import Path

from intent_classifier import (
    IntentClassifierError,
    classify_intent,
    get_intent_confidence_threshold,
)
from platform_gate import resolve_platform


QDRANT_PATH = "output/qdrant_storage"
COLLECTION_NAME = "rag_rules_bge_small_zh_v1_5"
MANIFEST_PATH = Path("output/embedding_manifest.json")
OUTPUT_PATH = Path("output/evidence_bundle_qdrant.json")

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
TOP_K = 5

RERANKER_ID = "BAAI/bge-reranker-base"
TOP_K_RERANK = 3

# Reranker 分数阈值初始防线：尚未通过评测标定，仅用于拦截明显不相关证据。
DEFAULT_MIN_RERANK_SCORE = 0.75


class EvidenceGateConfigError(ValueError):
    """Evidence Gate 配置（如 MIN_RERANK_SCORE）非法。"""


def get_min_rerank_score() -> float:
    """Reranker 分数阈值（环境变量 MIN_RERANK_SCORE 可覆盖，默认 0.75）。

    必须是有限数字：NaN / inf / 非法字符串一律抛出
    ``EvidenceGateConfigError``，配置错误一律 fail closed。
    空白值视同未设置。
    注意：默认值未经评测确定，不是可靠阈值，调参前请先做检索评测。
    """
    raw = os.environ.get("MIN_RERANK_SCORE")
    if not raw or not raw.strip():
        return DEFAULT_MIN_RERANK_SCORE

    try:
        value = float(raw)
    except ValueError as exc:
        raise EvidenceGateConfigError(
            f"Invalid MIN_RERANK_SCORE: {raw!r}"
        ) from exc

    if not math.isfinite(value):
        raise EvidenceGateConfigError(
            f"MIN_RERANK_SCORE must be a finite number: {raw!r}"
        )
    return value


def build_evidence_bundle(
    query: str,
    status: str,
    reason: str,
    requested_platform: object,
    entry_platform: object,
    *,
    intent_result: dict[str, object] | None = None,
    evidence_gate: dict[str, object] | None = None,
    evidence: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """组装证据包；既有字段全部保留，新增意图与证据门字段。"""
    if intent_result is None:
        intent_value: object = None
        confidence_value: object = None
        reason_value: object = None
    else:
        intent_value = intent_result["intent"]
        confidence_value = intent_result["confidence"]
        reason_value = intent_result["reason"]

    return {
        "query": query,
        "status": status,
        "reason": reason,
        "requested_platform": requested_platform,
        "entry_platform": entry_platform,
        "intent": intent_value,
        "intent_confidence": confidence_value,
        "intent_reason": reason_value,
        "evidence_gate": evidence_gate,
        "evidence": evidence if evidence is not None else [],
    }


def decide_after_intent(
    intent_result: dict[str, object],
) -> tuple[str | None, str]:
    """根据意图分类结果决定下一步状态。

    返回 ``(status, reason)``；status 为 None 表示通过，允许进入检索。
    """
    intent = intent_result["intent"]
    confidence = float(intent_result["confidence"])  # type: ignore[arg-type]

    if intent == "unrelated":
        return (
            "blocked_unrelated_question",
            "Intent classifier marked the question as unrelated",
        )

    if intent == "uncertain":
        return (
            "blocked_intent_uncertain",
            "Intent classifier returned uncertain",
        )

    threshold = get_intent_confidence_threshold()
    if confidence < threshold:
        return (
            "blocked_intent_uncertain",
            f"Intent confidence {confidence:.2f} below threshold {threshold:.2f}",
        )

    return None, "Platform gate and intent classification passed"


def evaluate_evidence(
    requested_platform: str,
    reranked_candidates: list[dict[str, object]],
) -> tuple[str, str, list[dict[str, object]], dict[str, object]]:
    """Evidence Gate 二次校验（独立函数，不混入 Citation Validator）。

    至少检查：
    - 阈值配置是否为有限数字（非法时 fail closed，
      新增状态 ``blocked_evidence_gate_config_error``）；
    - 是否存在候选证据；
    - 每条证据的 platform 是否等于 requested_platform；
    - 最高 rerank 分数是否达到阈值（get_min_rerank_score）；
    - 进入证据包的条数不超过 TOP_K_RERANK。

    返回 ``(status, reason, evidence, gate_info)``；
    blocked 状态下 evidence 恒为 []，gate_info["passed"] 为 False。
    """
    # 配置校验最先执行：阈值非法时拒绝给出任何“通过”结论。
    try:
        min_score = get_min_rerank_score()
        config_error: str | None = None
    except EvidenceGateConfigError as exc:
        min_score = None
        config_error = str(exc)

    scores = [float(item["rerank_score"]) for item in reranked_candidates]
    finite_scores = [score for score in scores if math.isfinite(score)]

    gate_info: dict[str, object] = {
        "passed": False,
        "min_rerank_score": min_score,
        "checked_candidates": len(reranked_candidates),
        "top_rerank_score": max(finite_scores) if finite_scores else None,
    }

    if config_error is not None:
        return (
            "blocked_evidence_gate_config_error",
            f"Evidence gate misconfigured: {config_error}",
            [],
            gate_info,
        )

    if not reranked_candidates:
        return (
            "blocked_no_matching_source",
            "No candidate matches the requested platform",
            [],
            gate_info,
        )

    # 防御：非有限分数（NaN/inf）按低相关处理，避免逃过阈值比较。
    if len(finite_scores) != len(scores):
        return (
            "blocked_low_relevance",
            "Reranker produced non-finite scores",
            [],
            gate_info,
        )

    mismatched_platforms = sorted(
        {
            str(item["record"].get("platform"))
            for item in reranked_candidates
            if item["record"].get("platform") != requested_platform
        }
    )
    if mismatched_platforms:
        return (
            "blocked_platform_evidence_mismatch",
            (
                f"Evidence platforms {mismatched_platforms} do not match "
                f"requested platform {requested_platform}"
            ),
            [],
            gate_info,
        )

    top_score = float(gate_info["top_rerank_score"])  # type: ignore[arg-type]
    if top_score < min_score:
        return (
            "blocked_low_relevance",
            f"Top rerank score {top_score:.3f} below minimum {min_score:.3f}",
            [],
            gate_info,
        )

    qualified = [
        item for item in reranked_candidates
        if float(item["rerank_score"]) >= min_score
    ]
    qualified.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
    limited = qualified[:TOP_K_RERANK]

    evidence: list[dict[str, object]] = []
    for index, item in enumerate(limited, start=1):
        record = item["record"]
        evidence.append(
            {
                "citation_id": f"E{index}",
                "chunk_id": record["chunk_id"],
                "source_id": record["source_id"],
                "platform": record["platform"],
                "headings": record["headings"],
                "text": record["text"],
                "retrieve_score": item["retrieve_score"],
                "rerank_score": item["rerank_score"],
            }
        )

    gate_info["passed"] = True
    return (
        "ready_for_grounding",
        (
            "Evidence gate passed; answer stage must still verify "
            "that evidence supports the claim"
        ),
        evidence,
        gate_info,
    )


def retrieve_and_rank(
    query: str,
    requested_platform: str,
) -> list[dict[str, object]]:
    """粗筛（Qdrant 平台过滤检索） + 重排。

    只有平台门控与意图分类都通过后才会被调用，因此冲突 / 缺失 /
    无关 / 不确定的场景不会加载 Embedding 模型、
    也不会访问 Qdrant 或 Reranker。

    边界行为：
    - 零候选：在加载 Reranker 之前直接返回 ``[]``，
      由 Evidence Gate 给出正式状态（blocked_no_matching_source）；
    - 跨平台候选：不再抛异常，原样交给 Evidence Gate
      判定为 blocked_platform_evidence_mismatch。
    """
    import torch
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    model = SentenceTransformer(manifest["model_id"], device="cpu")
    query_embedding = model.encode(
        [QUERY_PREFIX + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    client = QdrantClient(path=QDRANT_PATH)
    points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="platform",
                    match=models.MatchValue(value=requested_platform),
                )
            ]
        ),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    ).points
    client.close()

    candidates = [
        {
            "record": point.payload,
            "retrieve_rank": rank,
            "retrieve_score": float(point.score),
        }
        for rank, point in enumerate(points, start=1)
    ]

    # 零候选：不加载 Reranker、不调用 Tokenizer，直接返回空列表，
    # 让 Evidence Gate 生成正式的 blocked_no_matching_source。
    if not candidates:
        return []

    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_ID)
    reranker = AutoModelForSequenceClassification.from_pretrained(RERANKER_ID)
    reranker.eval()

    pairs = [
        [query, item["record"]["contextualized_text"]]
        for item in candidates
    ]
    inputs = rerank_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.inference_mode():
        rerank_scores = reranker(
            **inputs,
            return_dict=True,
        ).logits.view(-1).float().tolist()

    for item, rerank_score in zip(candidates, rerank_scores):
        item["rerank_score"] = rerank_score

    return sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )


def write_and_print(bundle: dict[str, object]) -> None:
    """写出并打印证据包。"""
    OUTPUT_PATH.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(bundle, ensure_ascii=False, indent=2))


def emit_intent_classifier_error(
    query: str,
    gate: dict[str, object],
    exc: IntentClassifierError,
) -> None:
    """写出意图分类失败的 blocked 证据包并打印。"""
    write_and_print(
        build_evidence_bundle(
            query=query,
            status="blocked_intent_classifier_error",
            reason=f"Intent classifier failed: {exc}",
            requested_platform=gate["requested_platform"],
            entry_platform=gate["entry_platform"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the evidence bundle through the platform-gated, "
            "intent-classified RAG pipeline."
        )
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="User question.",
    )
    parser.add_argument(
        "--user-platform",
        default="",
        help=(
            "Platform the user entered from (aliexpress or temu). "
            "Simulates the platform API in production."
        ),
    )
    args = parser.parse_args()

    query = " ".join(args.query)

    # 第 1 步：确定性平台门控（blocked 时不会触发意图分类和检索）。
    gate = resolve_platform(query, args.user_platform)

    if gate["status"] != "platform_resolved":
        write_and_print(
            build_evidence_bundle(
                query=query,
                status=str(gate["status"]),
                reason=str(gate["reason"]),
                requested_platform=gate["requested_platform"],
                entry_platform=gate["entry_platform"],
            )
        )
        return

    requested_platform = gate["requested_platform"]

    # 第 2 步：DeepSeek 意图分类；失败属于 blocked_intent_classifier_error。
    try:
        intent_result = classify_intent(query)
    except IntentClassifierError as exc:
        emit_intent_classifier_error(query, gate, exc)
        return

    # 第 3 步：意图决策；unrelated / uncertain / 低置信度都在此拦截。
    # 意图阈值配置非法（如 INTENT_CONFIDENCE_THRESHOLD=abc）同样
    # 属于分类器侧错误，必须转为 blocked 状态而不是崩溃。
    try:
        intent_status, intent_reason = decide_after_intent(intent_result)
    except IntentClassifierError as exc:
        emit_intent_classifier_error(query, gate, exc)
        return
    if intent_status is not None:
        write_and_print(
            build_evidence_bundle(
                query=query,
                status=intent_status,
                reason=intent_reason,
                requested_platform=requested_platform,
                entry_platform=gate["entry_platform"],
                intent_result=intent_result,
            )
        )
        return

    # 第 4 步：检索 + 重排（Qdrant 平台过滤 + BGE Reranker）。
    all_reranked = retrieve_and_rank(query, requested_platform)

    # 第 5 步：Evidence Gate 二次校验。
    evidence_status, evidence_reason, evidence, gate_info = evaluate_evidence(
        requested_platform,
        all_reranked,
    )

    write_and_print(
        build_evidence_bundle(
            query=query,
            status=evidence_status,
            reason=evidence_reason,
            requested_platform=requested_platform,
            entry_platform=gate["entry_platform"],
            intent_result=intent_result,
            evidence_gate=gate_info,
            evidence=evidence,
        )
    )


if __name__ == "__main__":
    main()
