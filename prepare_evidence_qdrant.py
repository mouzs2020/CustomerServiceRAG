import argparse
import json
from pathlib import Path

from platform_gate import resolve_platform


QDRANT_PATH = "output/qdrant_storage"
COLLECTION_NAME = "rag_rules_bge_small_zh_v1_5"
MANIFEST_PATH = Path("output/embedding_manifest.json")
OUTPUT_PATH = Path("output/evidence_bundle_qdrant.json")

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
TOP_K = 5

RERANKER_ID = "BAAI/bge-reranker-base"
TOP_K_RERANK = 3


def build_blocked_bundle(query: str, gate: dict[str, object]) -> dict[str, object]:
    return {
        "query": query,
        "status": gate["status"],
        "reason": gate["reason"],
        "requested_platform": gate["requested_platform"],
        "entry_platform": gate["entry_platform"],
        "evidence": [],
    }


def retrieve_and_rank(
    query: str,
    requested_platform: str,
) -> list[dict[str, object]]:
    """粗筛（Qdrant 平台过滤检索） + 重排。

    只有平台门控通过后才会被调用，因此冲突 / 缺失 / 无关的场景
    不会加载 Embedding 模型，也不会访问 Qdrant 或 Reranker。
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

    if any(
        item["record"]["platform"] != requested_platform
        for item in candidates
    ):
        raise RuntimeError("Qdrant platform filter was violated")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the evidence bundle for the platform-gated RAG pipeline."
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
    gate = resolve_platform(query, args.user_platform)

    if gate["status"] != "platform_resolved":
        bundle = build_blocked_bundle(query, gate)
        OUTPUT_PATH.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return

    requested_platform = gate["requested_platform"]
    all_reranked = retrieve_and_rank(query, requested_platform)

    eligible = [
        item
        for item in all_reranked
        if item["record"]["platform"] == requested_platform
    ][:TOP_K_RERANK]

    if eligible:
        status = "ready_for_grounding"
        reason = (
            "Platform gate passed; answer stage must "
            "still verify that evidence supports the claim"
        )
    else:
        status = "blocked_no_matching_source"
        reason = "No candidate matches the requested platform"

    evidence = []
    for index, item in enumerate(eligible, start=1):
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

    bundle = {
        "query": query,
        "status": status,
        "reason": reason,
        "requested_platform": requested_platform,
        "entry_platform": gate["entry_platform"],
        "evidence": evidence,
    }

    OUTPUT_PATH.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
