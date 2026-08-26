from qdrant_client import QdrantClient, models
import json
import sys
from pathlib import Path


from sentence_transformers import SentenceTransformer
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

QDRANT_PATH = "output/qdrant_storage"
COLLECTION_NAME = "rag_rules_bge_small_zh_v1_5"
MANIFEST_PATH = Path("output/embedding_manifest.json")

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
TOP_K = 5

RERANKER_ID = "BAAI/bge-reranker-base"
TOP_K_RERANK = 3


if len(sys.argv) < 2:
    raise SystemExit(
        'Usage: python retrieve_numpy.py "你的问题"'
    )

query = " ".join(sys.argv[1:])

query_lower = query.lower()
detected_platforms = set()

if "速卖通" in query or "aliexpress" in query_lower:
    detected_platforms.add("aliexpress")
if "temu" in query_lower:
    detected_platforms.add("temu")

if len(detected_platforms) != 1:
    status = (
        "blocked_missing_platform"
        if not detected_platforms
        else "blocked_multiple_platforms"
    )
    bundle = {
        "query": query,
        "status": status,
        "reason": "Query must specify exactly one platform",
        "requested_platform": None,
        "evidence": [],
    }
    output_path = Path("output/evidence_bundle_qdrant.json")
    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    raise SystemExit(0)

requested_platform = detected_platforms.pop()
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

rerank_tokenizer = AutoTokenizer.from_pretrained(
    RERANKER_ID
)
reranker = AutoModelForSequenceClassification.from_pretrained(
    RERANKER_ID
)
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

for item, rerank_score in zip(
    candidates,
    rerank_scores,
):
    item["rerank_score"] = rerank_score

all_reranked = sorted(
    candidates,
    key=lambda item: item["rerank_score"],
    reverse=True,
)

query_lower = query.lower()
detected_platforms = set()

if "速卖通" in query or "aliexpress" in query_lower:
    detected_platforms.add("aliexpress")

if "temu" in query_lower:
    detected_platforms.add("temu")

if not detected_platforms:
    status = "blocked_missing_platform"
    reason = "Query does not specify a platform"
    requested_platform = None
    eligible = []

elif len(detected_platforms) > 1:
    status = "blocked_multiple_platforms"
    reason = "Query mentions multiple platforms"
    requested_platform = None
    eligible = []

else:
    requested_platform = detected_platforms.pop()
    eligible = [
        item for item in all_reranked
        if item["record"]["platform"]
        == requested_platform
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
    "evidence": evidence,
}

output_path = Path("output/evidence_bundle_qdrant.json")
output_path.write_text(
    json.dumps(bundle, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(bundle, ensure_ascii=False, indent=2))