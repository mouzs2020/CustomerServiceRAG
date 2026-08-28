import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

CHUNKS_PATH = Path("output/chunks_merged.jsonl")
EMBEDDINGS_PATH = Path("output/embeddings.npy")
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

chunk_bytes = CHUNKS_PATH.read_bytes()
records = [
    json.loads(line)
    for line in chunk_bytes.decode("utf-8").splitlines()
    if line.strip()
]

manifest = json.loads(
    MANIFEST_PATH.read_text(encoding="utf-8")
)
embeddings = np.load(EMBEDDINGS_PATH)

current_hash = hashlib.sha256(chunk_bytes).hexdigest()

if current_hash != manifest["source_chunks_sha256"]:
    raise ValueError("Chunks changed after embedding")

if [r["chunk_id"] for r in records] != manifest["chunk_ids"]:
    raise ValueError("Chunk order does not match embeddings")

if embeddings.shape[0] != len(records):
    raise ValueError("Embedding count does not match chunks")

model = SentenceTransformer(
    manifest["model_id"],
    device="cpu",
)

query_embedding = model.encode(
    [QUERY_PREFIX + query],
    convert_to_numpy=True,
    normalize_embeddings=True,
)[0]

scores = embeddings @ query_embedding

top_indices = np.argsort(scores)[::-1][:TOP_K]

candidates = []

for retrieve_rank, index in enumerate(
    top_indices,
    start=1,
):
    candidates.append(
        {
            "record": records[int(index)],
            "retrieve_rank": retrieve_rank,
            "retrieve_score": float(scores[index]),
        }
    )

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

output_path = Path("output/evidence_bundle.json")
output_path.write_text(
    json.dumps(bundle, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(bundle, ensure_ascii=False, indent=2))