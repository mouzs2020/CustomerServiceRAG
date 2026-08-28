import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("output/chunks_merged.jsonl")
EMBEDDINGS_PATH = Path("output/embeddings.npy")
MANIFEST_PATH = Path("output/embedding_manifest.json")

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
TOP_K = 5


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

print(f"\nquery: {query}")

for rank, index in enumerate(top_indices, start=1):
    record = records[int(index)]

    print("\n" + "=" * 70)
    print(f"rank: {rank}")
    print(f"score: {scores[index]:.6f}")
    print(f"platform: {record['platform']}")
    print(f"chunk_id: {record['chunk_id']}")
    print(f"headings: {record['headings']}")
    print(f"text:\n{record['text']}")