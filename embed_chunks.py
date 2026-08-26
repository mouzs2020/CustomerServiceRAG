import hashlib
import json
from pathlib import Path

import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("output/chunks_merged.jsonl")
EMBEDDINGS_PATH = Path("output/embeddings.npy")
MANIFEST_PATH = Path("output/embedding_manifest.json")

MODEL_ID = "BAAI/bge-small-zh-v1.5"
EXPECTED_DIMENSION = 512
BATCH_SIZE = 8


chunk_bytes = CHUNKS_PATH.read_bytes()
chunks_hash = hashlib.sha256(chunk_bytes).hexdigest()

records = [
    json.loads(line)
    for line in chunk_bytes.decode("utf-8").splitlines()
    if line.strip()
]

texts = [record["contextualized_text"] for record in records]
chunk_ids = [record["chunk_id"] for record in records]

if len(chunk_ids) != len(set(chunk_ids)):
    raise ValueError("Duplicate chunk_id detected")

model = SentenceTransformer(
    MODEL_ID,
    device="cpu",
)

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

embeddings = embeddings.astype(np.float32, copy=False)
norms = np.linalg.norm(embeddings, axis=1)

if embeddings.shape != (len(records), EXPECTED_DIMENSION):
    raise ValueError(
        f"Unexpected shape: {embeddings.shape}"
    )

if not np.isfinite(embeddings).all():
    raise ValueError("Embedding contains NaN or infinity")

if not np.allclose(norms, 1.0, atol=1e-5):
    raise ValueError("Embedding normalization failed")

np.save(EMBEDDINGS_PATH, embeddings)

manifest = {
    "model_id": MODEL_ID,
    "sentence_transformers_version": (
        sentence_transformers.__version__
    ),
    "source_chunks_file": str(CHUNKS_PATH),
    "source_chunks_sha256": chunks_hash,
    "chunk_count": len(records),
    "embedding_dimension": embeddings.shape[1],
    "dtype": str(embeddings.dtype),
    "normalized": True,
    "chunk_ids": chunk_ids,
}

MANIFEST_PATH.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"shape: {embeddings.shape}")
print(f"dtype: {embeddings.dtype}")
print(f"norm_min: {norms.min():.6f}")
print(f"norm_max: {norms.max():.6f}")
print(f"chunks_sha256: {chunks_hash}")
print(f"embeddings: {EMBEDDINGS_PATH}")
print(f"manifest: {MANIFEST_PATH}")