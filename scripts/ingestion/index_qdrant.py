import json
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


CHUNKS_PATH = Path("output/chunks_merged.jsonl")
EMBEDDINGS_PATH = Path("output/embeddings.npy")
MANIFEST_PATH = Path("output/embedding_manifest.json")
QDRANT_PATH = "output/qdrant_storage"
COLLECTION_NAME = "rag_rules_bge_small_zh_v1_5"


chunks = [
    json.loads(line)
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
vectors = np.load(EMBEDDINGS_PATH)
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

if len(chunks) != len(vectors):
    raise ValueError(
        f"Chunk/vector count mismatch: {len(chunks)} != {len(vectors)}"
    )

if vectors.shape[1] != manifest["embedding_dimension"]:
    raise ValueError("Embedding dimension does not match manifest")

expected_ids = (
    manifest.get("chunk_ids")
    or manifest.get("ordered_chunk_ids")
)
actual_ids = [chunk["chunk_id"] for chunk in chunks]

if expected_ids is not None and expected_ids != actual_ids:
    raise ValueError("Chunk order does not match embedding order")

client = QdrantClient(path=QDRANT_PATH)

# 教学阶段采用全量重建，防止旧数据残留。
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=vectors.shape[1],
        distance=models.Distance.COSINE,
    ),
)

points = []
for chunk, vector in zip(chunks, vectors):
    # Qdrant 的 Point ID 使用整数或 UUID；
    # 原始 chunk_id 仍完整保存在 Payload 中。
    point_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])
    )

    points.append(
        models.PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=chunk,
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,
)

stored_count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True,
).count

collection = client.get_collection(COLLECTION_NAME)

print("collection:", COLLECTION_NAME)
print("input_chunks:", len(chunks))
print("stored_points:", stored_count)
print("vector_dimension:", vectors.shape[1])
print("distance:", collection.config.params.vectors.distance)
print("storage:", QDRANT_PATH)

client.close()