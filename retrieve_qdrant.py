import argparse

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


MODEL_ID = "BAAI/bge-small-zh-v1.5"
QDRANT_PATH = "output/qdrant_storage"
COLLECTION_NAME = "rag_rules_bge_small_zh_v1_5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True)
parser.add_argument(
    "--platform",
    required=True,
    choices=["aliexpress", "temu"],
)
parser.add_argument("--top-k", type=int, default=3)
args = parser.parse_args()

model = SentenceTransformer(MODEL_ID, device="cpu")

query_vector = model.encode(
    QUERY_PREFIX + args.query,
    normalize_embeddings=True,
).tolist()

client = QdrantClient(path=QDRANT_PATH)

platform_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="platform",
            match=models.MatchValue(value=args.platform),
        )
    ]
)

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=platform_filter,
    limit=args.top_k,
    with_payload=True,
    with_vectors=False,
).points

print("query:", args.query)
print("platform_filter:", args.platform)
print("results:", len(results))

for rank, point in enumerate(results, start=1):
    payload = point.payload

    print("\n" + "=" * 70)
    print("rank:", rank)
    print("score:", f"{point.score:.6f}")
    print("platform:", payload["platform"])
    print("chunk_id:", payload["chunk_id"])
    print("headings:", payload.get("headings"))
    print("text:")
    print(payload["text"])

client.close()