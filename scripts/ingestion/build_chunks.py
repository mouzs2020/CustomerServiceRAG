import json
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer


INPUTS = [
    {
        "path": Path("output/测试速卖通RAG_结构化对照.json"),
        "platform": "aliexpress",
        "structure_mode": "structured",
        "corpus_role": "target",
    },
    {
        "path": Path("output/测试TemuRAG.json"),
        "platform": "temu",
        "structure_mode": "flat",
        "corpus_role": "negative_control",
    },
]

OUTPUT_PATH = Path("output/chunks.jsonl")

TOKENIZER_ID = "BAAI/bge-small-zh-v1.5"
MAX_TOKENS = 400
CHUNKING_VERSION = "hybrid-bge-zh-400-v1"


hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)

records = []
document_summaries = []

for input_config in INPUTS:
    json_path = input_config["path"]
    document = DoclingDocument.load_from_json(json_path)

    if document.origin is None:
        raise ValueError(
            f"Document origin metadata is missing: {json_path}"
        )

    source_file = document.origin.filename
    source_hash = str(document.origin.binary_hash)
    document_id = Path(source_file).stem
    source_id = f"{document_id}:{source_hash}"

    document_chunks = list(
        chunker.chunk(dl_doc=document)
    )

    for index, chunk in enumerate(
        document_chunks,
        start=1,
    ):
        contextualized_text = chunker.contextualize(
            chunk=chunk
        )
        token_count = tokenizer.count_tokens(
            contextualized_text
        )
        metadata = chunk.meta.export_json_dict()

        if not chunk.text.strip():
            raise ValueError(
                f"Empty chunk: {source_file}, index={index}"
            )

        if token_count > MAX_TOKENS:
            raise ValueError(
                f"Chunk exceeds limit: "
                f"{source_file}, index={index}, "
                f"tokens={token_count}"
            )

        chunk_id = (
            f"{source_id}::{CHUNKING_VERSION}::"
            f"chunk-{index:04d}"
        )

        records.append(
            {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "document_id": document_id,
                "source_file": source_file,
                "source_hash": source_hash,
                "platform": input_config["platform"],
                "structure_mode": input_config[
                    "structure_mode"
                ],
                "corpus_role": input_config["corpus_role"],
                "chunking_version": CHUNKING_VERSION,
                "chunk_index": index,
                "token_count": token_count,
                "headings": metadata.get("headings") or [],
                "text": chunk.text,
                "contextualized_text": contextualized_text,
                "metadata": metadata,
            }
        )

    document_summaries.append(
        (source_file, len(document_chunks))
    )

chunk_ids = [record["chunk_id"] for record in records]

if len(chunk_ids) != len(set(chunk_ids)):
    raise ValueError("Duplicate chunk_id detected")

OUTPUT_PATH.parent.mkdir(exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
    for record in records:
        output_file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

token_counts = [record["token_count"] for record in records]

for source_file, chunk_count in document_summaries:
    print(f"{source_file}: {chunk_count} chunks")

print(f"output: {OUTPUT_PATH}")
print(f"total_chunks: {len(records)}")
print(f"min_tokens: {min(token_counts)}")
print(f"max_tokens: {max(token_counts)}")
print(f"avg_tokens: {sum(token_counts) / len(token_counts):.1f}")