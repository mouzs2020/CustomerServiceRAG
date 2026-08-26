from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer


TOKENIZER_ID = "BAAI/bge-small-zh-v1.5"
MAX_TOKENS = 400

JSON_FILES = [
    Path("output/测试TemuRAG.json"),
    Path("output/测试速卖通RAG_结构化对照.json"),
]


hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)


for json_path in JSON_FILES:
    document = DoclingDocument.load_from_json(json_path)
    chunks = list(chunker.chunk(dl_doc=document))

    if document.origin:
        source_id = (
            f"{document.origin.filename}:"
            f"{document.origin.binary_hash}"
        )
    else:
        source_id = json_path.stem

    print("\n" + "=" * 70)
    print(f"document: {json_path.name}")
    print(f"source_id: {source_id}")
    print(f"chunk_count: {len(chunks)}")

    for index, chunk in enumerate(chunks[:5], start=1):
        chunk_id = f"{source_id}::chunk-{index:04d}"
        context_text = chunker.contextualize(chunk=chunk)
        metadata = chunk.meta.export_json_dict()

        print("\n" + "-" * 70)
        print(f"chunk_id: {chunk_id}")
        print(f"tokens: {tokenizer.count_tokens(context_text)}")
        print(f"headings: {metadata.get('headings')}")
        print(f"raw text:\n{chunk.text}")
        print(f"context text:\n{context_text}")