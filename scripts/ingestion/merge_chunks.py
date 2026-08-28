import json
from pathlib import Path

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer


INPUT = Path("output/chunks.jsonl")
OUTPUT = Path("output/chunks_merged.jsonl")
MAX_TOKENS = 400
VERSION = "hybrid-bge-zh-400-postmerge-v1"

title = "速卖通平台商品退款与售后处理规则"
parents = [
    [title, "第三章 退款审核流程", "第一条 审核流程"],
    [title, "第三章 退款审核流程", "第二条 审核时效"],
]

records = [
    json.loads(line)
    for line in INPUT.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5"),
    max_tokens=MAX_TOKENS,
)

consumed = set()
replacements = {}

for parent in parents:
    group = [
        record for record in records
        if record["platform"] == "aliexpress"
        and record["headings"][:len(parent)] == parent
    ]

    indices = [record["chunk_index"] for record in group]
    if not group or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"Missing or non-contiguous group: {parent}")

    parts = []
    for record in group:
        parts.extend(record["headings"][len(parent):])
        parts.append(record["text"])

    merged_text = "\n".join(parts)
    context_text = "\n".join([*parent, merged_text])
    token_count = tokenizer.count_tokens(context_text)

    if token_count > MAX_TOKENS:
        raise ValueError(f"Merged chunk exceeds limit: {token_count}")

    first, last = group[0], group[-1]
    merged = first.copy()
    merged.update({
        "chunk_id": (
            f"{first['source_id']}::{VERSION}::"
            f"merged-{indices[0]:04d}-{indices[-1]:04d}"
        ),
        "chunking_version": VERSION,
        "token_count": token_count,
        "headings": parent,
        "text": merged_text,
        "contextualized_text": context_text,
        "merged_from_chunk_ids": [r["chunk_id"] for r in group],
        "metadata": {
            "merge_type": "same_parent_section",
            "source_chunk_metadata": [r["metadata"] for r in group],
        },
    })

    replacements[first["chunk_id"]] = merged
    consumed.update(record["chunk_id"] for record in group)

final_records = []

for record in records:
    if record["chunk_id"] in replacements:
        final_records.append(replacements[record["chunk_id"]])
    elif record["chunk_id"] not in consumed:
        final_records.append(record)

with OUTPUT.open("w", encoding="utf-8") as output_file:
    for record in final_records:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"before: {len(records)}")
print(f"after: {len(final_records)}")
for record in replacements.values():
    print(record["chunk_id"], record["token_count"])