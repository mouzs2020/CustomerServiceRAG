import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


CHUNKS_PATH = Path("output/chunks_merged.jsonl")
MODEL_ID = "BAAI/bge-reranker-base"
QUERY = "速卖通平台收到退款申请后，会按照什么流程审核？"

records = [
    json.loads(line)
    for line in CHUNKS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

correct = next(
    record for record in records
    if "merged-0010-0014" in record["chunk_id"]
)

unrelated = next(
    record for record in records
    if record["platform"] == "aliexpress"
    and "第七章 附则" in record["headings"]
)

temu_control = next(
    record for record in records
    if record["platform"] == "temu"
)

candidates = [
    ("correct_aliexpress", correct),
    ("unrelated_aliexpress", unrelated),
    ("temu_control", temu_control),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID
)
model.eval()

pairs = [
    [QUERY, record["contextualized_text"]]
    for _, record in candidates
]

inputs = tokenizer(
    pairs,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

with torch.inference_mode():
    scores = model(
        **inputs,
        return_dict=True,
    ).logits.view(-1).float()

ranked = sorted(
    zip(candidates, scores.tolist()),
    key=lambda item: item[1],
    reverse=True,
)

for rank, ((label, record), score) in enumerate(
    ranked,
    start=1,
):
    print(f"rank: {rank}")
    print(f"label: {label}")
    print(f"score: {score:.6f}")
    print(f"platform: {record['platform']}")
    print(f"headings: {record['headings']}")
    print()