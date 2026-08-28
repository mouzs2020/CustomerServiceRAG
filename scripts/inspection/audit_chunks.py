import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


CHUNKS_PATH = Path("output/chunks_merged.jsonl")
MAX_TOKENS = 400

REQUIRED_FIELDS = {
    "chunk_id",
    "source_id",
    "source_file",
    "platform",
    "token_count",
    "headings",
    "text",
    "contextualized_text",
}

records = []

with CHUNKS_PATH.open(encoding="utf-8") as input_file:
    for line_number, line in enumerate(input_file, start=1):
        if not line.strip():
            continue

        record = json.loads(line)
        record["_line_number"] = line_number
        records.append(record)

errors = []
flags = defaultdict(list)

id_counts = Counter(
    record.get("chunk_id") for record in records
)

for record in records:
    chunk_id = record.get(
        "chunk_id",
        f"line-{record['_line_number']}",
    )

    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        errors.append(
            f"{chunk_id}: missing fields {sorted(missing)}"
        )
        continue

    if id_counts[record["chunk_id"]] > 1:
        errors.append(f"{chunk_id}: duplicate chunk_id")

    if not record["text"].strip():
        errors.append(f"{chunk_id}: empty text")

    if record["token_count"] > MAX_TOKENS:
        errors.append(
            f"{chunk_id}: {record['token_count']} tokens"
        )

    if record["text"] not in record["contextualized_text"]:
        errors.append(
            f"{chunk_id}: raw text missing from context"
        )

    if record["token_count"] >= 360:
        flags["near_token_limit"].append(chunk_id)

    if record["token_count"] < 50:
        flags["very_small"].append(chunk_id)

    if (
        record["platform"] == "aliexpress"
        and not record["headings"]
    ):
        flags["target_without_headings"].append(chunk_id)

    chapter_names = set(
        re.findall(
            r"第[一二三四五六七八九十百零〇0-9]+章",
            record["text"],
        )
    )

    if len(chapter_names) > 1:
        flags["cross_chapter"].append(chunk_id)

print(f"records: {len(records)}")
print(f"errors: {len(errors)}")

by_platform = defaultdict(list)

for record in records:
    by_platform[record["platform"]].append(record)

for platform, platform_records in by_platform.items():
    token_counts = [
        record["token_count"]
        for record in platform_records
    ]
    heading_count = sum(
        bool(record["headings"])
        for record in platform_records
    )

    print(f"\nplatform: {platform}")
    print(f"chunks: {len(platform_records)}")
    print(
        "tokens: "
        f"min={min(token_counts)}, "
        f"avg={mean(token_counts):.1f}, "
        f"max={max(token_counts)}"
    )
    print(
        "heading_coverage: "
        f"{heading_count}/{len(platform_records)}"
    )

print("\nflags:")

for flag_name, chunk_ids in flags.items():
    print(f"{flag_name}: {len(chunk_ids)}")

    for chunk_id in chunk_ids[:5]:
        print(f"  {chunk_id}")

if errors:
    print("\nvalidation errors:")

    for error in errors:
        print(f"  {error}")