from pathlib import Path

from docling.document_converter import DocumentConverter


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"


def main() -> None:
    source_files = sorted(DATA_DIR.glob("*.docx"))
    if not source_files:
        raise FileNotFoundError(f"No DOCX files found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    converter = DocumentConverter()

    for source_path in source_files:
        print(f"[parse] {source_path.name}")
        result = converter.convert(source_path)
        document = result.document

        json_path = OUTPUT_DIR / f"{source_path.stem}.json"
        markdown_path = OUTPUT_DIR / f"{source_path.stem}.md"

        document.save_as_json(json_path, indent=2)
        document.save_as_markdown(markdown_path)

        print(
            f"[done] status={result.status.value}, "
            f"texts={len(document.texts)}, "
            f"tables={len(document.tables)}, "
            f"pictures={len(document.pictures)}"
        )
        print(f"       JSON: {json_path}")
        print(f"       Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
