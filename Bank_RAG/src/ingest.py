from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from pypdf import PdfReader

RAW_DIRECTORIES: Tuple[Tuple[str, Sequence[Path]], ...] = (
    ("rbi_master_circulars", (Path("data/raw/rbi/master_circulars"),),),
    ("sbi", (Path("data/raw/sbi"),)),
)
OUTPUT_DIR = Path("data/cleaned")
STATIC_INGESTION_TIMESTAMP = "STATIC_INGESTION_TIMESTAMP"


def iter_pdf_files() -> Iterable[Tuple[str, Path]]:
    for category, candidates in sorted(RAW_DIRECTORIES, key=lambda item: item[0]):
        base_dir = next((c for c in candidates if c.exists()), None)
        if base_dir is None:
            print(f"Skipping missing directory for category '{category}': {', '.join(str(c) for c in candidates)}")
            continue
        for pdf_path in sorted(base_dir.rglob("*.pdf")):
            yield category, pdf_path


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n")
    lines = text.split("\n")

    cleaned_lines = []
    last_blank = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:page\s*)?\d+\s*$", stripped, flags=re.IGNORECASE):
            continue
        if not stripped:
            if cleaned_lines and not last_blank:
                cleaned_lines.append("")
            last_blank = True
            continue
        normalized = re.sub(r"\s+", " ", stripped)
        cleaned_lines.append(normalized)
        last_blank = False

    return "\n".join(cleaned_lines).strip()


def process_pdfs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_chars = 0

    for category, pdf_path in iter_pdf_files():
        print(f"Processing: {pdf_path}")
        try:
            raw_text = extract_pdf_text(pdf_path)
            cleaned = clean_text(raw_text)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {pdf_path}: {exc}")
            continue

        document_id = pdf_path.stem
        output = {
            "document_id": document_id,
            "source_category": category,
            "original_filename": pdf_path.name,
            "text": cleaned,
            "char_count": len(cleaned),
            "extracted_at": STATIC_INGESTION_TIMESTAMP,
        }

        output_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        total_files += 1
        total_chars += len(cleaned)

    print(f"Total files processed: {total_files}")
    print(f"Total characters extracted: {total_chars}")


def main() -> None:
    process_pdfs()


if __name__ == "__main__":
    main()
