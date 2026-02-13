import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
PARA_BREAK_TOKEN = "\n\n"


def load_cleaned_documents(cleaned_dir: Path) -> Iterable[Tuple[Path, Dict[str, str]]]:
    for path in sorted(cleaned_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            yield path, json.load(handle)


def sanitize_paragraph(paragraph: str) -> str:
    normalized = re.sub(r"\s+", " ", paragraph.strip())
    tokens = normalized.split(" ")
    fixed: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            if re.fullmatch(r"[A-Za-z]{4,}", token) and re.fullmatch(r"[a-z]{1,3}", nxt):
                fixed.append(token + nxt)
                idx += 2
                continue
        fixed.append(token)
        idx += 1
    return " ".join(fixed).strip()


def strip_control_characters(text: str) -> str:
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep printable characters and explicit newlines; drop other control characters.
    return "".join(ch if (ch == "\n" or ch.isprintable()) else " " for ch in normalized_newlines)


def clean_text(raw_text: str) -> str:
    filtered_text = strip_control_characters(raw_text)
    paragraphs = [p for p in re.split(r"\n\s*\n", filtered_text) if p.strip()]
    cleaned_paragraphs = [sanitize_paragraph(paragraph) for paragraph in paragraphs]
    return PARA_BREAK_TOKEN.join(cleaned_paragraphs)


def build_words_and_breaks(text: str) -> Tuple[List[str], Set[int]]:
    paragraphs = [p for p in text.split(PARA_BREAK_TOKEN) if p.strip()]
    words: List[str] = []
    break_after: Set[int] = set()
    cursor = 0
    for idx, paragraph in enumerate(paragraphs):
        para_words = paragraph.split()
        words.extend(para_words)
        cursor += len(para_words)
        if idx < len(paragraphs) - 1:
            break_after.add(cursor)
    return words, break_after


def render_chunk(words: Sequence[str], start: int, end: int, break_after: Set[int]) -> str:
    pieces: List[str] = []
    for position in range(start, end):
        pieces.append(words[position])
        global_pos = position + 1
        if global_pos in break_after and position + 1 < end:
            pieces.append(PARA_BREAK_TOKEN)
    text = " ".join(pieces)
    text = text.replace(" \n\n ", "\n\n").replace("\n\n ", "\n\n").replace(" \n\n", "\n\n")
    return text.strip()


def create_chunks(words: List[str], break_after: Set[int]) -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = min(len(words), start + CHUNK_SIZE)
        chunk_text = render_chunk(words, start, end, break_after)
        chunk_words = end - start
        chunks.append({
            "chunk_id": f"{{doc_id}}_chunk_{chunk_index + 1:04d}",
            "chunk_index": chunk_index,
            "text": chunk_text,
            "word_count": chunk_words,
        })
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_index += 1
    return chunks


def attach_metadata(doc_id: str, source_category: str, raw_chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for entry in raw_chunks:
        enriched.append({
            "chunk_id": entry["chunk_id"].replace("{doc_id}", doc_id),
            "document_id": doc_id,
            "source_category": source_category,
            "chunk_index": entry["chunk_index"],
            "text": entry["text"],
            "word_count": entry["word_count"],
        })
    return enriched


def process_documents(base_dir: Path) -> None:
    cleaned_dir = base_dir / "data" / "cleaned"
    chunks_dir = base_dir / "data" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    for path, doc in load_cleaned_documents(cleaned_dir):
        document_id = doc.get("document_id", path.stem)
        source_category = doc.get("source_category", "")
        raw_text = doc.get("text", "")
        cleaned_text = clean_text(raw_text)
        raw_char_count = len(raw_text)
        cleaned_char_count = len(cleaned_text)
        words, break_after = build_words_and_breaks(cleaned_text)
        word_count = len(words)
        print(
            f"{document_id} diagnostics: raw_chars={raw_char_count}, "
            f"cleaned_chars={cleaned_char_count}, words={word_count}"
        )
        if word_count == 0:
            if cleaned_char_count > 0:
                raise ValueError(
                    f"No words found after cleaning for document '{document_id}'. "
                    f"raw_chars={raw_char_count}, cleaned_chars={cleaned_char_count}"
                )
            raw_chunks = [{
                "chunk_id": f"{document_id}_chunk_0001",
                "chunk_index": 0,
                "text": "",
                "word_count": 0,
            }]
        else:
            raw_chunks = create_chunks(words, break_after)
        chunks = attach_metadata(document_id, source_category, raw_chunks)

        output_path = chunks_dir / f"{document_id}_chunks.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(chunks, handle, ensure_ascii=False, indent=2)

        print(f"{document_id}: {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"Total chunks: {total_chunks}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    process_documents(base_dir)


if __name__ == "__main__":
    main()
