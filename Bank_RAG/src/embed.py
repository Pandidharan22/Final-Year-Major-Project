import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STATIC_EMBED_TIMESTAMP = "STATIC_EMBED_TIMESTAMP"
SEED = 42
CHUNKS_DIR = Path("data/chunks")
VECTOR_STORE_DIR = Path("vector_store")
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"


def set_determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_chunks(chunk_dir: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    chunks: List[Dict[str, object]] = []
    documents: List[str] = []
    for path in sorted(chunk_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        # Ensure deterministic order within file
        payload_sorted = sorted(payload, key=lambda item: item.get("chunk_index", 0))
        chunks.extend(payload_sorted)
        doc_id = path.stem.replace("_chunks", "")
        documents.append(doc_id)
    return chunks, sorted(set(documents))


def build_embeddings(model: SentenceTransformer, chunks: List[Dict[str, object]]) -> np.ndarray:
    texts = [entry["text"] for entry in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device="cpu",
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def write_metadata(base_dir: Path, model_name: str, chunk_count: int, documents: List[str]) -> None:
    metadata_path = base_dir / METADATA_PATH
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "embedding_model": model_name,
        "chunk_count": chunk_count,
        "documents_indexed": documents,
        "created_at": STATIC_EMBED_TIMESTAMP,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    chunk_dir = base_dir / CHUNKS_DIR
    set_determinism()

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.eval()
    dim = model.get_sentence_embedding_dimension()

    chunks, documents = load_chunks(chunk_dir)
    if not chunks:
        raise ValueError("No chunks found to embed.")

    print(f"Embedding {len(chunks)} chunks from {len(documents)} documents")
    per_doc_counts: Dict[str, int] = {}
    for entry in chunks:
        doc_id = entry.get("document_id", "")
        per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
    for doc_id in sorted(per_doc_counts):
        print(f"  {doc_id}: {per_doc_counts[doc_id]} chunks")

    embeddings = build_embeddings(model, chunks)
    print(f"Embedding dimension: {embeddings.shape[1]}")

    index = build_faiss_index(embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(base_dir / INDEX_PATH))
    write_metadata(base_dir, MODEL_NAME, len(chunks), documents)

    print(f"Total embeddings generated: {len(chunks)}")


if __name__ == "__main__":
    main()
