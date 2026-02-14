import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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


def load_chunks(chunk_dir: Path) -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    for path in sorted(chunk_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload_sorted = sorted(payload, key=lambda item: item.get("chunk_index", 0))
        chunks.extend(payload_sorted)
    return chunks


def load_index_and_metadata(base_dir: Path) -> Tuple[faiss.Index, Dict[str, object]]:
    index = faiss.read_index(str(base_dir / INDEX_PATH))
    with (base_dir / METADATA_PATH).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return index, metadata


def build_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.eval()
    return model


def retrieve(
    query: str,
    top_k: int,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device="cpu",
    ).astype("float32")
    scores, idxs = index.search(query_emb, top_k)
    results: List[Dict[str, object]] = []
    for rank, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist()), start=1):
        chunk = chunks[idx]
        preview = chunk.get("text", "")[:300]
        results.append({
            "rank": rank,
            "chunk_id": chunk.get("chunk_id", ""),
            "document_id": chunk.get("document_id", ""),
            "source_category": chunk.get("source_category", ""),
            "score": float(score),
            "text_preview": preview,
        })
    return results


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    set_determinism()
    model = build_model()
    chunks = load_chunks(base_dir / CHUNKS_DIR)
    if not chunks:
        raise ValueError("No chunks loaded for retrieval.")
    index, metadata = load_index_and_metadata(base_dir)

    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    try:
        while True:
            query = input("Enter query (or 'exit' to quit): ").strip()
            if query.lower() in {"exit", "quit", ""}:
                break
            top_k = 5
            print(f"Top_k: {top_k}")
            results = retrieve(query, top_k, model, index, chunks)
            for item in results:
                print(
                    f"#{item['rank']} score={item['score']:.4f} "
                    f"doc={item['document_id']} chunk={item['chunk_id']}"
                )
                print(f"   source={item['source_category']} | preview={item['text_preview']}")
    except KeyboardInterrupt:
        print("\nExiting retrieval loop.")


if __name__ == "__main__":
    main()
