import json
import os
from pathlib import Path
from typing import Dict, List

import requests
import torch
from dotenv import load_dotenv

from src.retrieve import (
    CHUNKS_DIR,
    build_model,
    load_chunks,
    load_index_and_metadata,
    retrieve,
    set_determinism,
)

INFERENCE_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
DEFAULT_TOP_K = 5
MAX_PREVIEW_CHARS = 300


def build_chunk_map(chunks: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {entry.get("chunk_id", ""): entry for entry in chunks}


def format_context(chunk_texts: List[str]) -> str:
    lines: List[str] = []
    for idx, text in enumerate(chunk_texts, start=1):
        lines.append(f"=== Context Chunk {idx} ===")
        lines.append(text)
    return "\n".join(lines)


def prepare_prompt(context: str, query: str) -> str:
    system = (
        "You are a banking assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not in the context, say:\n"
        "\"I don't have sufficient information in the provided documents.\""
    )
    parts = [
        f"SYSTEM:\n{system}",
        f"CONTEXT:\n{context}",
        f"QUESTION:\n{query}",
    ]
    return "\n\n".join(parts)


def call_hf_inference(token: str, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.2,
            "max_new_tokens": 300,
            "return_full_text": False,
        },
    }
    response = requests.post(INFERENCE_URL, headers=headers, json=payload, timeout=60)
    if response.status_code == 429:
        raise RuntimeError("Rate limited by Hugging Face Inference API (429)")
    if not response.ok:
        raise RuntimeError(f"HF inference failed: status={response.status_code}, body={response.text}")

    data = response.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"]).strip()
    raise RuntimeError(f"Unexpected HF response format: {data}")


def rag_answer(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, object]:
    base_dir = Path(__file__).resolve().parent.parent
    load_dotenv(base_dir / ".env")
    token = os.getenv("HF_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_API_TOKEN is missing. Please set it in Bank_RAG/.env.")

    set_determinism()
    model = build_model()
    chunks = load_chunks(base_dir / CHUNKS_DIR)
    chunk_map = build_chunk_map(chunks)
    index, metadata = load_index_and_metadata(base_dir)

    results = retrieve(query, top_k, model, index, chunks)
    chunk_ids = [item["chunk_id"] for item in results]
    chunk_texts = [chunk_map[cid]["text"] for cid in chunk_ids if cid in chunk_map]

    context = format_context(chunk_texts)
    prompt = prepare_prompt(context, query)

    answer = call_hf_inference(token, prompt)

    return {
        "query": query,
        "answer": answer,
        "sources": chunk_ids,
        "context_length": len(context),
    }


def main() -> None:
    set_determinism()
    base_dir = Path(__file__).resolve().parent.parent
    load_dotenv(base_dir / ".env")
    token_present = bool(os.getenv("HF_API_TOKEN", "").strip())
    if not token_present:
        print("HF_API_TOKEN is missing. Please set it in Bank_RAG/.env.")
        return

    model = build_model()
    chunks = load_chunks(base_dir / CHUNKS_DIR)
    chunk_map = build_chunk_map(chunks)
    index, metadata = load_index_and_metadata(base_dir)

    print("Embedding dimension:", model.get_sentence_embedding_dimension())
    try:
        while True:
            query = input("Enter question (or 'exit' to quit): ").strip()
            if query.lower() in {"exit", "quit", ""}:
                break
            top_k = DEFAULT_TOP_K
            print(f"Top_k: {top_k}")
            results = retrieve(query, top_k, model, index, chunks)
            chunk_ids = [item["chunk_id"] for item in results]
            chunk_texts = [chunk_map[cid]["text"] for cid in chunk_ids if cid in chunk_map]
            print(f"Injected chunk IDs: {chunk_ids}")
            context = format_context(chunk_texts)
            print(f"Context length: {len(context)} characters")
            prompt = prepare_prompt(context, query)
            try:
                answer = call_hf_inference(os.getenv("HF_API_TOKEN", ""), prompt)
                print("Answer:\n", answer)
            except RuntimeError as exc:
                print(f"Error: {exc}")
    except KeyboardInterrupt:
        print("\nExiting RAG CLI.")


if __name__ == "__main__":
    main()
