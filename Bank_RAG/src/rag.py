import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from huggingface_hub import HfApi
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from src.retrieve import (  # noqa: E402
    CHUNKS_DIR,
    MODEL_NAME,
    build_model,
    load_chunks,
    load_index_and_metadata,
    retrieve,
    set_determinism,
)

MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
ROUTER_ENDPOINT = "https://router.huggingface.co"
PREFERRED_PROVIDER = os.getenv("HF_INFERENCE_PROVIDER")
DEFAULT_TOP_K = 5
MAX_PREVIEW_CHARS = 300
LOG_PATH = BASE_DIR / "logs" / "rag_traces.jsonl"
REFUSAL_PHRASE = "I don't have sufficient information"
EMBEDDING_MODEL = MODEL_NAME
SYSTEM_PROMPT = (
    "You are a banking assistant. Answer ONLY using the provided context. "
    "If the answer is not in the context, say: 'I don't have sufficient information in the provided documents.'"
)


def _run_single_pass(
    query: str,
    top_k: int,
    token: str,
    model,
    index,
    chunks: List[Dict[str, object]],
    chunk_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    results = retrieve(query, top_k, model, index, chunks)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

    chunk_ids = [item["chunk_id"] for item in results]
    chunk_texts = [chunk_map[cid]["text"] for cid in chunk_ids if cid in chunk_map]

    context = format_context(chunk_texts)
    prompt = prepare_prompt(context, query)

    generation_start = time.perf_counter()
    answer = call_hf_inference(token, prompt)
    generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0
    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    similarity_scores = [float(item.get("score", 0.0)) for item in results]
    mean_top_k_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    score_spread = similarity_scores[0] - similarity_scores[-1] if similarity_scores else 0.0
    retrieval_confidence_score = round((0.7 * mean_top_k_similarity) + (0.3 * score_spread), 4)
    if retrieval_confidence_score >= 0.55:
        confidence_level = "high"
    elif retrieval_confidence_score >= 0.40:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    context_length_chars = len(context)
    answer_length_chars = len(answer)
    answer_to_context_ratio = round(answer_length_chars / context_length_chars, 4) if context_length_chars else 0.0
    refusal_detected = REFUSAL_PHRASE in answer
    base_risk = (1 - retrieval_confidence_score) ** 2

    if refusal_detected:
        hallucination_risk_score = min(base_risk * 0.3, 0.30)
        hallucination_risk_score = round(hallucination_risk_score, 4)

        risk_level = "low"
        self_healing_trigger = False

    else:
        risk = base_risk

        # Scaled low-confidence penalty
        risk += 0.5 * (1 - retrieval_confidence_score) * 0.5

        # Scaled spread penalty
        spread_gap = max(0, 0.15 - score_spread)
        risk += 0.3 * spread_gap

        # Scaled ratio penalty
        ratio_gap = max(0, answer_to_context_ratio - 0.20)
        risk += 0.2 * ratio_gap

        risk = max(0, min(risk, 1))
        hallucination_risk_score = round(risk, 4)

        if hallucination_risk_score >= 0.70:
            risk_level = "high"
        elif hallucination_risk_score >= 0.45:
            risk_level = "medium"
        else:
            risk_level = "low"

        self_healing_trigger = (risk_level == "high")

    return {
        "answer": answer,
        "chunk_ids": chunk_ids,
        "similarity_scores": similarity_scores,
        "mean_top_k_similarity": mean_top_k_similarity,
        "score_spread": score_spread,
        "retrieval_confidence_score": retrieval_confidence_score,
        "confidence_level": confidence_level,
        "answer_to_context_ratio": answer_to_context_ratio,
        "hallucination_risk_score": hallucination_risk_score,
        "risk_level": risk_level,
        "self_healing_trigger": self_healing_trigger,
        "top_k": top_k,
        "context_length_chars": context_length_chars,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms,
        "answer_length_chars": answer_length_chars,
        "refusal_detected": refusal_detected,
        "model_name": MODEL_ID,
        "embedding_model": EMBEDDING_MODEL,
    }


def build_chunk_map(chunks: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {entry.get("chunk_id", ""): entry for entry in chunks}


def format_context(chunk_texts: List[str]) -> str:
    lines: List[str] = []
    for idx, text in enumerate(chunk_texts, start=1):
        lines.append(f"=== Context Chunk {idx} ===")
        lines.append(text)
    return "\n".join(lines)


def append_trace(trace: Dict[str, object]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False))
        handle.write("\n")


def prepare_prompt(context: str, query: str) -> str:
    # User message content combining context and question for conversational schema
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def resolve_provider(model_id: str, preferred: Optional[str] = None) -> str:
    """
    Pick an inference provider that is actually mapped for the model.

    The HF router returns 404 when the model has no provider mapping (e.g., gemma-2b-it).
    Fail fast with a helpful error instead of letting the request fail deep in the stack.
    """

    info = HfApi().model_info(model_id, expand=["inferenceProviderMapping"])
    mappings = info.inference_provider_mapping or []

    if preferred:
        for mapping in mappings:
            if mapping.provider == preferred:
                return preferred
        available = ", ".join(sorted({m.provider for m in mappings})) or "none"
        raise RuntimeError(
            f"Preferred provider '{preferred}' is not available for model '{model_id}'. "
            f"Available providers: {available}."
        )

    if mappings:
        return mappings[0].provider

    raise RuntimeError(
        "No inference providers are registered for this model on the HF router. "
        "Choose a model with an inferenceProviderMapping (e.g., google/gemma-2-9b-it) "
        "or deploy your own endpoint and set HF_MODEL_ID and HF_INFERENCE_PROVIDER."
    )


def call_hf_inference(token: str, prompt: str) -> str:
    try:
        provider = resolve_provider(MODEL_ID, PREFERRED_PROVIDER)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(str(exc)) from exc

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    url = f"{ROUTER_ENDPOINT}/{provider}/v1/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:  # noqa: BLE001
        raise RuntimeError(f"request failed for {url}: {exc!r}") from exc

    if response.status_code == 429:
        raise RuntimeError("Rate limited by Hugging Face Inference API (429)")
    if response.status_code == 404:
        raise RuntimeError(f"endpoint not found (404) at {url}: {response.text}")
    if response.status_code == 401:
        raise RuntimeError(f"unauthorized (401) at {url}: {response.text}")
    if not response.ok:
        raise RuntimeError(
            f"HF inference failed at {url}: status={response.status_code}, body={response.text}"
        )

    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if choices and isinstance(choices, list) and choices and choices[0].get("message", {}).get("content"):
        return str(choices[0]["message"]["content"]).strip()

    print(f"Raw HF response (unexpected format): {data}")
    raise RuntimeError(f"Unexpected HF response format from provider '{provider}'")


def run_rag_pipeline(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, object]:
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

    first_run = _run_single_pass(query, top_k, token, model, index, chunks, chunk_map)

    retry_attempted = False
    retry_top_k: Optional[int] = None
    retry_improved = False
    retry_run: Optional[Dict[str, object]] = None

    if first_run["self_healing_trigger"] and not first_run["refusal_detected"]:
        retry_attempted = True
        retry_top_k = 8
        retry_run = _run_single_pass(query, retry_top_k, token, model, index, chunks, chunk_map)
        if retry_run["hallucination_risk_score"] < first_run["hallucination_risk_score"]:
            retry_improved = True
            final_run = retry_run
        else:
            final_run = first_run
    else:
        final_run = first_run

    trace = {
        "query": query,
        "retrieved_chunk_ids": first_run["chunk_ids"],
        "similarity_scores": first_run["similarity_scores"],
        "mean_top_k_similarity": first_run["mean_top_k_similarity"],
        "score_spread": first_run["score_spread"],
        "retrieval_confidence_score": first_run["retrieval_confidence_score"],
        "confidence_level": first_run["confidence_level"],
        "answer_to_context_ratio": first_run["answer_to_context_ratio"],
        "hallucination_risk_score": first_run["hallucination_risk_score"],
        "risk_level": first_run["risk_level"],
        "self_healing_trigger": first_run["self_healing_trigger"],
        "top_k": int(first_run["top_k"]),
        "context_length_chars": first_run["context_length_chars"],
        "retrieval_latency_ms": first_run["retrieval_latency_ms"],
        "generation_latency_ms": first_run["generation_latency_ms"],
        "total_latency_ms": first_run["total_latency_ms"],
        "answer_length_chars": first_run["answer_length_chars"],
        "refusal_detected": first_run["refusal_detected"],
        "model_name": MODEL_ID,
        "embedding_model": EMBEDDING_MODEL,
        "retry_attempted": retry_attempted,
        "retry_top_k": retry_top_k,
        "retry_improved": retry_improved,
        "final_risk_score": final_run["hallucination_risk_score"],
        "final_risk_level": final_run["risk_level"],
    }

    if retry_run:
        trace.update(
            {
                "retry_retrieved_chunk_ids": retry_run["chunk_ids"],
                "retry_similarity_scores": retry_run["similarity_scores"],
                "retry_mean_top_k_similarity": retry_run["mean_top_k_similarity"],
                "retry_score_spread": retry_run["score_spread"],
                "retry_retrieval_confidence_score": retry_run["retrieval_confidence_score"],
                "retry_confidence_level": retry_run["confidence_level"],
                "retry_answer_to_context_ratio": retry_run["answer_to_context_ratio"],
                "retry_hallucination_risk_score": retry_run["hallucination_risk_score"],
                "retry_risk_level": retry_run["risk_level"],
                "retry_self_healing_trigger": retry_run["self_healing_trigger"],
                "retry_context_length_chars": retry_run["context_length_chars"],
                "retry_retrieval_latency_ms": retry_run["retrieval_latency_ms"],
                "retry_generation_latency_ms": retry_run["generation_latency_ms"],
                "retry_total_latency_ms": retry_run["total_latency_ms"],
                "retry_answer_length_chars": retry_run["answer_length_chars"],
                "retry_refusal_detected": retry_run["refusal_detected"],
            }
        )

    append_trace(trace)

    return {
        "answer": final_run["answer"],
        "retrieval_confidence": final_run["retrieval_confidence_score"],
        "confidence_level": final_run["confidence_level"],
        "hallucination_risk": final_run["hallucination_risk_score"],
        "risk_level": final_run["risk_level"],
        "self_healing_triggered": final_run["self_healing_trigger"],
        "refusal_detected": final_run["refusal_detected"],
        "latency": {
            "retrieval_ms": final_run["retrieval_latency_ms"],
            "generation_ms": final_run["generation_latency_ms"],
            "total_ms": final_run["total_latency_ms"],
        },
        "similarity_scores": final_run["similarity_scores"],
        "retrieved_chunk_ids": final_run["chunk_ids"],
        "query": query,
        "context_length": final_run["context_length_chars"],
        "top_k": int(final_run["top_k"]),
    }


def rag_answer(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, object]:
    result = run_rag_pipeline(query, top_k)
    return {
        "query": query,
        "answer": result.get("answer", ""),
        "sources": result.get("retrieved_chunk_ids", []),
        "context_length": result.get("context_length", 0),
    }


def main() -> None:
    set_determinism()
    base_dir = Path(__file__).resolve().parent.parent
    load_dotenv(base_dir / ".env")
    token_present = bool(os.getenv("HF_API_TOKEN", "").strip())
    if not token_present:
        print("HF_API_TOKEN is missing. Please set it in Bank_RAG/.env.")
        return

    print(f"Using HF model: {MODEL_ID}")

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
            try:
                result = run_rag_pipeline(query, top_k)
                print(f"Injected chunk IDs: {result.get('retrieved_chunk_ids', [])}")
                print(f"Context length: {result.get('context_length', 0)} characters")
                print("Answer:\n", result.get("answer", ""))
            except RuntimeError as exc:
                print(f"Error: {exc}")
    except KeyboardInterrupt:
        print("\nExiting RAG CLI.")


if __name__ == "__main__":
    main()
