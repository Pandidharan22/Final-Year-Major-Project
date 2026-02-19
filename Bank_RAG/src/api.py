"""
Step 10 – REST API Layer

Exposes the Banking RAG system as an HTTP service using FastAPI.

Endpoints
---------
GET  /health          Liveness probe; confirms the index and model are loaded.
POST /ask             Submit a banking query; returns the answer, source chunk IDs,
                      context length, retrieval confidence score, and confidence level.

Run with:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.retrieve import (  # noqa: E402
    CHUNKS_DIR,
    MODEL_NAME,
    build_model,
    load_chunks,
    load_index_and_metadata,
    set_determinism,
)
from src.rag import MODEL_ID, rag_answer  # noqa: E402

app = FastAPI(
    title="Banking RAG API",
    description="Retrieval-Augmented Generation API backed by RBI and SBI documents.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Startup: load heavy resources once and cache them on the app state
# ---------------------------------------------------------------------------

set_determinism()
_model = build_model()
_chunks = load_chunks(BASE_DIR / CHUNKS_DIR)
_index_loaded = False
try:
    _index, _metadata = load_index_and_metadata(BASE_DIR)
    _index_loaded = True
except (FileNotFoundError, RuntimeError) as exc:
    import warnings
    warnings.warn(f"Vector index could not be loaded at startup: {exc}", stacklevel=1)
    _index = None
    _metadata = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    context_length: int
    retrieval_confidence_score: float
    confidence_level: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness probe – confirms the vector index and embedding model are ready."""
    return {
        "status": "ok",
        "index_loaded": _index_loaded,
        "chunks_loaded": len(_chunks),
        "embedding_model": MODEL_NAME,
        "llm_model": MODEL_ID,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Submit a banking query.

    The system retrieves the most relevant document chunks, injects them as
    context into a prompt, and returns the LLM answer together with confidence
    metadata. Every call is logged to logs/rag_traces.jsonl.
    """
    if not _index_loaded:
        raise HTTPException(status_code=503, detail="Vector index is not available.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="'query' must not be empty.")

    top_k = max(1, min(request.top_k, 20))

    try:
        result = rag_answer(query, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        query=result["query"],
        answer=result["answer"],
        sources=result["sources"],
        context_length=result["context_length"],
        retrieval_confidence_score=result["retrieval_confidence_score"],
        confidence_level=result["confidence_level"],
    )

