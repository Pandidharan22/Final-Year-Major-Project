## BANK_RAG

End-to-end Retrieval-Augmented Generation (RAG) system for banking domain FAQs and policy documents. The project ingests curated PDFs/JSON, builds a FAISS vector index with MiniLM embeddings, and serves guardrailed answers via a Hugging Face Inference endpoint (Mistral-7B-Instruct by default). An observability-focused Streamlit dashboard surfaces live query metrics, latency, and logs.

### Key Features
- Deterministic RAG pipeline: fixed seeds, stable ordering, no random UUIDs.
- FAISS IndexFlatIP with sentence-transformers/all-MiniLM-L6-v2 embeddings (384-dim).
- HF Router-backed chat completion (Mistral-7B-Instruct default; configurable provider).
- Guardrails: refusal phrase enforcement, refusal-aware hallucination risk, self-healing trigger flag.
- Telemetry: JSONL traces with similarity scores, confidence, risk, refusal flag, latency.
- Streamlit dashboard: live query UI, metric panels, log explorer (latest entries).
- Evaluation hooks and tests (see `tests/`).

### Repository Structure
- `src/` – ingestion and RAG pipeline (`ingest.py`, `rag.py`, `retrieve.py`, etc.).
- `data/` – raw and cleaned documents; chunks and prepared datasets.
- `vector_store/` – FAISS index and metadata.
- `logs/rag_traces.jsonl` – append-only telemetry traces.
- `dashboard/` – Streamlit app (`app.py`) and helpers (`utils.py`).
- `prompts/` – prompt templates and related assets.
- `tests/` – evaluation and regression tests.

### Prerequisites
- Python 3.10+ (project uses 3.13 in the provided venv).
- Git and a configured Hugging Face API token with access to the chosen model/provider.
- Recommended: virtual environment (venv) located at `.venv/`.

### Setup
```bash
# From repo root
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` at repo root:
```
HF_API_TOKEN=your_hf_token
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
HF_INFERENCE_PROVIDER=  # optional, e.g., azure, aws, runpod
```

### Data Ingestion (if needed)
```bash
cd src
python ingest.py
```
Outputs: chunked data under `data/chunks/` and FAISS index + metadata under `vector_store/`.

### Running the RAG CLI
```bash
cd src
python rag.py
```
Enter a question at the prompt; traces append to `logs/rag_traces.jsonl`.

### Streamlit Dashboard
```bash
cd Bank_RAG
streamlit run dashboard/app.py
```
Dashboard shows live query inputs, returned answers, confidence/risk metrics, latency breakdown (retrieval, generation, total), and a log explorer (latest 10 entries).

### RAG Pipeline API
`run_rag_pipeline(query: str) -> dict` (in `src/rag.py`) returns:
- `answer`: str
- `retrieval_confidence`: float
- `confidence_level`: str
- `hallucination_risk`: float
- `risk_level`: str
- `self_healing_triggered`: bool
- `refusal_detected`: bool
- `latency`: {`retrieval_ms`, `generation_ms`, `total_ms`}
- `similarity_scores`: list[float]
- `retrieved_chunk_ids`: list[str]

Telemetry logging to `logs/rag_traces.jsonl` remains unchanged and is appended per request.

### Evaluation
- Tests and evaluation scripts live under `tests/` (invoke with `python -m pytest` or custom eval runner if provided).
- Use the same `.env` and FAISS artifacts for consistent results.

### Guardrails and Risk Logic
- Refusal phrase: "I don't have sufficient information in the provided documents." (checked via substring)
- Risk scoring is refusal-aware: refusal reduces risk and disables self-healing trigger; penalties apply only to non-refusal answers.
- Self-healing trigger is flagged when risk is high (for future retry/fallback hooks).

### Logging and Observability
- `logs/rag_traces.jsonl` captures query, chunk IDs, similarity scores, confidence, risk, refusal flag, latency, and models used.
- Streamlit dashboard reads the latest 10 entries for quick inspection.

### Troubleshooting
- Missing token: set `HF_API_TOKEN` in `.env`.
- Provider errors (404/429/401): verify `HF_MODEL_ID` and `HF_INFERENCE_PROVIDER` mapping on HF Router; consider rate limits.
- Empty or low-confidence answers: confirm FAISS index/chunks exist under `vector_store/` and `data/chunks/`.
- Dashboard errors: ensure the app is launched from `Bank_RAG` with the venv active and that logs file is present (empty is fine).

### Contributing Workflow
- Keep changes deterministic; avoid introducing randomness in retrieval or risk scoring.
- Add/update tests for regressions where applicable.
- Use meaningful commits (e.g., `feat: ...`, `fix: ...`, `docs: ...`).

### License
Project code is provided for educational and internal use. Review repository terms or add a LICENSE file if distribution is needed.
