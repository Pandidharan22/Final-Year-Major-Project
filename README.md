# Autonomous Self-Healing LLM Ops Platform (Banking Support)

An MVP for an **Autonomous Self-Healing LLM Ops Platform for Banking Customer Support** that enforces semantic validation, trust scoring, and deterministic guardrails over retrieval-augmented responses. The focus is reliability: measuring confidence, compressing risk, triggering self-healing retries, and exposing observability for banking-domain assistance.

## What This MVP Delivers
- **Deterministic ingestion + chunking** of RBI/SBI policy docs into structured context for banking QA.
- **Retrieval + semantic validation** with calibrated confidence scoring and refusal awareness.
- **Hallucination risk modeling** with refusal override, proportional penalties, and nonlinear base compression.
- **Single-shot self-healing retry** via top-k expansion when risk is high (non-refusal cases only).
- **Observability & analytics**: latency KPIs, evaluation metrics, and self-healing analytics (triggers, failure patterns, risk–confidence correlation, refusal distribution).
- **Dashboard**: Streamlit app for live queries, metrics, and guardrail insights.

## Architecture (MVP)
- **Data pipeline**: deterministic PDF ingestion (pypdf + pdfplumber fallback), cleaning, semantic chunking (500 words, 75 overlap), MiniLM embeddings, FAISS index.
- **Retrieval layer**: FAISS cosine search over normalized vectors; deterministic seeds and ordering.
- **LLM layer**: Mistral-7B-Instruct via HF router with chat-style messages; temperature 0.2, max_tokens 300.
- **Trust & risk**:
  - Confidence = 0.7 * mean similarity + 0.3 * score spread (high/medium/low tiers).
  - Base risk = (1 - confidence)^2 with proportional penalties; refusal path caps risk and blocks self-healing.
- **Self-healing**: one retry with top_k=8 adopted only if risk decreases.
- **Observability**: JSONL trace logging (append-only), evaluation metrics, dashboards for latency and self-healing analytics.

## Repository Layout (key paths)
- `src/ingest.py` — deterministic PDF ingestion.
- `src/chunk.py` — semantic chunking.
- `vector_store/` — FAISS index + metadata.
- `src/rag.py` — retrieval, LLM call, confidence/risk, self-healing.
- `dashboard/app.py` — Streamlit UI (live queries, metrics, self-healing analytics).
- `logs/rag_traces.jsonl` — append-only telemetry traces.
- `evaluation/` — evaluation dataset and metrics helpers.
- `DEVELOPMENT_LOG.md` — stepwise implementation log.

## Prerequisites
- Python 3.10+ with virtual environment (use the root `.venv`).
- Install deps: `pip install -r requirements.txt`.
- Hugging Face API token with access to `mistralai/Mistral-7B-Instruct-v0.2` (set `HF_API_TOKEN`).

## Quickstart
1. **Activate env**: `.\.venv\Scripts\activate` (PowerShell) or `source .venv/bin/activate` (Unix).
2. **Install**: `pip install -r requirements.txt`.
3. **Run CLI** (from `src/`): `python rag.py`.
4. **Run dashboard** (from repo root): `streamlit run dashboard/app.py`.

## Dashboard Features
- Live banking Q&A with confidence, risk, refusal, and latency breakdowns.
- Similarity bars for top-k retrieval.
- Gauges for confidence vs risk.
- Latency analytics: histogram, mean retrieval vs generation.
- Evaluation summary: accuracy, average confidence/risk, refusal rate, self-healing trigger rate.
- **Autonomous Self-Healing Analytics**:
  - Trigger distribution (pie + KPI)
  - Failure patterns (high risk, low confidence, refusals)
  - Risk vs confidence scatter with diagonal reference
  - Refusal distribution (pie + KPI)
  - Stability trend when timestamps become available (skips gracefully otherwise)

## Logging & Telemetry
- **Source**: `logs/rag_traces.jsonl` append-only JSONL.
- Fields include query, chunk IDs, similarity scores, confidence, risk, refusal flag, self-healing trigger, latencies, and retry outcomes.
- Logs are excluded from git by default to keep telemetry local.

## Trust & Risk Guardrails (MVP)
- Refusal-aware risk suppression; self-healing blocked on refusals.
- Quadratic base risk to reduce over-dominance of mid-uncertainty cases.
- Proportional penalties for confidence gap, spread gap, and answer/context ratio.
- High-risk triggers a single deterministic retry with expanded context; adopts only if risk drops.

## Determinism & Reproducibility
- Fixed seeds for Python/NumPy/torch; CPU inference.
- Sorted file and chunk ordering; static timestamps for artifacts.
- Deterministic prompt and generation parameters.

## Roadmap (next)
- Add timestamps to traces for stability trendlines.
- Broaden evaluation set and add regression checks for confidence/risk calibration.
- Expand self-healing strategies beyond top-k (prompt tightening, fallback models) while keeping determinism.

## Running Tests
- (Planned) Evaluation harness under `evaluation/` for retrieval/refusal metrics; extend as needed.

## License
Internal/MVP use; add license terms before distribution.
