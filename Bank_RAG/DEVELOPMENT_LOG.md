# Development Log

## Step 1 – Project Bootstrap & Environment Setup
- Intent: Establish baseline tooling requirements and tracking artifacts for the BANK_RAG RAG pipeline before coding begins.
- Files created or modified: requirements.txt, .gitignore, DEVELOPMENT_LOG.md
- Assumptions: Work will proceed inside a managed Python virtual environment (not created in this step); vector and embedding artifacts will be regenerated locally as needed.
- Observations / notes: Dependency list kept minimal and version-pinned to ensure reproducible installs; gitignore configured to keep large/generated artifacts out of version control.

## Step 2 – Dependency Fix & Installation
- Intent: Resolve the pip installation failure seen in the terminal and complete dependency installation inside the project virtual environment.
- Files created or modified: requirements.txt
- Assumptions: PyPI access is available and installation occurs inside the existing .venv configured for this workspace.
- Observations / notes: Updated `faiss-cpu` pin to 1.13.2 (latest Windows wheel) and reran `pip install -r requirements.txt` successfully after upgrading pip to 26.0.1.

## Step 2 – Deterministic PDF Ingestion
- Intent: Implement deterministic PDF ingestion that extracts, cleans, and serializes text from raw RBI master circular and SBI PDFs into structured JSON outputs.
- Files created: src/ingest.py
- Assumptions: Input PDFs reside under data/raw/rbi_master_circulars and data/raw/sbi; output directory data/cleaned/ is writable.
- Observations / notes: Processing order is fully sorted; page-number lines and excess whitespace are removed while preserving paragraph breaks; errors per file are logged without halting the run.

## Step 2 (Correction) – Determinism & Directory Fix
- What was wrong: RBI source directory name could include a space instead of an underscore, causing files to be skipped; dynamic timestamps in `extracted_at` violated determinism.
- What was fixed: Ingestion now resolves `data/raw/rbi_master_circulars` to `data/raw/rbi/master_circulars` deterministically and replaces `extracted_at` with the constant `STATIC_INGESTION_TIMESTAMP`.
- Why determinism matters: Stable inputs and outputs ensure reproducible runs, predictable JSON artifacts, and reliable downstream indexing.

## Step 3 – Controlled Semantic Chunking
- Intent: Derive deterministic, overlapping semantic chunks (~500 words, 75-word overlap) from cleaned documents to feed downstream indexing without altering ingestion outputs.
- Files created: src/chunk.py; data/chunks/*_chunks.json generated from data/cleaned/*.json.
- Chunk size and overlap rationale: 500-word windows capture coherent sections; 75-word overlap preserves cross-boundary context while keeping chunk counts predictable and stable across runs.
- Observations: Processing order sorted by filename for determinism; intraword splits compacted before chunking; paragraph breaks preserved where possible; `savings_account_general_terms_and_conditions` contained no words after cleaning and produced zero chunks.

## Environment Consolidation – Single Root Virtual Environment
- What was removed: Deleted Bank_RAG/.venv to eliminate nested environments and rely solely on the root .venv.
- Why single venv is required: Ensures one deterministic interpreter path for tooling, installs, and automation across the entire repository, avoiding drift between duplicated environments.
- Confirmation: Activated root .venv, upgraded pip, and installed Bank_RAG/requirements.txt into the root environment successfully.

## Step 3 Correction – Robust Cleaning & Zero-Chunk Protection
- Root cause analysis: ASCII-only stripping in chunking was overly aggressive and could discard valid printable characters, and the absence of safeguards allowed documents with empty or depleted text (e.g., `savings_account_general_terms_and_conditions`) to yield zero chunks.
- Cleaning adjustments: Replaced ASCII stripping with a control-character filter that preserves printable Unicode and newlines, normalized newlines, and retained paragraph-aware token fixes; added diagnostics for raw/cleaned character counts and word totals per document.
- Safety behavior: If cleaned characters exist but yield zero words, chunking now raises explicitly; truly empty documents emit a single deterministic empty chunk to avoid zero-chunk outputs.
- Final chunk counts (post-fix run): interest_rates=7, kyc_master_direction=68, prudential_norms=21, savings_account_general_terms_and_conditions=1, schedule_of_charges=1; total=98 chunks.

## Step 2 Adjustment – Excluding Image-Only PDFs
- Root cause: `savings_account_general_terms_and_conditions.pdf` is image-only; both pypdf and the pdfplumber fallback returned zero characters, and OCR was intentionally rejected to avoid nondeterministic, heavyweight dependencies.
- Fallback implementation: Added pdfplumber as the deterministic fallback after pypdf, logged raw and cleaned character counts plus the extraction method, and skip writing outputs when cleaned text remains empty while removing any stale JSON artifact.
- Determinism and scope: Extraction order unchanged; only text-bearing PDFs are persisted to data/cleaned/, and image-only/unreadable files are logged and excluded to keep the dataset deterministic and text-only.
- Extraction outcomes (post-adjustment): interest_rates (pypdf, raw=24551, cleaned=23806); kyc_master_direction (pypdf, raw=241476, cleaned=232569); prudential_norms (pypdf, raw=71341, cleaned=70056); schedule_of_charges (pypdf, raw=3104, cleaned=2886); savings_account_general_terms_and_conditions skipped as image-only.

## Step 4 – Embedding Layer & FAISS Index
- Model used: sentence-transformers/all-MiniLM-L6-v2 on CPU with deterministic seeds.
- Vector dimension: 384; total chunks embedded: 97 across 4 documents (interest_rates=7, kyc_master_direction=68, prudential_norms=21, schedule_of_charges=1).
- Determinism strategy: Fixed Python/NumPy/torch seeds, deterministic torch algorithms, sorted chunk files and chunk indices prior to encoding, CPU-only inference, and normalized embeddings.
- Outputs: FAISS index stored at vector_store/index.faiss and metadata at vector_store/metadata.json recording model, chunk count, documents indexed, and STATIC_EMBED_TIMESTAMP.

## Step 5 – Semantic Retrieval Engine
- Retrieval design: Load FAISS index and chunk metadata, embed queries with all-MiniLM-L6-v2, and return top-k nearest chunks with rank, IDs, source category, score, and 300-char preview.
- Similarity metric: Cosine via inner-product over normalized embeddings using faiss.IndexFlatIP.
- Determinism strategy: Fixed seeds for Python/NumPy/torch, deterministic torch algorithms, CPU-only inference, and sorted chunk loading to maintain stable index-to-metadata alignment.
- Observations: CLI prompt provides ranked results; embedding dimension 384; default top_k=5; smoke test query returned expected interest_rates chunks with stable scores.

## Step 6 – Hugging Face Gemma RAG Integration
- Model/endpoint: google/gemma-2b-it via Hugging Face Inference API at https://api-inference.huggingface.co/models/google/gemma-2b-it.
- Deterministic parameters: temperature=0.2, max_new_tokens=300, return_full_text=false; retrieval top_k=5 with sorted chunk loading and fixed seeds via existing retrieval helpers.
- Integration design: CLI prompt loads HF_API_TOKEN from .env, gathers top_k chunks through retrieve(), injects structured context blocks, and sends a single POST to the inference API; returns answer plus source chunk IDs.
- Failure handling: Clear error when token missing, explicit status reporting for non-200 responses, 429 rate-limit warning, and graceful KeyboardInterrupt exit.
- Observations: Latency not measured here (depends on remote API); context length and injected chunk IDs are printed for visibility.

## Step 6 Adjustment – Switch to Mistral-7B-Instruct (HF Router Compatible)
- Gemma router limitation: google/gemma-2b-it is not available through HF router free providers, causing 404/401 failures.
- New model choice: mistralai/Mistral-7B-Instruct-v0.2.
- Rationale: Use a model with free-tier compatible HF router access while keeping the same payload structure, deterministic generation parameters, and retrieval pipeline.

## Step 6 Fix – Switch to Conversational Task Schema for Mistral
- Provider limitation: HF router maps Mistral-7B-Instruct via providers that require the conversational task, so text-generation payloads fail.
- Payload format change: Adopted OpenAI-style messages array with system/user roles (no `inputs` or `parameters`), temperature=0.2, max_tokens=300.
- Response structure adjustment: Parse `choices[0].message.content` and log raw JSON when `choices` is absent for easier troubleshooting.

## Milestone 1 – Fully Functional Deterministic Banking RAG System
### Architecture Overview
- Ingestion: deterministic PDF parsing with fallback handling.
- Cleaning: control-character filtering and newline normalization to preserve printable text.
- Chunking: 500-word windows with 75-word overlap for stable context slices.
- MiniLM embeddings (384-dim): sentence-transformers/all-MiniLM-L6-v2 on CPU with fixed seeds.
- FAISS index: IndexFlatIP over normalized vectors with aligned metadata.
- HF Router LLM: Mistral-7B-Instruct via router.huggingface.co conversational API.
- Deterministic guardrails: fixed seeds, ordered processing, static timestamps, and stable prompts.

### Problems Encountered & Solutions
- PDF Extraction Failure: savings_account PDF was image-only; excluded rather than adding OCR to maintain determinism and scope.
- Zero Chunk Issue: over-aggressive cleaning removed content; added diagnostics and safeguards to prevent empty outputs.
- HF API Endpoint Deprecation: api-inference endpoint returned 410; migrated to router.huggingface.co.
- Gemma Model Unsupported on Router: no inference providers registered; switched to Mistral-7B-Instruct.
- Payload Format Mismatch: text-generation schema rejected by conversational providers; adopted chat-style messages payload.

### Determinism Strategy
- Static timestamps for artifacts.
- Sorted processing of files, chunks, and retrieval inputs.
- Fixed chunk size and overlap parameters.
- Generation settings locked: temperature=0.2, max_tokens=300.
- Controlled context injection with structured prompts.

### Current System Capabilities
- Accurate retrieval from FAISS-backed index.
- Grounded answers constrained to supplied context.
- Explicit refusals for out-of-scope or missing-context queries.

## Step 7 – Query Trace Logging & Observability Layer
- Logged telemetry fields: query, retrieved_chunk_ids, similarity_scores, top_k, context_length_chars, retrieval_latency_ms, generation_latency_ms, total_latency_ms, answer_length_chars, refusal_detected, model_name, embedding_model.
- Metric rationale: retrieval/generation/total latency expose performance hotspots; context and answer lengths track prompt/response size; refusal flag surfaces guardrail activations; similarity scores and chunk IDs enable replay and quality inspection; model metadata keeps runs attributable.
- Observability design: append-only JSONL at logs/rag_traces.jsonl with stable field order, no timestamps or random IDs, directory auto-created for deterministic, replayable traces.

## Step 8 – Retrieval Confidence Scoring
- Confidence formula: confidence = (0.7 * mean_top_k_similarity) + (0.3 * score_spread), rounded to 4 decimals.
- Rationale: emphasize overall similarity strength (mean) while incorporating dispersion (spread) to reflect ranking separation.
- Classification thresholds: >= 0.55 => high; >= 0.40 => medium; else low.
- Intended use: drive a self-healing loop to adapt behavior (e.g., prompt tweaks, re-query, or fallback) when confidence is low, without introducing randomness.

## Milestone 2 – Observability & Retrieval Confidence Layer
### Query Trace Logging (Step 7)
- Telemetry: query, retrieved_chunk_ids, similarity_scores, top_k, context_length_chars, retrieval_latency_ms, generation_latency_ms, total_latency_ms, answer_length_chars, refusal_detected, model_name, embedding_model.
- Latency tracking: separate retrieval, generation, and total timings via perf_counter for hotspot analysis.
- Refusal detection: flag when the deterministic refusal phrase is present.
- Model metadata: logs include LLM and embedding model identifiers for attribution.
- JSONL append-only: stable field order, no timestamps or random IDs, replayable for audits.

### Retrieval Confidence Scoring (Step 8)
- Metrics: mean_top_k_similarity and score_spread.
- Confidence formula: retrieval_confidence_score = (0.7 * mean_top_k_similarity) + (0.3 * score_spread), rounded to 4 decimals.
- Classification: >= 0.55 → high; >= 0.40 → medium; else low.

### Observations
- In-domain query → medium confidence.
- Out-of-domain query → low confidence.
- Confidence correlates with refusal detection for unsupported questions.

### Engineering Rationale
- Confidence is a precursor signal for self-healing behaviors (retry, re-prompt, or fallback).
- Structured logging enables offline evaluation and regression checks.
- Deterministic scoring and append-only traces ensure reproducibility.

## Step 9 – Evaluation Dataset & Metrics Engine
- Dataset: evaluation/questions.json with labeled in-domain (RBI, schedule_of_charges) and out-of-domain queries for balanced assessment.
- Metrics: total queries, in/out counts, retrieval_accuracy (in-domain hit rate), refusal_accuracy (out-of-domain refusals), mean confidence segmented by domain.
- Why these metrics: measure grounding quality (retrieval_accuracy), guardrail correctness (refusal_accuracy), and confidence calibration to inform future self-healing loops.

## Step 10 – REST API Layer
- Intent: Expose the Banking RAG system as a deployable HTTP service so external clients (frontends, scripts, other services) can query it without running Python directly.
- Files created or modified: src/api.py, requirements.txt (added fastapi==0.115.6, uvicorn==0.34.0).
- Endpoints: GET /health (liveness probe; confirms index, model, and chunk counts) and POST /ask (accepts {"query", "top_k"}, returns answer, sources, context_length, retrieval_confidence_score, and confidence_level).
- Design decisions:
  - Heavy resources (embedding model, FAISS index, chunk map) are loaded once at startup and reused across requests to minimise per-request latency.
  - The /ask handler delegates to rag_answer() to avoid duplicating confidence-scoring and trace-logging logic; rag_answer() was updated to include retrieval_confidence_score and confidence_level in its return value.
  - top_k is clamped to [1, 20] to prevent abuse.
  - Missing HF_API_TOKEN or unavailable index returns a 503; LLM failures surface as 502.
  - Index-load failures at startup use specific exception types (FileNotFoundError, RuntimeError) and emit a warning for easier diagnostics.
  - All calls continue to be logged to logs/rag_traces.jsonl for observability continuity.
- Run command: uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload (from Bank_RAG/ directory).
