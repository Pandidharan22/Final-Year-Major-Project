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
