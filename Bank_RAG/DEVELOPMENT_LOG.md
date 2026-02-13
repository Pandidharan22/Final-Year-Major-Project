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
