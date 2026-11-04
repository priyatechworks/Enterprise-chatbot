# Enterprise Graph RAG demo

A multimodal Agentic RAG demo for the research paper **Attention Is All You Need**. It ingests PDFs with Docling, separates text, tables, formulas and figures, preserves item-level page/section/provenance metadata, and indexes text and figure vectors in Neo4j. A bounded query planner routes each question to the requested content types; Neo4j vector search + Lucene BM25 + RRF + graph expansion gather the evidence. SQLite stores sessions and the document catalog. The app remains one Streamlit frontend and one FastAPI backend.

## Architecture and runtime flow

```mermaid
flowchart TB
  PDF[Transformer research PDF] --> Docling[Docling + HybridChunker]
  Docling --> Text[Text chunks]
  Docling --> Tables[Tables + structured JSON]
  Docling --> Formulas[Formula items / optional LaTeX enrichment]
  Docling --> Figures[Figure crops + captions]
  Figures --> Gemini[Optional Gemini descriptions]
  Text --> NomicText[Nomic Text embeddings]
  Tables --> NomicText
  Formulas --> NomicText
  Gemini --> NomicText
  Figures --> NomicVision[Nomic Vision embeddings]
  NomicText --> Neo4j[Neo4j vector + BM25 + typed graph]
  NomicVision --> Neo4j
  Text --> Manifest[Per-item metadata and artifacts]
  Tables --> Manifest
  Formulas --> Manifest
  Figures --> Manifest
  User[User question] --> Planner[Agent query planner]
  Planner --> Router[Text / table / figure / formula router]
  Router --> Retrieval[Neo4j vector + BM25]
  Retrieval --> ACL[Cypher role filter]
  ACL --> RRF[RRF + graph expansion]
  RRF --> Gate[Evidence and safety gate]
  Gate --> LLM[Grounded answer model]
  LLM --> Response[Answer + citations + media]
```

1. Admin uploads PDFs through Streamlit. The backend checks file type, header, size, and the selected `Public` or `Admin-only` classification. The default is `Admin-only`.
2. Docling parses the PDF. HybridChunker keeps heading context, page provenance, and table headers. The existing manifest remains useful for initial sample-folder documents, but user uploads do not need a manifest row.
3. Text, table and formula records use Nomic Embed Text v1.5; extracted figures use Nomic Embed Vision v1.5. These v1.5 encoders share a 768-dimensional space, so natural-language queries can retrieve figure vectors from the same Neo4j vector index. Text/table/formula/metadata artifacts are persisted under `data/extracted/`; figure crops remain under `data/assets/`. Every item retains its type, page, section, caption, Docling reference, bounding box/provenance and file path. Picture extraction is enabled by default; Gemini descriptions require `GEMINI_API_KEY`. Formula enrichment is enabled by default for the demo; OCR remains opt-in for scanned PDFs.
4. The query planner creates a separate retrieval task for each requested type: text, tables, figures and formulas. Each task searches only that type with role-filtered Neo4j vector + Lucene BM25, RRF and authorized graph expansion. The merge reserves coverage for the strongest available requested types and returns at most three chunks overall. Typed `Section` and `KnowledgeItem` nodes are linked by `HAS_SECTION`, `HAS_CONTENT`, `CONTAINS_*`, `NEXT` and source-grounded `RELATED_ON_PAGE` relationships.
5. The evidence gate and prompt-injection checks run before answer generation. The OpenAI answer model receives at most three authorized items: text and table data as text, formulas as LaTeX, and selected figure crops as actual image inputs. Prior user questions are supplied only for reference resolution; previous assistant answers are excluded. The response cites the source filename and PDF page, or abstains.

The current controller is a bounded Python query planner, not LangGraph. When the configured LLM is available, it returns a structured JSON retrieval plan; otherwise deterministic keyword rules route the request. LangGraph is not required for this implementation.

## Access policies and document replacement

- `Public`: Admin and Employee may list and retrieve the document.
- `Admin-only`: Admin may list and retrieve the document; Employee receives no content.
- Admin can access either policy. Existing HR authentication is retained only for backward compatibility; new policy assignments use Admin and Employee.
- Every new document receives an automatically generated ID. Re-upload by the same filename replaces its graph nodes, relationships, full-text/vector index entries, BM25-indexed content, SQLite metadata, and stale extracted image assets without duplicating chunks.
- Admin uploads are stored under ignored `data/documents/`. The seeded sample-folder import reads `data/sample_docs/*.pdf` and uses manifest policy as its seed default; unknown PDFs imported from that folder default to Admin-only.
- Only Admin may call either ingestion endpoint. Backend checks remain authoritative even if the UI is bypassed.

## Setup (Windows / PowerShell)

Requirements: Python 3.10/3.11, 8 GB RAM minimum (16 GB preferred), several GB free for model caches, and a reachable Neo4j AuraDB instance. Docker is not required.

```powershell
cd D:\Users\DELL\Documents\enterprise-rag-demo
python -m venv .venv
New-Item -ItemType Directory -Force data\tmp,data\pip-cache,data\model_cache | Out-Null
$env:TEMP=(Resolve-Path data\tmp).Path
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=(Resolve-Path data\pip-cache).Path
$env:HF_HOME=(Resolve-Path data\model_cache).Path
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you do not already have a local `.env`. Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` and a long random `TOKEN_SECRET`. Set `DEMO_PASSWORD` to the demo password you want. Set `LLM_API_KEY` for the OpenAI API. The default endpoint is `https://api.openai.com/v1` and the default vision-capable model is `gpt-4o-mini`; `LLM_BASE_URL` and `LLM_MODEL` can override these defaults. Add `GEMINI_API_KEY` for figure descriptions (optional; captions are used otherwise). Set `DOCLING_ENABLE_FORMULA_ENRICHMENT=true` only when ready to test Docling's formula model. Never commit `.env` or uploaded documents.

## Run the application

Open two PowerShell terminals in the project directory.

Terminal 1 - API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Terminal 2 - Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/ui.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Sign in as `admin` to upload PDFs and choose access, or sign in as `employee` to query Public documents. Initial demo credentials come from `DEMO_PASSWORD` in `.env` (the code fallback is `RagDemo!2026`). To index the seeded sample, use the Admin sidebar action **Import sample_docs folder**. New uploads can be indexed at any time with **Upload multiple PDFs**.

Ingestion result rows report indexed/failed status, document ID, classification, pages, counts by content type, artifact locations, parser path and warnings. Figure crops use Nomic Vision embeddings; Gemini adds a searchable visual description when configured, otherwise the extracted caption is used.

## Performance and optional parsing

The default setup favors a quick first pass:

| Setting | Default | Purpose |
|---|---:|---|
| `CHUNK_MAX_TOKENS` | `512` | HybridChunker token ceiling for the configured Nomic tokenizer. |
| `EMBEDDING_BATCH_SIZE` | `32` | Text embedding batch size. |
| `RETRIEVAL_CANDIDATE_K` | `30` | Candidates from each Neo4j retrieval path before fusion. |
| `MAX_UPLOAD_MB` | `25` | Upload size limit. |
| `CROSS_ENCODER_ENABLED` | `false` | Load the CrossEncoder only when enabled. |
| `DOCLING_ENABLE_OCR` | `false` | Enable OCR for scanned PDFs when needed. |
| `DOCLING_ENABLE_FORMULA_ENRICHMENT` | `true` | Docling formula enrichment for the research-paper demo. |
| `DOCLING_GENERATE_PAGE_IMAGES` | `false` | Optional full-page image generation. |
| `DOCLING_EXTRACT_PICTURES` | `true` | Extract figure crops and create Nomic Vision embeddings. |

Docling table recognition is always enabled in FAST mode. Formula enrichment is enabled by default for this demo; it invokes Docling's extra formula model and should be validated on the target paper. `GEMINI_API_KEY` enables Gemini figure descriptions, while the caption-only fallback works without it. `AGENT_PLANNER_ENABLED=true` enables an LLM-backed JSON planner when the OpenAI-compatible endpoint is configured; otherwise the heuristic planner is used. PyMuPDF remains the text-only fallback. CrossEncoder only reranks evidence that already passed Neo4j ACL filtering. Models are cached in process and in `data/model_cache`.

## Evaluation and tests

Run isolated automated tests (they mock Neo4j/model/API responses and do not need paid API calls):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run the deterministic smoke/evidence evaluation against a running API after indexing the sample PDF:

```powershell
.\.venv\Scripts\python.exe evaluate_demo.py
```

Ragas answer faithfulness and retrieval-context recall are a separate, optional LLM-as-judge evaluation. Install the extra dependencies only for that step, configure the approved LLM endpoint, start the backend, and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe evaluate_ragas.py
```

Ragas evaluation sends the sample questions, generated answers, and authorized retrieved excerpts to the configured judge endpoint. Its output is saved to `outputs/ragas_evaluation.json`.

## Current limits

- Neo4j is required for live ingestion and retrieval. Automated tests use mocks and do not establish real AuraDB availability or production ACL isolation.
- The current paper/evaluation set is a small demo corpus; quality numbers are not general benchmarks.
- Formula enrichment is enabled by default for this research-paper demo; the first ingestion may download the Docling formula model. OCR remains opt-in for scanned PDFs. Figure crops are enabled by default; Gemini descriptions are optional and fall back to the source caption. Validate formula and figure extraction against the target PDF before relying on them.
- PII detection reports regex matches; it does not redact them. Prompt-injection checks are pattern-based demo guardrails.
- Ingestion runs synchronously. SQLite session and rate-limit state is intended for a single host; multi-replica deployment needs shared state.
- This demo uses local shared-password identities. Add SSO, managed secrets, audit/retention, malware scanning, OCR quality evaluation, durable ingestion workers, and monitoring before production use.


## Local developer diagnostics

Set `RAG_DEBUG_TRACE=true` in your local, uncommitted `.env` file and restart FastAPI and Streamlit. Each chat request then prints a `[RAG_DEBUG_TRACE]` JSON event to the FastAPI terminal and shows a **Developer trace · every retrieval step** expander in Streamlit. It includes the plan, modality filters, vector/BM25 candidate records, scores, graph neighbors, evidence-gate decisions, answer context, LLM errors, asset/file paths, timestamps and stage timings. The trace can contain full retrieved document text; use it only with demo or data you are allowed to inspect, and turn it off before production.

Formula enrichment is enabled by default in `app.config`; add `DOCLING_ENABLE_FORMULA_ENRICHMENT=true` to `.env` to make the setting explicit. Re-ingest the PDF after changing extraction settings. Check the ingestion response for the `formulas` count and its extraction trace before investigating retrieval.

## Modality-wise question answering and developer console

- Every query is planned into separate type-specific searches. A figure search cannot return text/formula items, and the same holds for the other modalities.
- Each modality pass can retrieve candidates, but the final answer context is capped at **three chunks total**. When multiple types were requested, the merge first reserves the best available item from each type, then fills remaining slots by ranking.
- The OpenAI request receives text, table JSON and LaTeX as text. For each selected figure, the backend reads the extracted crop and attaches it as an image input to the vision-capable OpenAI Chat Completions request. A caption alone is not treated as equivalent to an image.
- Set `RAG_DEBUG_TRACE=true` locally to view the planner, type-specific retrieval, scores, final top-three selection, file paths, actual image attachment status, answer context and per-stage timings in the Streamlit **Developer console**. The console can show document content and local file paths; use demo documents only and disable it before a normal demo.
- The default answer model is `gpt-4o-mini`. Set `LLM_API_KEY` in the local `.env`; do not commit or share it. The planner and answer model both use the configured OpenAI endpoint/model.
