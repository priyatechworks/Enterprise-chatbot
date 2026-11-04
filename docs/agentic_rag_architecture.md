# Multimodal Agentic RAG architecture

## Scope

The seeded demo corpus is the research paper **Attention Is All You Need**. The ingestion and retrieval implementation can accept additional PDFs, but the demo questions and explanations are centered on this paper.

## End-to-end architecture

~~~mermaid
flowchart TB
  PDF[Research paper PDF] --> DOC[Docling PDF pipeline]
  DOC --> CHUNK[HybridChunker + page/heading provenance]
  DOC --> TABLE[Table structure + Markdown + structured payload]
  DOC --> FORMULA[Formula items / optional LaTeX enrichment]
  DOC --> FIGURE[Picture crop + caption + page/bbox]
  FIGURE --> VLM[Optional Gemini figure description]
  CHUNK --> META[Item metadata + extracted artifacts]
  TABLE --> META
  FORMULA --> META
  VLM --> META
  FIGURE --> VISION[Nomic Embed Vision v1.5]
  META --> TEXT[Nomic Embed Text v1.5]
  TEXT --> NEO[(Neo4j)]
  VISION --> NEO
  META --> MANIFEST[JSONL metadata manifest + type-specific files]
  NEO --> GRAPH[Document / Section / KnowledgeItem graph]
  GRAPH --> INDEX[768-dim vector index + Lucene BM25]
  USER[User question] --> PLAN[Query planner]
  PLAN --> ROUTER[Content-type router]
  ROUTER --> RETRIEVE[Text / table / figure / formula searches]
  RETRIEVE --> ACL[Neo4j role filter]
  ACL --> RRF[Hybrid RRF merge]
  RRF --> EXPAND[Authorized NEXT / same-page graph expansion]
  EXPAND --> GATE[Evidence and safety validation]
  GATE --> LLM[Grounded answer generation]
  LLM --> ANSWER[Answer + citations + figure/table/formula evidence]
~~~

## Ingestion components

- **Docling 2.145.0** converts the PDF into structured document items. HybridChunker creates heading-aware text and table chunks. Table recognition uses TableFormer FAST mode.
- **Text** is chunked and embedded with nomic-ai/nomic-embed-text-v1.5.
- **Tables** retain the chunk's Markdown/text representation and a JSON serialization of Docling's structured table payload. Both the searchable text and a table JSON file are kept.
- **Figures** are extracted as separate image crops. Each figure keeps its caption, page number, provenance/bounding box, Docling reference and asset path. nomic-ai/nomic-embed-vision-v1.5 produces the image vector. The v1.5 text and vision encoders are aligned in a shared 768-dimensional space, so a text query can retrieve a figure.
- **Figure descriptions** use Gemini through the google-genai SDK when GEMINI_API_KEY is configured. Without a key, the paper's caption is the searchable figure description. Vision embeddings and image descriptions have different jobs: the former supports similarity retrieval; the latter provides text evidence to the answer model.
- **Formulas** are captured as Docling FORMULA text items when present. Set DOCLING_ENABLE_FORMULA_ENRICHMENT=true to enable Docling's additional LaTeX extraction model. This is intentionally opt-in because it adds model execution and needs validation on the paper.
- **OCR** remains optional (DOCLING_ENABLE_OCR=true) for scanned PDFs. The supplied research paper is a digitally generated PDF, so OCR is not required for the seeded demo.
- A full Docling document JSON export is saved under data/extracted/docling/ when the API is available.

Generated data layout:

~~~text
data/
  assets/                  # extracted figure crops, protected by /assets/{asset_name}
  extracted/
    docling/               # full Docling JSON export
    text/                  # searchable text chunks as JSONL
    tables/                # one structured JSON file per table
    formulas/              # formula items as JSONL
    metadata/              # item manifest and document summary
~~~

The item manifest records an item ID, document ID/name, content type, page, section/section key, caption, file/asset path, Docling reference, provenance, classification and allowed roles. Neo4j stores the item metadata required for retrieval, while SQLite continues to store the document catalog, sessions, messages and rate-limit state.

## Neo4j graph and indexes

Neo4j remains the only vector database. There is no FAISS or second vector store.

- Document nodes represent source PDFs.
- Section nodes represent document heading groups.
- KnowledgeItem nodes represent searchable text, table, figure and formula items.
- HAS_SECTION, HAS_ITEM, HAS_CONTENT, CONTAINS_TEXT, CONTAINS_TABLE, CONTAINS_FIGURE, CONTAINS_FORMULA, NEXT and RELATED_ON_PAGE relationships preserve hierarchy, order and source-grounded co-location.
- Optional Docling Graph extraction can add KGEntity/KG_RELATION nodes when a graph template and LLM endpoint are configured.

knowledge_embedding is a 768-dimensional cosine vector index. Text/formula/table descriptions use Nomic Text vectors; figure crops use the aligned Nomic Vision vectors. knowledge_text is Neo4j Lucene full-text search (BM25). Both retrieval paths apply allowed-role filters in Cypher before node content is returned. RRF fuses the two candidate lists using 1 / (60 + rank).

## Query planning and answering

1. **Safety and identity:** FastAPI validates the user, session, rate limit and prompt-injection screen.
2. **Planning:** app.agent.plan_query() creates a structured JSON plan of up to four tasks. The configured LLM endpoint may create this plan; deterministic keyword routing is the fallback.
3. **Routing:** Each task is sent to the Neo4j retrieval function with content_types restricted to text, table, figure or formula.
4. **Retrieval:** The task runs Nomic query embedding + vector search and Lucene BM25 full-text retrieval. Neo4j applies role ACLs to both. RRF de-duplicates by item/chunk ID.
5. **Graph expansion:** Authorized NEXT, RELATED_ON_PAGE, and optional semantic KG relationships add nearby supporting items. Graph expansion remains bounded.
6. **Evidence gate:** Similarity, overlap and retrieved-content prompt-injection checks run before generation. Unsupported requests produce an abstention.
7. **Answer:** The configured OpenAI-compatible model receives text evidence with content type, caption, section, page and LaTeX metadata. It is instructed to preserve formulas, report table values carefully, explain only supplied figure descriptions and cite [filename, page N].
8. **UI:** The Streamlit source panel displays the original protected figure when available, shows table content as Markdown, and renders formula notation using LaTeX.

## Access and data handling

- Public documents are available to Admin and Employee; Admin-only documents are available only to Admin.
- ACL values are inherited by each content item and stored in Neo4j. Every vector/full-text candidate is filtered before the record is returned. The image endpoint independently verifies permission for the exact asset path.
- Re-uploading a document with the same filename reuses its document ID, replaces the old graph items and relationships, and removes stale figure assets after the graph replacement succeeds.
- Extracted text/table/formula/metadata files and figure assets are local generated data. data/extracted/ and data/assets/ are ignored by Git.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| DOCLING_EXTRACT_PICTURES | true | Extract figures and embed with Nomic Vision. |
| GEMINI_API_KEY | empty | Optional Gemini visual descriptions; captions are fallback. |
| GEMINI_VISION_MODEL | gemini-2.5-flash | Gemini model for figure description. |
| AGENT_PLANNER_ENABLED | true | Enable LLM-backed JSON planning when LLM credentials are present; otherwise heuristics. |
| DOCLING_ENABLE_FORMULA_ENRICHMENT | false | Opt in to Docling LaTeX enrichment and validate its extraction. |
| DOCLING_ENABLE_OCR | false | Enable OCR for scanned PDFs. |
| DOCLING_GENERATE_PAGE_IMAGES | false | Optional full-page images. |
| CROSS_ENCODER_ENABLED | false | Optional reranking after ACL filtering. |

## Validation

The automated tests mock Neo4j and embedding/LLM services. They cover access control, vector/BM25 routing, RRF, answer citations, abstention, bounded agentic retrieval, modality classification, artifact persistence and shared-dimension text/vision row preparation. They do not verify live Neo4j Cypher execution, real Docling parsing quality, Gemini responses or formula extraction fidelity. After pulling the branch, run pytest -q, ingest the Transformer PDF into your configured Neo4j, and inspect the generated item manifest and citations before treating the pipeline as validated.
