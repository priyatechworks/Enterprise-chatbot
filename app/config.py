from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "sample_docs"
UPLOAD_DIR = DATA_DIR / "documents"
TMP_DIR = DATA_DIR / "tmp"
ASSET_DIR = DATA_DIR / "assets"
EXTRACTED_DIR = DATA_DIR / "extracted"
TEXT_DIR = EXTRACTED_DIR / "text"
TABLE_DIR = EXTRACTED_DIR / "tables"
FORMULA_DIR = EXTRACTED_DIR / "formulas"
METADATA_DIR = EXTRACTED_DIR / "metadata"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"
DB_PATH = DATA_DIR / "rag_demo.sqlite3"
MANIFEST_PATH = DOCS_DIR / "role_access_manifest.csv"
EVALUATION_PATH = DOCS_DIR / "demo_test_questions.json"
load_dotenv(ROOT / ".env")
# Force model caches onto D: even if Windows has a user-level HF cache on C:.
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "local-rag-change-this")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_VECTOR_INDEX = "knowledge_embedding"
TEXT_MODEL = os.getenv("TEXT_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
VISION_MODEL = os.getenv("VISION_EMBED_MODEL", "nomic-ai/nomic-embed-vision-v1.5")
DOCLING_GRAPH_TEMPLATE = os.getenv("DOCLING_GRAPH_TEMPLATE", "")
DOC_GRAPH_PROVIDER = os.getenv("DOC_GRAPH_PROVIDER", "openai")
EMBEDDING_DIMENSION = 768
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
AGENT_PLANNER_ENABLED = _env_bool("AGENT_PLANNER_ENABLED", True)
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "RagDemo!2026")
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "local-demo-only-change-before-deployment")
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.30"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "30"))
USER_MAX_SESSIONS = int(os.getenv("USER_MAX_SESSIONS", "5"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", "2"))
for folder in (DOCS_DIR, ASSET_DIR, EXTRACTED_DIR, TEXT_DIR, TABLE_DIR, FORMULA_DIR, METADATA_DIR, MODEL_CACHE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
for folder in (UPLOAD_DIR, TMP_DIR):
    folder.mkdir(parents=True, exist_ok=True)

def _env_bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

CHUNK_MAX_TOKENS = max(64, int(os.getenv("CHUNK_MAX_TOKENS", "512")))
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
RETRIEVAL_CANDIDATE_K = max(1, int(os.getenv("RETRIEVAL_CANDIDATE_K", "30")))
MAX_UPLOAD_MB = max(1, int(os.getenv("MAX_UPLOAD_MB", "25")))
CROSS_ENCODER_ENABLED = _env_bool("CROSS_ENCODER_ENABLED", False)
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DOCLING_ENABLE_OCR = _env_bool("DOCLING_ENABLE_OCR", False)
DOCLING_ENABLE_FORMULA_ENRICHMENT = _env_bool("DOCLING_ENABLE_FORMULA_ENRICHMENT", False)
DOCLING_GENERATE_PAGE_IMAGES = _env_bool("DOCLING_GENERATE_PAGE_IMAGES", False)
DOCLING_EXTRACT_PICTURES = _env_bool("DOCLING_EXTRACT_PICTURES", True)
