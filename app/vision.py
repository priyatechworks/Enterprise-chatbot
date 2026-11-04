"""Optional Gemini-based descriptions for extracted research-paper figures."""
from pathlib import Path

from app.config import GEMINI_API_KEY, GEMINI_VISION_MODEL

_client = None


def describe_image(path: str | Path, caption: str = "", document_name: str = "", page: int = 0):
    """Return (searchable description, warning); gracefully fall back to the PDF caption."""
    global _client
    fallback = caption.strip() or f"Figure extracted from {document_name}, page {page}."
    if not GEMINI_API_KEY:
        return fallback, "GEMINI_API_KEY is not configured; figure caption used as its searchable description"
    try:
        from google import genai
        from PIL import Image

        if _client is None:
            _client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            "Describe this scientific-paper figure for accurate retrieval. Identify diagram/chart type, "
            "visible components, labels, arrows, stages, axes and relationships. Do not guess unreadable "
            "labels or infer claims not visible. "
            f"Paper: {document_name}; page: {page}; caption: {caption or 'not available'}. "
            "Return a concise but technically useful description."
        )
        with Image.open(path) as image:
            response = _client.models.generate_content(
                model=GEMINI_VISION_MODEL, contents=[prompt, image]
            )
        description = (getattr(response, "text", None) or "").strip()
        if not description:
            return fallback, "Gemini returned an empty description; caption used instead"
        if caption and caption.casefold() not in description.casefold():
            description = f"Caption: {caption}\nVisual description: {description}"
        return description, None
    except Exception as exc:
        return fallback, f"Gemini figure description failed ({type(exc).__name__}); caption used instead"
