"""Persist extracted document elements and a traceable item-level metadata manifest."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from app.config import EXTRACTED_DIR, FORMULA_DIR, METADATA_DIR, TABLE_DIR, TEXT_DIR

def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value[:100] or hashlib.sha256(str(value).encode()).hexdigest()[:16]

def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    for method in ("model_dump", "dict", "to_dict"):
        if hasattr(value, method):
            try:
                return _jsonable(getattr(value, method)())
            except Exception:
                pass
    return str(value)

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    temporary.replace(path)

def persist_extracted_records(document_id, document_name, page_count, classification, allowed_roles, records):
    """Write type-specific artifacts and attach durable provenance to each record."""
    document_key = _slug(document_id)
    for folder in (EXTRACTED_DIR, TEXT_DIR, TABLE_DIR, FORMULA_DIR, METADATA_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    text_rows, formula_rows, manifest_rows, table_paths = [], [], [], []
    counts = {"text": 0, "table": 0, "formula": 0, "figure": 0}

    for folder, pattern in (
        (TEXT_DIR, f"{document_key}_*.jsonl"),
        (TABLE_DIR, f"{document_key}_table_*.json"),
        (FORMULA_DIR, f"{document_key}_*.jsonl"),
        (METADATA_DIR, f"{document_key}_*.json*"),
    ):
        for old in folder.glob(pattern):
            old.unlink(missing_ok=True)

    for ordinal, record in enumerate(records):
        kind = str(record.get("kind", "text")).lower()
        if kind == "image":
            kind = "figure"
            record["kind"] = kind
        item_id = str(record.get("item_id") or f"{document_id}:{ordinal}")
        headings = record.get("headings") or []
        section = " > ".join(str(part) for part in headings if str(part).strip()) or "Unsectioned"
        section_key = hashlib.sha256(f"{document_id}:{section.casefold()}".encode()).hexdigest()[:24]
        page = int(record.get("page") or 1)
        record.update({
            "item_id": item_id, "document_id": document_id, "document": document_name,
            "content_type": kind, "section": section, "section_key": section_key,
            "page": page, "classification": classification, "allowed_roles": list(allowed_roles),
        })
        if kind == "text":
            counts["text"] += 1
            text_rows.append(record)
        elif kind == "table":
            counts["table"] += 1
            table_path = TABLE_DIR / f"{document_key}_table_{counts['table']:04d}.json"
            record["file_path"] = str(table_path.resolve())
            _write_json(table_path, {
                "item_id": item_id, "document_id": document_id, "document": document_name,
                "page": page, "section": section, "caption": record.get("caption", ""),
                "markdown": record.get("text", ""), "structured_data": record.get("structured_data"),
                "provenance": record.get("provenance", {}),
            })
            table_paths.append(str(table_path.resolve()))
        elif kind == "formula":
            counts["formula"] += 1
            formula_rows.append(record)
        elif kind == "figure":
            counts["figure"] += 1
            record["file_path"] = record.get("asset_path", "")

        manifest_rows.append({
            "item_id": item_id, "document_id": document_id, "document": document_name,
            "content_type": kind, "page": page, "section": section, "section_key": section_key,
            "caption": record.get("caption", ""), "file_path": record.get("file_path", ""),
            "asset_path": record.get("asset_path", ""), "docling_ref": record.get("docling_ref", ""),
            "provenance": record.get("provenance", {}), "classification": classification,
            "allowed_roles": list(allowed_roles),
        })

    text_path = TEXT_DIR / f"{document_key}_chunks.jsonl"
    formula_path = FORMULA_DIR / f"{document_key}_formulas.jsonl"
    manifest_path = METADATA_DIR / f"{document_key}_items.jsonl"
    for record in text_rows:
        record["file_path"] = str(text_path.resolve())
    for record in formula_rows:
        record["file_path"] = str(formula_path.resolve())
    _write_jsonl(text_path, text_rows)
    _write_jsonl(formula_path, formula_rows)
    for row in manifest_rows:
        if row["content_type"] == "text":
            row["file_path"] = str(text_path.resolve())
        elif row["content_type"] == "formula":
            row["file_path"] = str(formula_path.resolve())
    _write_jsonl(manifest_path, manifest_rows)

    summary = {
        "document_id": document_id, "document": document_name, "page_count": int(page_count),
        "classification": classification, "allowed_roles": list(allowed_roles), "counts": counts,
        "artifacts": {
            "text": str(text_path.resolve()), "tables": table_paths,
            "formulas": str(formula_path.resolve()), "metadata": str(manifest_path.resolve()),
        },
    }
    _write_json(METADATA_DIR / f"{document_key}_document.json", summary)
    return records, summary
