"""PDF ingestion, hybrid retrieval, and role-scoped Neo4j graph search."""
import csv
import hashlib
import importlib
import json
import os
import re
import uuid
import time
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

from app.config import (
    ASSET_DIR, EXTRACTED_DIR, CHUNK_MAX_TOKENS, CROSS_ENCODER_ENABLED, CROSS_ENCODER_MODEL, EMBEDDING_BATCH_SIZE,
    DOCLING_ENABLE_FORMULA_ENRICHMENT, DOCLING_ENABLE_OCR,
    DOCLING_EXTRACT_PICTURES, DOCLING_GENERATE_PAGE_IMAGES, DOCLING_GRAPH_TEMPLATE,
    DOC_GRAPH_PROVIDER, EMBEDDING_DIMENSION, GRAPH_MAX_HOPS, LLM_API_KEY,
    LLM_BASE_URL, LLM_MODEL, MANIFEST_PATH, MODEL_CACHE_DIR, NEO4J_DATABASE,
    NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_THRESHOLD, TEXT_MODEL, VISION_MODEL, RAG_DEBUG_TRACE,
)
from app.storage import get_document, record_failure, upsert_document
from app.metadata import persist_extracted_records
from app.vision import describe_image
from app.agent import plan_query
from app.trace import RequestTrace

_driver = None
_neo4j_supports_search = False
_neo4j_supports_fulltext_search = False
_neo4j_version = ''
_text_model = None
_vision_processor = None
_vision_model = None
_hybrid_chunker = None
_cross_encoder = None
_docling_converter = None
_QUERY_STOPWORDS = {
    'a','an','the','and','or','is','are','was','were','what','how','many','much',
    'do','does','did','i','me','my','we','you','your','for','to','in','on','of',
    'at','by','with','from','after','before','should','can','could','would','please',
    'tell','show','give','it','this','that','information','paper','one','way',
}


def driver():
    global _driver, _neo4j_supports_search, _neo4j_supports_fulltext_search, _neo4j_version
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_pool_size=20, connection_timeout=5,
        )
        _driver.verify_connectivity()
        with _driver.session(database=NEO4J_DATABASE) as session:
            components = list(session.run('CALL dbms.components() YIELD name, versions RETURN name, versions'))
            kernel = next((row for row in components if 'kernel' in str(row['name']).lower()), components[0] if components else None)
            _neo4j_version = str(kernel['versions'][0]) if kernel and kernel['versions'] else ''
            version_match = re.match(r'^(\d{4})\.(\d+)', _neo4j_version)
            version_tuple = (int(version_match.group(1)), int(version_match.group(2))) if version_match else (0, 0)
            _neo4j_supports_search = version_tuple >= (2026, 1)
            _neo4j_supports_fulltext_search = version_tuple >= (2026, 9)
            session.run(
                "CREATE VECTOR INDEX knowledge_embedding IF NOT EXISTS FOR (n:KnowledgeItem) "
                f"ON (n.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {int(EMBEDDING_DIMENSION)}, "
                "`vector.similarity_function`: 'cosine'}}",
            )
            session.run(
                "CREATE FULLTEXT INDEX knowledge_text IF NOT EXISTS FOR (n:KnowledgeItem) "
                "ON EACH [n.text, n.document]"
            )
    return _driver


def _vector_query(session, query_vector, role, limit, candidate_limit, restricted=False, kinds=None):
    kinds = kinds or ["text", "table", "figure", "formula"]
    if _neo4j_supports_search:
        authorization = 'NOT $role IN node.allowed_roles' if restricted else '$role IN node.allowed_roles'
        similarity = ' AND score>=0.34' if restricted else ''
        return list(session.run(
            'CYPHER 25 MATCH (node:KnowledgeItem) '
            'SEARCH node IN (VECTOR INDEX knowledge_embedding FOR $embedding LIMIT $limit) SCORE AS score '
            f'WHERE {authorization} AND node.kind IN $kinds{similarity} '
            + ('RETURN score ORDER BY score DESC LIMIT 1' if restricted else 'RETURN node,score ORDER BY score DESC LIMIT $candidate_limit'),
            embedding=query_vector, limit=limit, candidate_limit=candidate_limit, role=role, kinds=kinds,
        ))
    if restricted:
        return list(session.run(
            "CALL db.index.vector.queryNodes('knowledge_embedding',$limit,$embedding) YIELD node,score "
            "WHERE NOT $role IN node.allowed_roles AND node.kind IN $kinds AND score>=0.34 "
            "RETURN score ORDER BY score DESC LIMIT 1",
            limit=limit, embedding=query_vector, role=role, kinds=kinds,
        ))
    return list(session.run(
        "CALL db.index.vector.queryNodes('knowledge_embedding',$limit,$embedding) YIELD node,score "
        "WHERE $role IN node.allowed_roles AND node.kind IN $kinds "
        "RETURN node,score ORDER BY score DESC LIMIT $candidate_limit",
        limit=limit, candidate_limit=candidate_limit, embedding=query_vector, role=role, kinds=kinds,
    ))

def _fulltext_query(session, lucene_query, role, limit, candidate_limit, restricted=False, kinds=None):
    kinds = kinds or ["text", "table", "figure", "formula"]
    if _neo4j_supports_fulltext_search:
        authorization = 'NOT $role IN node.allowed_roles' if restricted else '$role IN node.allowed_roles'
        return list(session.run(
            'CYPHER 25 MATCH (node:KnowledgeItem) '
            'SEARCH node IN (FULLTEXT INDEX knowledge_text FOR $lucene_query LIMIT $limit) SCORE AS score '
            f'WHERE {authorization} AND node.kind IN $kinds '
            + ('RETURN score ORDER BY score DESC LIMIT 1' if restricted else 'RETURN node,score ORDER BY score DESC LIMIT $candidate_limit'),
            lucene_query=lucene_query, limit=limit, candidate_limit=candidate_limit, role=role, kinds=kinds,
        ))
    if restricted:
        return list(session.run(
            "CALL db.index.fulltext.queryNodes('knowledge_text',$lucene_query,{limit:$limit}) YIELD node,score "
            "WHERE NOT $role IN node.allowed_roles AND node.kind IN $kinds "
            "RETURN score ORDER BY score DESC LIMIT 1",
            lucene_query=lucene_query, limit=limit, role=role, kinds=kinds,
        ))
    return list(session.run(
        "CALL db.index.fulltext.queryNodes('knowledge_text',$lucene_query,{limit:$limit}) YIELD node,score "
        "WHERE $role IN node.allowed_roles AND node.kind IN $kinds "
        "RETURN node,score ORDER BY score DESC LIMIT $candidate_limit",
        lucene_query=lucene_query, limit=limit, candidate_limit=candidate_limit, role=role, kinds=kinds,
    ))

def text_model():
    global _text_model
    if _text_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        _text_model = SentenceTransformer(
            TEXT_MODEL, trust_remote_code=True, cache_folder=str(MODEL_CACHE_DIR)
        )
    return _text_model


def text_embeddings(texts, query=False):
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    prefix = 'search_query: ' if query else 'search_document: '
    vectors = text_model().encode(
        [prefix + text for text in texts], batch_size=EMBEDDING_BATCH_SIZE, normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def image_embedding(path):
    global _vision_model, _vision_processor
    if _vision_model is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        torch.set_num_threads(2)
        _vision_processor = AutoImageProcessor.from_pretrained(
            VISION_MODEL, cache_dir=str(MODEL_CACHE_DIR), trust_remote_code=True
        )
        _vision_model = AutoModel.from_pretrained(
            VISION_MODEL, cache_dir=str(MODEL_CACHE_DIR), trust_remote_code=True
        ).to('cpu').eval()
    else:
        import torch
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert('RGB')
    inputs = _vision_processor(image, return_tensors='pt')
    with torch.inference_mode():
        output = _vision_model(**inputs)
        vector = torch.nn.functional.normalize(output.last_hidden_state[:, 0, :], dim=-1)
    return vector[0].cpu().numpy().astype(np.float32)


def _chunker():
    """Build the installed Docling Core 2.105 HybridChunker with the Nomic tokenizer."""
    global _hybrid_chunker
    if _hybrid_chunker is None:
        from transformers import AutoTokenizer
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_MODEL, trust_remote_code=True, cache_dir=str(MODEL_CACHE_DIR)
        )
        tokenizer_limit = getattr(tokenizer, 'model_max_length', CHUNK_MAX_TOKENS)
        encoder_limit = getattr(text_model(), 'max_seq_length', CHUNK_MAX_TOKENS)
        limits = [value for value in (tokenizer_limit, encoder_limit) if isinstance(value, int) and 0 < value < 100000]
        model_limit = min(limits) if limits else CHUNK_MAX_TOKENS
        token_limit = max(64, min(CHUNK_MAX_TOKENS, model_limit))
        _hybrid_chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=token_limit),
            repeat_table_header=True,
            merge_peers=True,
        )
    return _hybrid_chunker


def _docling_converter_for_settings():
    global _docling_converter
    if _docling_converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
        options = PdfPipelineOptions()
        options.do_ocr = DOCLING_ENABLE_OCR
        options.do_table_structure = True
        options.table_structure_options.mode = TableFormerMode.FAST
        options.do_formula_enrichment = DOCLING_ENABLE_FORMULA_ENRICHMENT
        options.generate_page_images = DOCLING_GENERATE_PAGE_IMAGES
        options.generate_picture_images = DOCLING_EXTRACT_PICTURES
        _docling_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
    return _docling_converter


def load_manifest():
    """Read optional seed-document defaults; uploads do not depend on this file."""
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open(newline='', encoding='utf-8-sig') as stream:
        return {
            row['filename']: {
                'document_id': row.get('document_id', ''),
                'classification': row.get('classification', 'Admin-only'),
                'allowed_roles': [role.strip() for role in row.get('allowed_roles', '').split('|') if role.strip()],
            }
            for row in csv.DictReader(stream)
        }


def _policy_roles(classification):
    if classification == 'Public':
        return ['Admin', 'Employee']
    if classification == 'Admin-only':
        return ['Admin']
    raise ValueError("Classification must be 'Public' or 'Admin-only'")


def _plain_payload(value):
    """Convert Docling/Pydantic objects into JSON-safe metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_payload(item) for item in value]
    for method in ("model_dump", "dict", "to_dict"):
        if hasattr(value, method):
            try:
                return _plain_payload(getattr(value, method)())
            except Exception:
                pass
    return str(value)


def _item_provenance(item):
    provenance = []
    for prov in (getattr(item, "prov", None) or []):
        bbox = getattr(prov, "bbox", None)
        provenance.append({
            "page_no": getattr(prov, "page_no", None),
            "bbox": _plain_payload(bbox),
        })
    return provenance


def _item_caption(item, document):
    try:
        caption = item.caption_text(document)
        return str(caption).strip() if caption else ""
    except Exception:
        return ""


def _item_ref(item):
    try:
        ref = item.get_ref()
        return str(getattr(ref, "cref", ref))
    except Exception:
        return ""


def _extract_docling(path, document_name, document_id=None):
    result = _docling_converter_for_settings().convert(path)
    document = result.document
    records = []
    page_headings = {}
    chunker = _chunker()

    # Text and tables are chunked using Docling's layout-aware HybridChunker.
    for chunk in chunker.chunk(document):
        chunk_items = list(getattr(getattr(chunk, "meta", None), "doc_items", []) or [])
        labels = {
            str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", ""))).lower()
            for item in chunk_items
        }
        page_numbers = [
            int(prov.page_no)
            for item in chunk_items
            for prov in (getattr(item, "prov", []) or [])
            if getattr(prov, "page_no", None) is not None
        ]
        page = page_numbers[0] if page_numbers else 1
        text = chunker.contextualize(chunk).strip()
        if not text:
            continue
        headings = list(getattr(getattr(chunk, "meta", None), "headings", []) or [])
        if headings:
            for page_number in (page_numbers or [page]):
                page_headings[int(page_number)] = headings
        table_items = [
            item for item in chunk_items
            if str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", ""))).lower() == "table"
        ]
        table_payload = None
        caption = ""
        provenance = []
        docling_ref = ""
        if table_items:
            table_item = table_items[0]
            table_payload = _plain_payload(getattr(table_item, "data", None))
            caption = _item_caption(table_item, document)
            provenance = _item_provenance(table_item)
            docling_ref = _item_ref(table_item)
        records.append({
            "kind": "table" if "table" in labels else "text",
            "page": page,
            "text": text,
            "headings": headings,
            "caption": caption,
            "structured_data": table_payload,
            "provenance": provenance,
            "docling_ref": docling_ref,
            "bbox": provenance[0].get("bbox") if provenance else None,
        })

    # Formula understanding produces TextItems labelled FORMULA. Keep them as
    # independent retrievable records in addition to their surrounding text.
    for item, _level in document.iterate_items():
        label = str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", ""))).lower()
        if label != "formula":
            continue
        formula_text = str(getattr(item, "text", "") or getattr(item, "orig", "") or "").strip()
        if not formula_text:
            continue
        provenance = _item_provenance(item)
        page = int(provenance[0].get("page_no") or 1) if provenance else 1
        records.append({
            "kind": "formula",
            "page": page,
            "text": formula_text,
            "latex": formula_text,
            "caption": "",
            "headings": page_headings.get(page, []),
            "provenance": provenance,
            "bbox": provenance[0].get("bbox") if provenance else None,
            "docling_ref": _item_ref(item),
        })

    # Figures are stored as individual crops. Nomic Vision vectors are aligned
    # with Nomic Text v1.5, so a natural-language query can find these vectors.
    if DOCLING_EXTRACT_PICTURES:
        for picture in getattr(document, "pictures", []) or []:
            image = picture.get_image(doc=document)
            if image is None:
                continue
            page_prov = _item_provenance(picture)
            page = int(page_prov[0].get("page_no") or 1) if page_prov else 1
            caption = _item_caption(picture, document)
            target = ASSET_DIR / f"{uuid.uuid4().hex}.png"
            image.save(target)
            description, description_warning = describe_image(
                target, caption=caption, document_name=document_name, page=page
            )
            records.append({
                "kind": "figure",
                "page": page,
                "text": description,
                "caption": caption,
                "asset_path": str(target.resolve()),
                "headings": page_headings.get(page, []),
                "provenance": page_prov,
                "bbox": page_prov[0].get("bbox") if page_prov else None,
                "docling_ref": _item_ref(picture),
                "description_warning": description_warning,
            })

    if DOCLING_GENERATE_PAGE_IMAGES:
        for page_number, page_item in document.pages.items():
            image_ref = getattr(page_item, "image", None)
            page_image = getattr(image_ref, "pil_image", None) if image_ref else None
            if page_image is None:
                continue
            target = ASSET_DIR / f"{uuid.uuid4().hex}.png"
            page_image.save(target)
            description, description_warning = describe_image(
                target, caption=f"Full page image from {document_name}", document_name=document_name, page=int(page_number)
            )
            records.append({
                "kind": "figure", "page": int(page_number), "text": description,
                "caption": f"Full page image from {document_name}, page {page_number}",
                "asset_path": str(target.resolve()), "headings": page_headings.get(int(page_number), []),
                "provenance": [{"page_no": int(page_number), "bbox": None}],
                "bbox": None, "docling_ref": "", "description_warning": description_warning,
            })

    # Retain Docling's full structured document export for debugging and audit.
    master_path = None
    if document_id and hasattr(document, "export_to_dict"):
        try:
            master_dir = EXTRACTED_DIR / "docling"
            master_dir.mkdir(parents=True, exist_ok=True)
            master_path = master_dir / f"{document_id}.json"
            master_path.write_text(
                json.dumps(_plain_payload(document.export_to_dict()), ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            master_path = None

    if not records:
        raise ValueError("Docling found no searchable text, tables, formulas or figures")
    return len(document.pages), records

def _fallback_text_records(path):
    """Fallback emits page-scoped chunks bounded by the same model tokenizer."""
    import fitz
    tokenizer = _chunker().tokenizer.tokenizer
    pdf = fitz.open(path)
    records = []
    try:
        max_tokens = _chunker().tokenizer.max_tokens
        overlap = min(40, max_tokens // 8)
        for page_index, page in enumerate(pdf):
            text = re.sub(r'\s+', ' ', page.get_text()).strip()
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            cursor = 0
            while cursor < len(ids):
                end = min(len(ids), cursor + max_tokens)
                part = tokenizer.decode(ids[cursor:end], skip_special_tokens=True).strip()
                if part:
                    records.append({'kind': 'text', 'page': page_index + 1, 'text': part, 'headings': []})
                if end >= len(ids):
                    break
                cursor = max(cursor + 1, end - overlap)
        return len(pdf), records
    finally:
        pdf.close()


def _prepare_graph_rows(name, document_id, classification, roles, records):
    visual_kinds = {"image", "figure"}
    text_records = [record for record in records if record["kind"] not in visual_kinds]
    vectors = iter(text_embeddings([record["text"] for record in text_records]))
    rows = []
    for ordinal, record in enumerate(records):
        kind = "figure" if record["kind"] in visual_kinds else record["kind"]
        vector = image_embedding(record["asset_path"]) if kind == "figure" else next(vectors)
        chunk_id = str(record.get("item_id") or f"{document_id}:{ordinal}")
        section = str(record.get("section") or "Unsectioned")
        section_key = str(record.get("section_key") or hashlib.sha256(
            f"{document_id}:{section.casefold()}".encode()
        ).hexdigest()[:24])
        provenance = json.dumps(record.get("provenance", []), ensure_ascii=False, default=str)
        bbox = json.dumps(record.get("bbox"), ensure_ascii=False, default=str) if record.get("bbox") is not None else ""
        rows.append({
            "id": chunk_id, "chunk_id": chunk_id, "item_id": chunk_id,
            "document": name, "document_id": document_id,
            "classification": classification, "allowed_roles": roles,
            "page": int(record["page"]), "kind": kind, "content_type": kind,
            "text": str(record.get("text", "")), "headings": record.get("headings", []),
            "asset_path": record.get("asset_path", ""),
            "file_path": record.get("file_path", record.get("asset_path", "")),
            "caption": str(record.get("caption") or ""),
            "section": section, "section_key": section_key,
            "section_id": f"{document_id}:{section_key}",
            "docling_ref": str(record.get("docling_ref") or ""),
            "provenance_json": provenance, "bbox_json": bbox,
            "table_json": json.dumps(record.get("structured_data"), ensure_ascii=False, default=str)
                if record.get("structured_data") is not None else "",
            "latex": str(record.get("latex") or ""),
            "description_warning": str(record.get("description_warning") or ""),
            "ordinal": ordinal, "embedding": vector.tolist() if hasattr(vector, "tolist") else list(vector),
        })
    return rows


def _replace_graph_document(name, document_id, classification, roles, page_count, rows):
    database = driver()
    old_assets = []

    def replace(tx):
        old_assets.extend([
            record["asset_path"] for record in tx.run(
                "MATCH (k:KnowledgeItem) WHERE k.document_id=$document_id OR k.document=$name "
                "RETURN k.asset_path AS asset_path", document_id=document_id, name=name,
            ) if record["asset_path"]
        ])
        tx.run(
            "MATCH (d:Document) WHERE d.document_id=$document_id OR d.name=$name DETACH DELETE d",
            document_id=document_id, name=name,
        )
        tx.run(
            "MATCH (k:KnowledgeItem) WHERE k.document_id=$document_id OR k.document=$name DETACH DELETE k",
            document_id=document_id, name=name,
        )
        tx.run(
            "MATCH (s:Section) WHERE s.document_id=$document_id DETACH DELETE s",
            document_id=document_id,
        )
        tx.run(
            "MATCH (e:KGEntity) WHERE e.document_id=$document_id OR e.document=$name DETACH DELETE e",
            document_id=document_id, name=name,
        )
        tx.run(
            "CREATE (d:Document {document_id:$document_id,name:$name,classification:$classification,"
            "allowed_roles:$roles,page_count:$page_count,updated_at:datetime()})",
            document_id=document_id, name=name, classification=classification,
            roles=roles, page_count=page_count,
        )
        if rows:
            tx.run(
                "MATCH (d:Document {document_id:$document_id}) UNWIND $rows AS row "
                "MERGE (s:Section {id:row.section_id}) "
                "SET s.document_id=$document_id,s.title=row.section,s.section_key=row.section_key "
                "MERGE (d)-[:HAS_SECTION]->(s) "
                "CREATE (k:KnowledgeItem {id:row.id,chunk_id:row.chunk_id,item_id:row.item_id,"
                "document:row.document,document_id:row.document_id,classification:row.classification,"
                "allowed_roles:row.allowed_roles,page:row.page,kind:row.kind,content_type:row.content_type,"
                "text:row.text,headings:row.headings,asset_path:row.asset_path,file_path:row.file_path,"
                "caption:row.caption,section:row.section,section_key:row.section_key,section_id:row.section_id,"
                "docling_ref:row.docling_ref,provenance_json:row.provenance_json,bbox_json:row.bbox_json,"
                "table_json:row.table_json,latex:row.latex,description_warning:row.description_warning,"
                "ordinal:row.ordinal,embedding:row.embedding}) "
                "CREATE (d)-[:HAS_ITEM]->(k) MERGE (s)-[:HAS_CONTENT]->(k)",
                document_id=document_id, rows=rows,
            )
            tx.run(
                "MATCH (d:Document {document_id:$document_id})-[:HAS_ITEM]->(a:KnowledgeItem), "
                "(d)-[:HAS_ITEM]->(b:KnowledgeItem) WHERE b.ordinal=a.ordinal+1 "
                "MERGE (a)-[:NEXT]->(b)", document_id=document_id,
            )
            # Explicit type edges make the section's multimodal contents discoverable.
            for kind, relationship in (
                ("text", "CONTAINS_TEXT"), ("table", "CONTAINS_TABLE"),
                ("figure", "CONTAINS_FIGURE"), ("formula", "CONTAINS_FORMULA"),
            ):
                tx.run(
                    "MATCH (s:Section)-[:HAS_CONTENT]->(k:KnowledgeItem) "
                    "WHERE k.document_id=$document_id AND k.kind=$kind "
                    f"MERGE (s)-[:{relationship}]->(k)",
                    document_id=document_id, kind=kind,
                )
            # Co-location is a source-grounded relation, not an LLM-invented semantic edge.
            tx.run(
                "MATCH (a:KnowledgeItem {document_id:$document_id}),(b:KnowledgeItem {document_id:$document_id}) "
                "WHERE a.page=b.page AND a.ordinal<b.ordinal AND a.kind<>b.kind "
                "MERGE (a)-[:RELATED_ON_PAGE]->(b)", document_id=document_id,
            )

    with database.session(database=NEO4J_DATABASE) as session:
        session.execute_write(replace)
    for asset in old_assets:
        candidate = Path(asset).resolve()
        if candidate.is_relative_to(ASSET_DIR.resolve()) and candidate.is_file():
            candidate.unlink(missing_ok=True)
    return len(rows)

def _extract_semantic_graph(path):
    if not DOCLING_GRAPH_TEMPLATE:
        return None
    if not (LLM_API_KEY and LLM_BASE_URL and LLM_MODEL):
        raise ValueError('DOCLING_GRAPH_TEMPLATE requires LLM_BASE_URL, LLM_API_KEY and LLM_MODEL')
    module_name, separator, class_name = DOCLING_GRAPH_TEMPLATE.partition(':')
    if not separator:
        raise ValueError('Set DOCLING_GRAPH_TEMPLATE as python.module:TemplateClass')
    template = getattr(importlib.import_module(module_name), class_name)
    os.environ['OPENAI_API_KEY'] = LLM_API_KEY
    os.environ['OPENAI_API_BASE'] = LLM_BASE_URL
    from docling_graph import run_pipeline
    return run_pipeline({
        'source': str(path), 'template': template, 'backend': 'llm', 'inference': 'remote',
        'processing_mode': 'many-to-one', 'extraction_contract': 'auto',
        'provider_override': DOC_GRAPH_PROVIDER, 'model_override': LLM_MODEL,
        'structured_output': True, 'use_chunking': True,
    }).knowledge_graph


def _store_semantic_graph(name, document_id, roles, graph):
    if graph is None:
        return {'entities': 0, 'relations': 0}
    key = hashlib.sha256(document_id.encode()).hexdigest()[:12]
    entities, relations, ids = [], [], {}
    for node_key, data in graph.nodes(data=True):
        payload = dict(data) if isinstance(data, dict) else {'value': str(data)}
        provenance = payload.get('__provenance__', {})
        page = int(payload.get('page') or payload.get('page_no') or provenance.get('page') or provenance.get('page_no') or 0) if isinstance(provenance, dict) else 0
        entity_id = f'{key}:{hashlib.sha256(str(node_key).encode()).hexdigest()[:20]}'
        ids[node_key] = entity_id
        entities.append({
            'id': entity_id, 'name': str(payload.get('name') or payload.get('title') or node_key)[:1000],
            'kind': str(payload.get('__type__') or payload.get('type') or 'Entity')[:100],
            'page': page, 'provenance': json.dumps(payload, default=str)[:4000],
        })
    for index, (source, target, data) in enumerate(graph.edges(data=True)):
        payload = dict(data) if isinstance(data, dict) else {'value': str(data)}
        relations.append({
            'id': f'{key}:rel:{index}', 'source': ids[source], 'target': ids[target],
            'kind': str(payload.get('type') or payload.get('label') or payload.get('relation') or 'RELATED_TO')[:100],
            'provenance': json.dumps(payload, default=str)[:4000],
        })
    with driver().session(database=NEO4J_DATABASE) as session:
        if entities:
            session.run(
                "UNWIND $rows AS row MATCH (d:Document {document_id:$document_id}) "
                "MERGE (e:KGEntity {id:row.id}) SET e.name=row.name,e.kind=row.kind,e.document=$name," 
                "e.document_id=$document_id,e.page=row.page,e.allowed_roles=$roles,e.provenance=row.provenance "
                "MERGE (d)-[:HAS_ENTITY]->(e)", rows=entities, name=name,
                document_id=document_id, roles=roles,
            )
        if relations:
            session.run(
                "UNWIND $rows AS row MATCH (a:KGEntity {id:row.source}),(b:KGEntity {id:row.target}) "
                "MERGE (a)-[r:KG_RELATION {id:row.id}]->(b) SET r.kind=row.kind,r.document=$name," 
                "r.document_id=$document_id,r.allowed_roles=$roles,r.provenance=row.provenance",
                rows=relations, name=name, document_id=document_id, roles=roles,
            )
        session.run(
            "MATCH (k:KnowledgeItem {document_id:$document_id}),(e:KGEntity {document_id:$document_id}) "
            "WHERE e.page>0 AND k.page=e.page MERGE (k)-[:MENTIONS]->(e)", document_id=document_id,
        )
    return {'entities': len(entities), 'relations': len(relations)}


def ingest_file(path, classification='Admin-only', document_name=None, document_id=None):
    path = Path(path)
    name = document_name or path.name
    ingestion_trace = RequestTrace(enabled=True) if RAG_DEBUG_TRACE else None
    ingestion_started = time.perf_counter()
    if ingestion_trace:
        ingestion_trace.add(
            "ingestion_started", "PDF ingestion endpoint",
            details={"filename": name, "classification": classification,
                     "formula_enrichment_enabled": DOCLING_ENABLE_FORMULA_ENRICHMENT,
                     "figure_extraction_enabled": DOCLING_EXTRACT_PICTURES,
                     "ocr_enabled": DOCLING_ENABLE_OCR},
        )
    try:
        if path.suffix.lower() != '.pdf' or not name.lower().endswith('.pdf'):
            raise ValueError('Only PDF files supported')
        roles = _policy_roles(classification)
        existing = get_document(name)
        document_id = (existing.get('document_id') if existing else None) or document_id or str(uuid.uuid4())
        parse_started = time.perf_counter()
        try:
            page_count, records = _extract_docling(path, name, document_id)
            parse_mode, parse_warning = 'docling-hybrid-chunker', None
            if ingestion_trace:
                counts = {kind: sum(record.get("kind") == kind for record in records)
                          for kind in ("text", "table", "formula", "figure")}
                formula_items = [
                    {"page": record.get("page"), "text": record.get("text", ""),
                     "latex": record.get("latex", ""), "docling_ref": record.get("docling_ref", "")}
                    for record in records if record.get("kind") == "formula"
                ]
                ingestion_trace.add(
                    "docling_extraction", "Docling PDF converter + HybridChunker",
                    started=parse_started,
                    status="complete" if counts["formula"] else "warning",
                    details={"pages": page_count, "record_counts": counts,
                             "formula_enrichment_enabled": DOCLING_ENABLE_FORMULA_ENRICHMENT,
                             "figure_extraction_enabled": DOCLING_EXTRACT_PICTURES,
                             "ocr_enabled": DOCLING_ENABLE_OCR,
                             "warning": None if counts["formula"] else
                             "Docling returned no items labelled formula; inspect the formula model/configuration and extracted Docling JSON."},
                    data={"records": [
                        {"content_type": record.get("kind"), "page": record.get("page"),
                         "section": record.get("section") or " > ".join(record.get("headings", [])),
                         "text": record.get("text", ""), "latex": record.get("latex", ""),
                         "caption": record.get("caption", ""),
                         "asset_path": record.get("asset_path", ""),
                         "docling_ref": record.get("docling_ref", ""),
                         "structured_data": record.get("structured_data")}
                        for record in records
                    ], "formula_items": formula_items},
                )
        except Exception as parse_error:
            if ingestion_trace:
                ingestion_trace.add(
                    "docling_extraction", "Docling PDF converter + HybridChunker",
                    started=parse_started, status="failed",
                    details={"exception_type": type(parse_error).__name__,
                             "exception_message": str(parse_error),
                             "fallback": "PyMuPDF selectable-text extraction only"},
                )
            page_count, records = _fallback_text_records(path)
            parse_mode, parse_warning = 'pymupdf-tokenizer-fallback', type(parse_error).__name__
            if ingestion_trace:
                ingestion_trace.add(
                    "pymupdf_fallback", "PyMuPDF page text + tokenizer chunking",
                    details={"record_count": len(records),
                             "content_types": sorted({record.get("kind", "text") for record in records}),
                             "limitation": "Fallback does not extract figures, structured tables or formulas"},
                    data={"records": records},
                )
            if not records:
                raise ValueError(f'Docling failed ({parse_warning}) and PyMuPDF found no selectable text')

        artifact_started = time.perf_counter()
        records, artifact_summary = persist_extracted_records(
            document_id, name, page_count, classification, roles, records
        )
        if ingestion_trace:
            ingestion_trace.add(
                "local_artifact_persistence", "app.metadata.persist_extracted_records",
                started=artifact_started,
                details={"record_counts": artifact_summary.get("counts", {}),
                         "document_id": document_id},
                data={"artifact_paths": artifact_summary.get("artifacts", {}),
                      "metadata_summary": artifact_summary,
                      "items": [
                          {"item_id": record.get("item_id"), "content_type": record.get("kind"),
                           "page": record.get("page"), "file_path": record.get("file_path", ""),
                           "asset_path": record.get("asset_path", ""),
                           "latex": record.get("latex", ""), "text": record.get("text", "")}
                          for record in records
                      ]},
            )

        semantic_graph = _extract_semantic_graph(path)
        embedding_started = time.perf_counter()
        rows = _prepare_graph_rows(name, document_id, classification, roles, records)
        if ingestion_trace:
            ingestion_trace.add(
                "embedding_generation", "Nomic Text and Nomic Vision embedding models",
                started=embedding_started,
                details={"indexed_items": len(rows),
                         "content_counts": {kind: sum(row["kind"] == kind for row in rows)
                                            for kind in ("text", "table", "formula", "figure")},
                         "text_model": TEXT_MODEL, "vision_model": VISION_MODEL,
                         "embedding_dimension": EMBEDDING_DIMENSION},
                data={"items": [
                    {"item_id": row["id"], "content_type": row["kind"], "page": row["page"],
                     "embedding_dimension": len(row["embedding"]),
                     "embedding_model": VISION_MODEL if row["kind"] == "figure" else TEXT_MODEL,
                     "asset_path": row.get("asset_path", ""), "file_path": row.get("file_path", "")}
                    for row in rows
                ]},
            )
        graph_started = time.perf_counter()
        count = _replace_graph_document(name, document_id, classification, roles, page_count, rows)
        if ingestion_trace:
            ingestion_trace.add(
                "neo4j_indexing", "Neo4j KnowledgeItem / Section graph + vector and full-text indexes",
                started=graph_started,
                details={"knowledge_items_written": count, "document_id": document_id,
                         "database": NEO4J_DATABASE,
                         "vector_index": "knowledge_embedding", "fulltext_index": "knowledge_text"},
            )
        semantic_warning = None
        try:
            semantic_counts = _store_semantic_graph(name, document_id, roles, semantic_graph)
        except Exception as exc:
            semantic_counts = {'entities': 0, 'relations': 0}
            semantic_warning = f'Optional semantic graph storage failed ({type(exc).__name__}); document text remains indexed'
        registry_started = time.perf_counter()
        upsert_document(name, page_count, count, roles, document_id, classification)
        if ingestion_trace:
            ingestion_trace.add(
                "sqlite_document_registry", "app.storage.upsert_document",
                started=registry_started,
                details={"document": name, "document_id": document_id, "classification": classification},
            )
        result = {
            'name': name, 'document_id': document_id, 'classification': classification,
            'allowed_roles': roles, 'pages': page_count, 'chunks': count,
            'tables': sum(row['kind'] == 'table' for row in rows),
            'figures': sum(row['kind'] == 'figure' for row in rows),
            'formulas': sum(row['kind'] == 'formula' for row in rows),
            'images': sum(row['kind'] == 'figure' for row in rows),
            'entities': semantic_counts['entities'], 'relations': semantic_counts['relations'],
            'artifacts': artifact_summary.get('artifacts', {}),
            'content_counts': artifact_summary.get('counts', {}),
            'formula_enrichment_enabled': DOCLING_ENABLE_FORMULA_ENRICHMENT,
            'parser': parse_mode, 'status': 'indexed',
        }
        description_warnings = [
            row.get('description_warning') for row in rows if row.get('description_warning')
        ]
        if description_warnings:
            result['figure_description_warning'] = '; '.join(dict.fromkeys(description_warnings))
        if parse_warning:
            result['warning'] = f'Docling failed; indexed selectable text via PyMuPDF ({parse_warning})'
        if semantic_warning:
            result['warning'] = '; '.join(filter(None, [result.get('warning'), semantic_warning]))
        if ingestion_trace:
            ingestion_trace.add(
                "ingestion_completed", "PDF ingestion pipeline",
                started=ingestion_started,
                details={"status": result["status"], "content_counts": result["content_counts"],
                         "artifact_paths": result["artifacts"]},
            )
            result["debug_trace"] = ingestion_trace.events
        return result
    except Exception as exc:
        if ingestion_trace:
            ingestion_trace.add(
                "ingestion_failed", "PDF ingestion pipeline", status="failed",
                started=ingestion_started,
                details={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
        record_failure(name, str(exc))
        return {'name': name, 'status': 'failed', 'message': str(exc),
                'debug_trace': ingestion_trace.events if ingestion_trace else []}


def ingest_many(paths, classification='Admin-only'):
    return [ingest_file(path, classification=classification) for path in paths]


def _lucene_query(query):
    terms = [term for term in re.findall(r'[a-z0-9]+', query.lower()) if len(term) > 1 and term not in _QUERY_STOPWORDS]
    return ' OR '.join(dict.fromkeys(terms))


def _cross_encoder_model():
    global _cross_encoder
    if _cross_encoder is None and CROSS_ENCODER_ENABLED:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device='cpu')
    return _cross_encoder


def _rerank(query, candidates):
    model = _cross_encoder_model()
    if not model or not candidates:
        return candidates
    scores = model.predict([(query, item['text']) for item in candidates], show_progress_bar=False)
    for item, score in zip(candidates, scores):
        item['rerank_score'] = float(score)
    return sorted(candidates, key=lambda item: item['rerank_score'], reverse=True)


def reciprocal_rank_fusion(vector_rows, keyword_rows, rrf_constant=60):
    """Fuse ranked Neo4j vector and Lucene BM25 results using 1/(k + rank)."""
    fused = {}
    for source, rows in (('vector', vector_rows), ('bm25', keyword_rows)):
        for rank, result in enumerate(rows, start=1):
            node = result['node']
            chunk_id = node.get('chunk_id') or node.get('id')
            item = fused.setdefault(chunk_id, {
                'id': chunk_id, 'chunk_id': chunk_id,
                'document_id': node.get('document_id', ''),
                'document': node['document'], 'page': int(node['page']),
                'kind': node['kind'], 'content_type': node.get('content_type', node['kind']),
                'text': node['text'], 'asset_path': node.get('asset_path', ''),
                'file_path': node.get('file_path', node.get('asset_path', '')),
                'caption': node.get('caption', ''), 'section': node.get('section', ''),
                'section_key': node.get('section_key', ''), 'latex': node.get('latex', ''),
                'docling_ref': node.get('docling_ref', ''),
                'bbox_json': node.get('bbox_json', ''),
                'provenance_json': node.get('provenance_json', ''),
                'table_json': node.get('table_json', ''),
                'allowed_roles': list(node.get('allowed_roles', [])), 'rrf_score': 0.0,
            })
            item[f'{source}_rank'] = rank
            item[f'{source}_score'] = float(result['score'])
            item['rrf_score'] += 1.0 / (rrf_constant + rank)
    for item in fused.values():
        vector_score = item.get('vector_score', 0.0)
        bm25_score = item.get('bm25_score', 0.0)
        bm25_confidence = bm25_score / (bm25_score + 1.5) if bm25_score > 0 else 0.0
        item['score'] = max(vector_score, bm25_confidence)
    return sorted(fused.values(), key=lambda item: item['rrf_score'], reverse=True)


def _debug_item(row, rank=None, source=None):
    """Return authorized evidence fields useful for understanding a retrieval result."""
    node = row.get("node", row)
    score = row.get("score")
    return {
        "rank": rank,
        "retrieval_source": source,
        "item_id": node.get("item_id") or node.get("id") or node.get("chunk_id"),
        "document": node.get("document", ""),
        "page": node.get("page"),
        "content_type": node.get("content_type", node.get("kind", "text")),
        "retrieval_score": round(float(score), 6) if score is not None else None,
        "vector_rank": row.get("vector_rank"),
        "vector_score": row.get("vector_score"),
        "bm25_rank": row.get("bm25_rank"),
        "bm25_score": row.get("bm25_score"),
        "rrf_score": row.get("rrf_score"),
        "final_score": row.get("score"),
        "section": node.get("section", ""),
        "caption": node.get("caption", ""),
        "text": str(node.get("text", "")),
        "latex": node.get("latex", ""),
        "asset_path": node.get("asset_path", ""),
        "file_path": node.get("file_path", node.get("asset_path", "")),
        "table_json": node.get("table_json", ""),
        "docling_ref": node.get("docling_ref", ""),
        "graph_expanded": bool(node.get("graph_expanded", False)),
        "retrieved_by": node.get("retrieved_by", []),
    }


def search(query, role, top_k=3, history=None, max_hops=None, content_types=None, debug_trace=None):
    allowed_types = list(dict.fromkeys(
        kind for kind in (content_types or ["text", "table", "figure", "formula"])
        if kind in {"text", "table", "figure", "formula"}
    )) or ["text"]
    database = driver()
    embedding_started = time.perf_counter()
    query_vector = np.asarray(text_embeddings([query], query=True)[0], dtype=np.float32).tolist()
    if debug_trace:
        debug_trace.add(
            "query_embedding", "Nomic Text embedding model", started=embedding_started,
            details={"query": query, "model": TEXT_MODEL, "prefix": "search_query:", "dimension": len(query_vector)},
            data={"query_vector_dimension": len(query_vector)},
        )
    candidate_limit = max(RETRIEVAL_CANDIDATE_K, top_k)
    index_limit = candidate_limit * 5  # over-fetch to account for inaccessible hits before ACL filtering
    keyword_query = _lucene_query(query)
    if debug_trace:
        debug_trace.add(
            "retrieval_parameters", "Neo4j vector + Lucene full-text search",
            details={"role": role, "requested_content_types": allowed_types, "candidate_limit": candidate_limit,
                     "index_limit": index_limit, "lucene_query": keyword_query, "top_k": top_k},
        )
    with database.session(database=NEO4J_DATABASE) as session:
        vector_started = time.perf_counter()
        vector_rows = _vector_query(session, query_vector, role, index_limit, candidate_limit, kinds=allowed_types)
        if debug_trace:
            debug_trace.add(
                "vector_search", "Neo4j vector index: knowledge_embedding",
                started=vector_started, details={"authorized_candidates": len(vector_rows), "role_filter_applied_before_return": True},
                data={"items": [_debug_item(row, i + 1, "vector") for i, row in enumerate(vector_rows)]},
            )
        keyword_rows = []
        restricted_vector = []
        restricted_keyword = []
        if keyword_query:
            bm25_started = time.perf_counter()
            keyword_rows = _fulltext_query(session, keyword_query, role, index_limit, candidate_limit, kinds=allowed_types)
            if debug_trace:
                debug_trace.add(
                    "bm25_search", "Neo4j Lucene full-text index: knowledge_text",
                    started=bm25_started, details={"authorized_candidates": len(keyword_rows), "lucene_query": keyword_query,
                                                   "role_filter_applied_before_return": True},
                    data={"items": [_debug_item(row, i + 1, "bm25") for i, row in enumerate(keyword_rows)]},
                )
            # These denial probes return scores only; no restricted document text or name escapes Neo4j.
            restricted_keyword = _fulltext_query(session, keyword_query, role, index_limit, candidate_limit, restricted=True, kinds=allowed_types)
        restricted_vector = _vector_query(session, query_vector, role, index_limit, candidate_limit, restricted=True, kinds=allowed_types)

        fusion_started = time.perf_counter()
        ranked = reciprocal_rank_fusion(vector_rows, keyword_rows)
        if debug_trace:
            debug_trace.add(
                "hybrid_fusion", "Reciprocal Rank Fusion (RRF)",
                started=fusion_started, details={"unique_candidates_after_deduplication": len(ranked), "rrf_constant": 60},
                data={"items": [_debug_item(item, i + 1, "vector+bm25 fusion") for i, item in enumerate(ranked)]},
            )
        fused = {item['id']: item for item in ranked}
        seeds = [item['id'] for item in ranked[:min(top_k, 5)]]
        expanded = []
        hops = min(max_hops if max_hops is not None else GRAPH_MAX_HOPS, 3)
        graph_started = time.perf_counter()
        if seeds and hops:
            expanded += list(session.run(
                "MATCH (seed:KnowledgeItem) WHERE seed.id IN $ids AND $role IN seed.allowed_roles "
                "MATCH p=(seed)-[:NEXT|RELATED_ON_PAGE*1..2]-(neighbor:KnowledgeItem) "
                "WHERE length(p)<=$hops AND $role IN neighbor.allowed_roles AND neighbor.kind IN $kinds "
                "RETURN DISTINCT neighbor ORDER BY neighbor.document,neighbor.page LIMIT 12",
                ids=seeds, role=role, hops=hops, kinds=allowed_types,
            ))
            if DOCLING_GRAPH_TEMPLATE:
                expanded += list(session.run(
                    "MATCH (seed:KnowledgeItem)-[:MENTIONS]->(anchor:KGEntity) "
                    "WHERE seed.id IN $ids AND $role IN seed.allowed_roles AND $role IN anchor.allowed_roles "
                    "MATCH p=(anchor)-[:KG_RELATION*1..2]-(related:KGEntity) "
                    "WHERE $role IN related.allowed_roles AND all(r IN relationships(p) WHERE $role IN r.allowed_roles) "
                    "MATCH (neighbor:KnowledgeItem)-[:MENTIONS]->(related) "
                    "WHERE $role IN neighbor.allowed_roles AND neighbor.kind IN $kinds "
                    "RETURN DISTINCT neighbor ORDER BY neighbor.document,neighbor.page LIMIT 12",
                    ids=seeds, role=role, kinds=allowed_types,
                ))
        if debug_trace:
            debug_trace.add(
                "graph_expansion", "Neo4j NEXT / RELATED_ON_PAGE / optional KG traversal",
                started=graph_started, details={"seed_items": seeds, "graph_hop_limit": hops,
                                                "new_neighbor_records": len(expanded),
                                                "only_authorized_neighbors_returned": True},
                data={"items": [_debug_item(item["neighbor"], i + 1, "graph expansion")
                                for i, item in enumerate(expanded)]},
            )
        seen = set(fused)
        for result in expanded:
            node = result['neighbor']
            chunk_id = node.get('chunk_id') or node.get('id')
            if chunk_id not in seen:
                fused[chunk_id] = {
                    'id': chunk_id, 'chunk_id': chunk_id,
                    'document_id': node.get('document_id', ''), 'document': node['document'],
                    'page': int(node['page']), 'kind': node['kind'],
                    'content_type': node.get('content_type', node['kind']), 'text': node['text'],
                    'asset_path': node.get('asset_path', ''), 'file_path': node.get('file_path', node.get('asset_path', '')),
                    'caption': node.get('caption', ''), 'section': node.get('section', ''),
                    'section_key': node.get('section_key', ''), 'latex': node.get('latex', ''),
                    'docling_ref': node.get('docling_ref', ''), 'bbox_json': node.get('bbox_json', ''),
                    'provenance_json': node.get('provenance_json', ''), 'table_json': node.get('table_json', ''),
                    'allowed_roles': list(node.get('allowed_roles', [])),
                    'score': 0.31, 'rrf_score': 0.0, 'graph_expanded': True,
                }
                seen.add(chunk_id)
        candidates = sorted(fused.values(), key=lambda item: item['rrf_score'], reverse=True)
        rerank_started = time.perf_counter()
        # ACL was applied by Neo4j before text was returned; only these allowed rows are reranked.
        candidates = _rerank(query, candidates)
        if debug_trace:
            debug_trace.add(
                "cross_encoder_rerank", "Optional Sentence Transformers CrossEncoder",
                started=rerank_started, details={"enabled": CROSS_ENCODER_ENABLED,
                                                  "candidate_count": len(candidates),
                                                  "model": CROSS_ENCODER_MODEL if CROSS_ENCODER_ENABLED else None},
                data={"items": [_debug_item(item, i + 1, "after rerank") for i, item in enumerate(candidates)]},
            )
        denied = bool(restricted_vector or restricted_keyword)
        final_items = candidates[:top_k]
        if debug_trace:
            debug_trace.add(
                "search_task_final", "Content-type-filtered retrieval result",
                details={"returned": len(final_items), "top_k": top_k, "restricted_match": denied,
                         "content_types": allowed_types},
                data={"items": [_debug_item(item, i + 1, "final search result") for i, item in enumerate(final_items)]},
            )
        return final_items, denied


def agentic_search(query, role, top_k=3, history=None, debug_trace=None):
    """Run separate searches per modality, then select a balanced global top-k."""
    top_k = min(max(1, int(top_k)), 3)
    planning_started = time.perf_counter()
    plan = plan_query(query)
    tasks = plan.get("tasks", [])[:4]
    requested_types = list(dict.fromkeys(
        kind for kind in plan.get("requested_types", [])
        if kind in {"text", "table", "figure", "formula"}
    ))
    if not requested_types:
        requested_types = list(dict.fromkeys(
            kind for task in tasks for kind in task.get("content_types", [])
            if kind in {"text", "table", "figure", "formula"}
        ))
    if debug_trace:
        debug_trace.add(
            "query_plan", f"Agent query planner ({plan.get('planner', 'unknown')})",
            started=planning_started,
            details={"original_question": query, "task_count": len(tasks),
                     "requested_types": requested_types,
                     "planner_reason": plan.get("planner_reason", "")},
            data={"plan": plan},
        )
    merged = {}
    denied = False
    task_trace = []
    for task_index, task in enumerate(tasks, start=1):
        subquery = task.get("query") or query
        kinds = task.get("content_types") or ["text"]
        # Enforce a single content type per retrieval pass, even for a stale/custom planner.
        for kind in kinds:
            if kind not in {"text", "table", "figure", "formula"}:
                continue
            task_started = time.perf_counter()
            if debug_trace:
                debug_trace.add(
                    "retrieval_task_started", f"Modality retrieval {task_index}/{len(tasks)}",
                    details={"subquery": subquery, "content_type": kind,
                             "purpose": f"Search only {kind} knowledge items"},
                )
                hits, blocked = search(
                    subquery, role, top_k, history=history,
                    max_hops=min(GRAPH_MAX_HOPS, 2), content_types=[kind],
                    debug_trace=debug_trace,
                )
            else:
                hits, blocked = search(
                    subquery, role, top_k, history=history,
                    max_hops=min(GRAPH_MAX_HOPS, 2), content_types=[kind],
                )
            denied = denied or blocked
            reason = {"query": subquery, "content_type": kind,
                      "retrieval_method": "vector + BM25 + RRF + authorized graph expansion"}
            task_trace.append({
                "query": subquery, "content_types": [kind],
                "hits": len(hits), "restricted_match": bool(blocked),
            })
            for original_hit in hits:
                hit = dict(original_hit)
                key = hit["id"]
                existing = merged.get(key)
                if existing:
                    existing_reasons = list(existing.get("retrieved_by", [])) + [reason]
                    if hit.get("rrf_score", 0) > existing.get("rrf_score", 0):
                        hit["retrieved_by"] = existing_reasons
                        merged[key] = hit
                    else:
                        existing["retrieved_by"] = existing_reasons
                else:
                    hit["retrieved_by"] = list(hit.get("retrieved_by", [])) + [reason]
                    merged[key] = hit
            if debug_trace:
                debug_trace.add(
                    "retrieval_task_completed", f"Modality retrieval {task_index}/{len(tasks)} · {kind}",
                    started=task_started,
                    details={"authorized_hits": len(hits), "restricted_match": bool(blocked),
                             "content_type": kind},
                    data={"items": [_debug_item(hit, i + 1, f"{kind} modality result")
                                    for i, hit in enumerate(hits)]},
                )

    ranked = sorted(merged.values(), key=lambda item: item.get("rrf_score", 0), reverse=True)
    # Reserve one slot for the best result of each requested type where available,
    # then fill remaining slots from the overall ranking. The final context is <= top_k.
    best_by_type = {}
    for item in ranked:
        kind = item.get("content_type", item.get("kind", "text"))
        if kind in requested_types and kind not in best_by_type:
            best_by_type[kind] = item
    coverage = sorted(
        best_by_type.values(), key=lambda item: item.get("rrf_score", 0), reverse=True
    )[:top_k]
    selected_ids = {item["id"] for item in coverage}
    selected = list(coverage)
    for item in ranked:
        if len(selected) >= top_k:
            break
        if item["id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["id"])
    result = sorted(selected, key=lambda item: item.get("rrf_score", 0), reverse=True)
    selected_types = sorted({item.get("content_type", item.get("kind", "text")) for item in result})
    missing_types = [kind for kind in requested_types if kind not in selected_types]
    if debug_trace:
        debug_trace.add(
            "agent_merge_final", "Modality-balanced merge and top-three selection",
            details={"unique_items_before_top_k": len(merged), "returned": len(result),
                     "top_k": top_k, "requested_types": requested_types,
                     "selected_types": selected_types, "types_without_evidence": missing_types,
                     "restricted_match": denied},
            data={"items": [_debug_item(item, i + 1, "final selected evidence") for i, item in enumerate(result)]},
        )
    return result, denied, {
        "rounds": len(task_trace), "graph_hops": min(GRAPH_MAX_HOPS, 2),
        "planner": plan.get("planner", "heuristic"),
        "planner_reason": plan.get("planner_reason", ""),
        "requested_types": requested_types,
        "selected_types": selected_types,
        "missing_types": missing_types,
        "tasks": task_trace,
    }

def has_evidence_overlap(query, passages):
    terms = {term for term in re.findall(r'[a-z0-9]+', query.lower()) if len(term) > 2 and term not in _QUERY_STOPWORDS}
    if not terms:
        return bool(passages and passages[0].get('score', 0) >= .72)
    evidence_terms = set().union(*(set(re.findall(r'[a-z0-9]+', passage['text'].lower())) for passage in passages)) if passages else set()
    overlap = terms & evidence_terms
    if len(overlap) >= min(2, len(terms)):
        return True
    return bool(overlap and passages and passages[0].get('score', 0) >= .72)


def status():
    try:
        with driver().session(database=NEO4J_DATABASE) as session:
            count = session.run('MATCH (n:KnowledgeItem) RETURN count(n) AS n').single()['n']
            return {'neo4j': 'connected', 'neo4j_version': _neo4j_version, 'knowledge_items': count}
    except Exception as exc:
        return {'neo4j': 'unavailable', 'error': type(exc).__name__}
