"""PDF ingestion, hybrid retrieval, and role-scoped Neo4j graph search."""
import csv
import hashlib
import importlib
import json
import os
import re
import uuid
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
    RETRIEVAL_THRESHOLD, TEXT_MODEL, VISION_MODEL,
)
from app.storage import get_document, record_failure, upsert_document
from app.metadata import persist_extracted_records
from app.vision import describe_image
from app.agent import plan_query

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


def _vector_query(session, query_vector, role, limit, candidate_limit, restricted=False):
    if _neo4j_supports_search:
        authorization = 'NOT $role IN node.allowed_roles' if restricted else '$role IN node.allowed_roles'
        similarity = ' AND score>=0.34' if restricted else ''
        return list(session.run(
            'CYPHER 25 MATCH (node:KnowledgeItem) '
            'SEARCH node IN (VECTOR INDEX knowledge_embedding FOR $embedding LIMIT $limit) SCORE AS score '
            f'WHERE {authorization}{similarity} '
            + ('RETURN score ORDER BY score DESC LIMIT 1' if restricted else 'RETURN node,score ORDER BY score DESC LIMIT $candidate_limit'),
            embedding=query_vector, limit=limit, candidate_limit=candidate_limit, role=role,
        ))
    if restricted:
        return list(session.run(
            "CALL db.index.vector.queryNodes('knowledge_embedding',$limit,$embedding) YIELD node,score "
            "WHERE NOT $role IN node.allowed_roles AND score>=0.34 RETURN score ORDER BY score DESC LIMIT 1",
            limit=limit, embedding=query_vector, role=role,
        ))
    return list(session.run(
        "CALL db.index.vector.queryNodes('knowledge_embedding',$limit,$embedding) YIELD node,score "
        "WHERE $role IN node.allowed_roles RETURN node,score ORDER BY score DESC LIMIT $candidate_limit",
        limit=limit, candidate_limit=candidate_limit, embedding=query_vector, role=role,
    ))


def _fulltext_query(session, lucene_query, role, limit, candidate_limit, restricted=False):
    if _neo4j_supports_fulltext_search:
        authorization = 'NOT $role IN node.allowed_roles' if restricted else '$role IN node.allowed_roles'
        return list(session.run(
            'CYPHER 25 MATCH (node:KnowledgeItem) '
            'SEARCH node IN (FULLTEXT INDEX knowledge_text FOR $lucene_query LIMIT $limit) SCORE AS score '
            f'WHERE {authorization} '
            + ('RETURN score ORDER BY score DESC LIMIT 1' if restricted else 'RETURN node,score ORDER BY score DESC LIMIT $candidate_limit'),
            lucene_query=lucene_query, limit=limit, candidate_limit=candidate_limit, role=role,
        ))
    if restricted:
        return list(session.run(
            "CALL db.index.fulltext.queryNodes('knowledge_text',$lucene_query,{limit:$limit}) YIELD node,score "
            "WHERE NOT $role IN node.allowed_roles RETURN score ORDER BY score DESC LIMIT 1",
            lucene_query=lucene_query, limit=limit, role=role,
        ))
    return list(session.run(
        "CALL db.index.fulltext.queryNodes('knowledge_text',$lucene_query,{limit:$limit}) YIELD node,score "
        "WHERE $role IN node.allowed_roles RETURN node,score ORDER BY score DESC LIMIT $candidate_limit",
        lucene_query=lucene_query, limit=limit, candidate_limit=candidate_limit, role=role,
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
    try:
        from docling_core.types.doc import TextItem
    except Exception:
        TextItem = ()
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
            "headings": [],
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
                "headings": [],
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
                "asset_path": str(target.resolve()), "headings": [],
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
    text_records = [record for record in records if record['kind'] != 'image']
    vectors = iter(text_embeddings([record['text'] for record in text_records]))
    rows = []
    for ordinal, record in enumerate(records):
        vector = image_embedding(record['asset_path']) if record['kind'] == 'image' else next(vectors)
        chunk_id = f'{document_id}:{ordinal}'
        rows.append({
            'id': chunk_id, 'chunk_id': chunk_id, 'document': name,
            'document_id': document_id, 'classification': classification,
            'allowed_roles': roles, 'page': int(record['page']), 'kind': record['kind'],
            'text': record['text'], 'headings': record.get('headings', []),
            'asset_path': record.get('asset_path', ''), 'ordinal': ordinal,
            'embedding': vector.tolist(),
        })
    return rows


def _replace_graph_document(name, document_id, classification, roles, page_count, rows):
    database = driver()
    old_assets = []

    def replace(tx):
        old_assets.extend([
            record['asset_path'] for record in tx.run(
                "MATCH (k:KnowledgeItem) WHERE k.document_id=$document_id OR k.document=$name "
                "RETURN k.asset_path AS asset_path", document_id=document_id, name=name,
            ) if record['asset_path']
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
                "CREATE (k:KnowledgeItem {id:row.id,chunk_id:row.chunk_id,document:row.document," 
                "document_id:row.document_id,classification:row.classification,allowed_roles:row.allowed_roles," 
                "page:row.page,kind:row.kind,text:row.text,headings:row.headings,asset_path:row.asset_path," 
                "ordinal:row.ordinal,embedding:row.embedding}) CREATE (d)-[:HAS_ITEM]->(k)",
                document_id=document_id, rows=rows,
            )
            tx.run(
                "MATCH (d:Document {document_id:$document_id})-[:HAS_ITEM]->(a:KnowledgeItem), "
                "(d)-[:HAS_ITEM]->(b:KnowledgeItem) WHERE b.ordinal=a.ordinal+1 "
                "MERGE (a)-[:NEXT]->(b)", document_id=document_id,
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
    try:
        if path.suffix.lower() != '.pdf' or not name.lower().endswith('.pdf'):
            raise ValueError('Only PDF files supported')
        roles = _policy_roles(classification)
        existing = get_document(name)
        document_id = (existing.get('document_id') if existing else None) or document_id or str(uuid.uuid4())
        try:
            page_count, records = _extract_docling(path, name)
            parse_mode, parse_warning = 'docling-hybrid-chunker', None
        except Exception as parse_error:
            page_count, records = _fallback_text_records(path)
            parse_mode, parse_warning = 'pymupdf-tokenizer-fallback', type(parse_error).__name__
            if not records:
                raise ValueError(f'Docling failed ({parse_warning}) and PyMuPDF found no selectable text')
        semantic_graph = _extract_semantic_graph(path)
        rows = _prepare_graph_rows(name, document_id, classification, roles, records)
        count = _replace_graph_document(name, document_id, classification, roles, page_count, rows)
        semantic_warning = None
        try:
            semantic_counts = _store_semantic_graph(name, document_id, roles, semantic_graph)
        except Exception as exc:
            semantic_counts = {'entities': 0, 'relations': 0}
            semantic_warning = f'Optional semantic graph storage failed ({type(exc).__name__}); document text remains indexed'
        upsert_document(name, page_count, count, roles, document_id, classification)
        result = {
            'name': name, 'document_id': document_id, 'classification': classification,
            'allowed_roles': roles, 'pages': page_count, 'chunks': count,
            'tables': sum(row['kind'] == 'table' for row in rows),
            'images': sum(row['kind'] == 'image' for row in rows),
            'entities': semantic_counts['entities'], 'relations': semantic_counts['relations'],
            'parser': parse_mode, 'status': 'indexed',
        }
        if parse_warning:
            result['warning'] = f'Docling failed; indexed selectable text via PyMuPDF ({parse_warning})'
        if semantic_warning:
            result['warning'] = '; '.join(filter(None, [result.get('warning'), semantic_warning]))
        return result
    except Exception as exc:
        record_failure(name, str(exc))
        return {'name': name, 'status': 'failed', 'message': str(exc)}


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
                'kind': node['kind'], 'text': node['text'],
                'asset_path': node.get('asset_path', ''),
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


def search(query, role, top_k=5, history=None, max_hops=None):
    database = driver()
    query_vector = np.asarray(text_embeddings([query], query=True)[0], dtype=np.float32).tolist()
    candidate_limit = max(RETRIEVAL_CANDIDATE_K, top_k)
    index_limit = candidate_limit * 5  # over-fetch to account for inaccessible hits before ACL filtering
    keyword_query = _lucene_query(query)
    with database.session(database=NEO4J_DATABASE) as session:
        vector_rows = _vector_query(session, query_vector, role, index_limit, candidate_limit)
        keyword_rows = []
        restricted_vector = []
        restricted_keyword = []
        if keyword_query:
            keyword_rows = _fulltext_query(session, keyword_query, role, index_limit, candidate_limit)
            # These denial probes return scores only; no restricted document text or name escapes Neo4j.
            restricted_keyword = _fulltext_query(session, keyword_query, role, index_limit, candidate_limit, restricted=True)
        restricted_vector = _vector_query(session, query_vector, role, index_limit, candidate_limit, restricted=True)

        ranked = reciprocal_rank_fusion(vector_rows, keyword_rows)
        fused = {item['id']: item for item in ranked}
        seeds = [item['id'] for item in ranked[:min(top_k, 5)]]
        expanded = []
        hops = min(max_hops if max_hops is not None else GRAPH_MAX_HOPS, 3)
        if seeds and hops:
            expanded += list(session.run(
                "MATCH (seed:KnowledgeItem) WHERE seed.id IN $ids AND $role IN seed.allowed_roles "
                "MATCH p=(seed)-[:NEXT*1..2]-(neighbor:KnowledgeItem) "
                "WHERE length(p)<=$hops AND $role IN neighbor.allowed_roles "
                "RETURN DISTINCT neighbor ORDER BY neighbor.document,neighbor.page LIMIT 12",
                ids=seeds, role=role, hops=hops,
            ))
            if DOCLING_GRAPH_TEMPLATE:
                expanded += list(session.run(
                    "MATCH (seed:KnowledgeItem)-[:MENTIONS]->(anchor:KGEntity) "
                    "WHERE seed.id IN $ids AND $role IN seed.allowed_roles AND $role IN anchor.allowed_roles "
                    "MATCH p=(anchor)-[:KG_RELATION*1..2]-(related:KGEntity) "
                    "WHERE $role IN related.allowed_roles AND all(r IN relationships(p) WHERE $role IN r.allowed_roles) "
                    "MATCH (neighbor:KnowledgeItem)-[:MENTIONS]->(related) "
                    "WHERE $role IN neighbor.allowed_roles RETURN DISTINCT neighbor "
                    "ORDER BY neighbor.document,neighbor.page LIMIT 12",
                    ids=seeds, role=role,
                ))
        seen = set(fused)
        for result in expanded:
            node = result['neighbor']
            chunk_id = node.get('chunk_id') or node.get('id')
            if chunk_id not in seen:
                fused[chunk_id] = {
                    'id': chunk_id, 'chunk_id': chunk_id,
                    'document_id': node.get('document_id', ''), 'document': node['document'],
                    'page': int(node['page']), 'kind': node['kind'], 'text': node['text'],
                    'asset_path': node.get('asset_path', ''),
                    'allowed_roles': list(node.get('allowed_roles', [])),
                    'score': 0.31, 'rrf_score': 0.0, 'graph_expanded': True,
                }
                seen.add(chunk_id)
        candidates = sorted(fused.values(), key=lambda item: item['rrf_score'], reverse=True)
        # ACL was applied by Neo4j before text was returned; only these allowed rows are reranked.
        candidates = _rerank(query, candidates)
        denied = bool(restricted_vector or restricted_keyword)
        return candidates[:top_k], denied


def agentic_search(query, role, top_k=5, history=None):
    """Bounded deterministic query decomposition with hybrid retrieval and graph expansion."""
    facets = [part.strip() for part in re.split(r'\b(?:and also|and|also|plus)\b|;', query, flags=re.I) if len(part.strip()) > 2]
    queries = list(dict.fromkeys([query] + [part for part in facets if part.lower() != query.lower()]))[:3]
    merged = {}
    denied = False
    for step, subquery in enumerate(queries):
        hits, blocked = search(subquery, role, top_k, history=history, max_hops=min(GRAPH_MAX_HOPS, 2))
        denied = denied or blocked
        for hit in hits:
            key = hit['id']
            if key not in merged or hit.get('rrf_score', 0) > merged[key].get('rrf_score', 0):
                merged[key] = hit
        current = list(merged.values())
        if current and has_evidence_overlap(subquery, current):
            continue
        if step >= 2:
            break
    result = sorted(merged.values(), key=lambda item: item.get('rrf_score', 0), reverse=True)[:top_k]
    return result, denied, {'rounds': len(queries), 'graph_hops': min(GRAPH_MAX_HOPS, 2)}


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
