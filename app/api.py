import time, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from app.config import (ASSET_DIR, DOCS_DIR, UPLOAD_DIR, TMP_DIR, MAX_UPLOAD_MB, NEO4J_DATABASE, RETRIEVAL_THRESHOLD, SESSION_TTL_SECONDS,
                        SESSION_MAX_TURNS, USER_MAX_SESSIONS, RATE_LIMIT_REQUESTS,
                        RATE_LIMIT_WINDOW_SECONDS, TEXT_MODEL, LLM_MODEL, RAG_DEBUG_TRACE)
from app.trace import RequestTrace
from app.storage import (init_db,list_documents,allow_request,create_session,
                         get_session,touch_session,deactivate_session,add_messages,get_history)
from app.rag import driver as graph_driver, ingest_file,agentic_search,has_evidence_overlap,load_manifest,status
from app.generation import answer
from app.auth import authenticate,current_user
from app.guardrails import is_prompt_injection,detect_pii

app=FastAPI(title='Enterprise Graph RAG Demo',version='2.0')
init_db()

class AskRequest(BaseModel):
    question:str=Field(min_length=2,max_length=1500)
    top_k:int=Field(default=3,ge=1,le=3)
    session_id:str|None=None
class LoginRequest(BaseModel):username:str;password:str

def _limit(key,max_count=RATE_LIMIT_REQUESTS):
    ok,retry=allow_request(key,time.time(),max_count,RATE_LIMIT_WINDOW_SECONDS)
    if not ok:raise HTTPException(status_code=429,detail='Rate limit exceeded',headers={'Retry-After':str(retry)})

def _flow_step(trace,step,component,status='complete',started=None,detail=''):
    row={'step':step,'component':component,'status':status}
    if started is not None:row['duration_ms']=round((time.perf_counter()-started)*1000,1)
    if detail:row['detail']=detail
    trace.append(row)

@app.middleware('http')
async def request_id_and_errors(request:Request,call_next):
    request_id=str(uuid.uuid4())
    try:
        response=await call_next(request)
    except Exception:
        response=JSONResponse(status_code=503,content={'detail':'Service temporarily unavailable','request_id':request_id})
    response.headers['X-Request-ID']=request_id
    return response

@app.get('/health')
def health():return {'status':'ok','storage':status()}

@app.post('/login')
def login(req:LoginRequest,request:Request):
    _limit(f"login:{request.client.host if request.client else 'unknown'}",8)
    data=authenticate(req.username,req.password)
    sid=str(uuid.uuid4());now=time.time()
    create_session(sid,req.username.lower(),data['role'],now,USER_MAX_SESSIONS)
    data['session_id']=sid
    return data

@app.post('/sessions')
def new_session(user=Depends(current_user)):
    sid=str(uuid.uuid4());create_session(sid,user['sub'],user['role'],time.time(),USER_MAX_SESSIONS)
    return {'session_id':sid,'expires_in_seconds':SESSION_TTL_SECONDS,'max_turns':SESSION_MAX_TURNS}

@app.delete('/sessions/{session_id}')
def close_session(session_id:str,user=Depends(current_user)):
    if not deactivate_session(session_id,user['sub']):raise HTTPException(status_code=404,detail='Session not found')
    return {'closed':True}

@app.get('/documents')
def documents(user=Depends(current_user)):
    return [d for d in list_documents() if user['role'] in d.get('allowed_roles',[])]

@app.get('/assets/{asset_name}')
def get_asset(asset_name:str,user=Depends(current_user)):
    # Asset IDs are opaque; authorize via graph metadata before returning bytes.
    if Path(asset_name).name!=asset_name:raise HTTPException(status_code=404,detail='Asset not found')
    target=(ASSET_DIR/asset_name).resolve()
    if not target.is_relative_to(ASSET_DIR.resolve()) or not target.is_file():raise HTTPException(status_code=404,detail='Asset not found')
    with graph_driver().session(database=NEO4J_DATABASE) as s:
        rec=s.run('MATCH (n:KnowledgeItem {asset_path:$path}) RETURN $role IN n.allowed_roles AS allowed LIMIT 1',path=str(target),role=user['role']).single()
    if not rec or not rec['allowed']:raise HTTPException(status_code=404,detail='Asset not found')
    return FileResponse(target,media_type='image/png')

def _admin(user):
    if user['role']!='Admin':raise HTTPException(status_code=403,detail='Admin role required')

@app.post('/ingest/folder')
def ingest_folder(user=Depends(current_user)):
    _admin(user);files=sorted(DOCS_DIR.glob('*.pdf'))
    if not files:return {'message':'No PDFs found in data/sample_docs','results':[]}
    manifest=load_manifest();start=time.perf_counter()
    results=[ingest_file(path,classification=manifest.get(path.name,{}).get('classification','Admin-only'),document_id=manifest.get(path.name,{}).get('document_id')) for path in files]
    return {'results':results,'items_indexed':sum(x.get('chunks',0) for x in results),'seconds':round(time.perf_counter()-start,2)}

@app.post('/ingest/upload')
def ingest_upload(files:list[UploadFile]=File(...),classification:str=Form('Admin-only'),user=Depends(current_user)):
    _admin(user)
    if classification not in {'Public','Admin-only'}:raise HTTPException(status_code=422,detail="Classification must be 'Public' or 'Admin-only'")
    results=[];seen=set()
    for upload in files:
        name=Path(upload.filename or 'upload.pdf').name
        if Path(name).suffix.lower()!='.pdf':results.append({'name':name,'status':'failed','message':'Only PDF files accepted'});continue
        if name.casefold() in seen:results.append({'name':name,'status':'failed','message':'Duplicate filename in this upload batch'});continue
        seen.add(name.casefold())
        data=upload.file.read(MAX_UPLOAD_MB*1024*1024+1)
        if len(data)>MAX_UPLOAD_MB*1024*1024:results.append({'name':name,'status':'failed','message':f'File exceeds {MAX_UPLOAD_MB} MB'});continue
        if not data.startswith(b'%PDF-'):results.append({'name':name,'status':'failed','message':'File does not have a valid PDF header'});continue
        temporary=TMP_DIR/f'{uuid.uuid4().hex}.pdf'
        target=UPLOAD_DIR/name
        try:
            temporary.parent.mkdir(parents=True,exist_ok=True)
            temporary.write_bytes(data)
            result=ingest_file(temporary,classification=classification,document_name=name)
            if result.get('status')=='indexed':
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary.replace(target)
            results.append(result)
        finally:
            temporary.unlink(missing_ok=True)
    return {'results':results,'items_indexed':sum(x.get('chunks',0) for x in results),'classification':classification}

@app.post('/ask')
def ask(req:AskRequest,user=Depends(current_user)):
    start=time.perf_counter();now=time.time();flow_trace=[]
    debug_trace = RequestTrace(enabled=True) if RAG_DEBUG_TRACE else None
    if debug_trace:
        debug_trace.add(
            "request_received", "FastAPI POST /ask",
            details={"question": req.question, "role": user["role"], "top_k": req.top_k,
                     "session_id": req.session_id},
        )
    _flow_step(flow_trace,'Identity check','FastAPI bearer dependency + HMAC-signed demo token','passed',detail=f"Authenticated role: {user['role']}")
    rate_start=time.perf_counter();_limit(f"ask:{user['sub']}")
    _flow_step(flow_trace,'Rate limit','SQLite request window', 'passed',rate_start,detail=f"Limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s")
    guard_start=time.perf_counter();injected=is_prompt_injection(req.question)
    _flow_step(flow_trace,'Question safety screen','app.guardrails.is_prompt_injection (Python regex checks)','blocked' if injected else 'passed',guard_start)
    if injected:
        _flow_step(flow_trace,'Answer generation','Skipped; unsafe question was blocked','skipped')
        return {'answer':'Request blocked by the prompt-injection safeguard.','sources':[],'mode':'blocked','session_id':req.session_id,'latency_seconds':round(time.perf_counter()-start,3),'flow_trace':flow_trace}
    session_start=time.perf_counter()
    session=get_session(req.session_id) if req.session_id else None
    if session and (session['username']!=user['sub'] or session['role']!=user['role']):raise HTTPException(status_code=404,detail='Session not found')
    if session and not session['active']:raise HTTPException(status_code=410,detail='Session is closed')
    if session and now-session['last_seen']>SESSION_TTL_SECONDS:raise HTTPException(status_code=410,detail='Session expired; create a new session')
    if session and session['turns']>=SESSION_MAX_TURNS:raise HTTPException(status_code=429,detail='Session turn limit reached; create a new session')
    if not session:
        sid=str(uuid.uuid4());create_session(sid,user['sub'],user['role'],now,USER_MAX_SESSIONS);session=get_session(sid)
    history=get_history(session['id'])
    _flow_step(flow_trace,'Session and conversation context','app.storage + SQLite', 'loaded',session_start,detail=f"Session ID: {session['id'][:8]}…; recent turns loaded")
    # Resolve follow-up references with a bounded window of this user's own session only.
    prior_questions=[m['content'] for m in history if m['role']=='user'][-2:]
    retrieval_query=' '.join(prior_questions+[req.question]) if prior_questions else req.question
    retrieval_start=time.perf_counter()
    try:
        if debug_trace:
            passages,restricted_match,agent_trace=agentic_search(
                retrieval_query,user['role'],req.top_k,history=history,debug_trace=debug_trace
            )
        else:
            passages,restricted_match,agent_trace=agentic_search(retrieval_query,user['role'],req.top_k,history=history)
    except Exception as exc:
        if debug_trace:
            debug_trace.add("retrieval_failed", "Agentic retrieval pipeline", status="failed",
                            details={"exception_type": type(exc).__name__, "exception_message": str(exc)})
        _flow_step(flow_trace,'Embedding and graph retrieval',f"Sentence Transformers ({TEXT_MODEL}) + Neo4j AuraDB vector index / Cypher","failed",retrieval_start,detail=f"{type(exc).__name__}; inspect backend logs and database connection")
        detail={'message':f"Retrieval service unavailable ({type(exc).__name__})",'flow_trace':flow_trace}
        if debug_trace:
            detail['debug_trace']=debug_trace.events
        raise HTTPException(status_code=503,detail=detail)
    _flow_step(flow_trace,'Agent-planned hybrid retrieval, ACL and graph expansion',f"Query planner + Nomic ({TEXT_MODEL}) text/figure vectors + Neo4j vector and BM25 indexes + RRF + Cypher ACL + graph traversal","complete",retrieval_start,detail=f"{len(passages)} candidates; planner={agent_trace.get('planner','heuristic')}; requested types={','.join(agent_trace.get('requested_types', ['text']))}; {agent_trace.get('rounds',0)} modality searches; up to {agent_trace.get('graph_hops',0)} graph hops; role filter: {user['role']}")
    if debug_trace:
        debug_trace.add(
            "agentic_retrieval_completed", "app.rag.agentic_search",
            started=retrieval_start,
            details={"candidate_count_before_evidence_gate": len(passages), "restricted_match": restricted_match,
                     "planner": agent_trace.get("planner"), "requested_types": agent_trace.get("requested_types")},
        )
    evidence_start=time.perf_counter()
    if restricted_match:
        text='Access denied: your role cannot access the document needed for this request.'; mode='access-denied';passages=[]
        evidence_status='blocked'
        if debug_trace:
            debug_trace.add("evidence_gate", "Authorization gate", status="blocked",
                            details={"reason": "A restricted matching document was detected; content was withheld"})
        _flow_step(flow_trace,'Evidence and relevance gate','Python similarity threshold + evidence-overlap check',evidence_status,evidence_start,detail='Restricted match found; evidence withheld')
        _flow_step(flow_trace,'Answer generation','Skipped because role authorization denied the request','skipped')
    else:
        retrieved_candidates = list(passages)
        thresholded = [p for p in retrieved_candidates if p['score'] >= RETRIEVAL_THRESHOLD]
        overlap_passed = has_evidence_overlap(req.question, thresholded)
        if debug_trace:
            debug_trace.add(
                "evidence_gate", "Similarity threshold + evidence overlap",
                started=evidence_start,
                details={"before_filter": len(retrieved_candidates), "above_threshold": len(thresholded),
                         "threshold": RETRIEVAL_THRESHOLD, "question_overlap_passed": overlap_passed,
                         "final_evidence_count": len(thresholded) if overlap_passed else 0},
                data={"items": [
                    {"id": p.get("id"), "document": p.get("document"), "page": p.get("page"),
                     "content_type": p.get("content_type", p.get("kind")), "score": p.get("score"),
                     "threshold_passed": p.get("score", 0) >= RETRIEVAL_THRESHOLD,
                     "text": p.get("text", ""), "latex": p.get("latex", ""),
                     "asset_path": p.get("asset_path", ""), "file_path": p.get("file_path", "")}
                    for p in retrieved_candidates
                ]},
            )
        passages = thresholded if overlap_passed else []
        evidence_status='passed' if passages else 'abstained'
        _flow_step(flow_trace,'Evidence and relevance gate','Python similarity threshold + evidence-overlap check',evidence_status,evidence_start,detail=f"{len(passages)} passages passed; threshold={RETRIEVAL_THRESHOLD}")
        retrieved_injection=any(is_prompt_injection(p['text']) for p in passages)
        _flow_step(flow_trace,'Retrieved-context safety screen','app.guardrails.is_prompt_injection (Python regex checks)','blocked' if retrieved_injection else ('passed' if passages else 'skipped'),detail='Checked retrieved passages before generation' if passages else 'No passages to inspect')
        if retrieved_injection:
            text='Retrieved text was blocked by the prompt-injection safeguard.';mode='blocked';passages=[]
            if debug_trace:
                debug_trace.add("retrieved_context_safety", "Retrieved-context prompt-injection screen",
                                status="blocked", details={"reason": "Potential prompt injection found in retrieved passage"})
            _flow_step(flow_trace,'Answer generation','Skipped because retrieved text was blocked','skipped')
        else:
            answer_start=time.perf_counter()
            if debug_trace:
                text,mode=answer(req.question,passages,history,debug_trace=debug_trace)
            else:
                text,mode=answer(req.question,passages,history)
            answer_component='app.generation.answer + OpenAI-compatible LLM API' if mode=='llm' else 'app.generation.answer (retrieval-only / abstention fallback)'
            answer_detail=f"Mode: {mode}"+(f"; model: {LLM_MODEL}" if mode=='llm' and LLM_MODEL else '')
            _flow_step(flow_trace,'Answer generation',answer_component,'called' if mode=='llm' else ('abstained' if mode=='no evidence' else 'fallback'),answer_start,detail=answer_detail)
    if debug_trace:
        debug_trace.add(
            "response_ready", "FastAPI response serializer",
            details={"answer_mode": mode, "final_evidence_count": len(passages),
                     "total_request_elapsed_ms": round((time.perf_counter()-start)*1000, 1)},
            data={"answer": text, "citations": [
                {"document": p["document"], "page": p["page"], "type": p.get("content_type", p["kind"]),
                 "score": p.get("score"), "asset_path": p.get("asset_path", ""),
                 "file_path": p.get("file_path", ""), "latex": p.get("latex", "")}
                for p in passages
            ]},
        )
    touch_session(session['id'],now);add_messages(session['id'],req.question,text,now)
    pii_start=time.perf_counter();pii_detected=sorted({c for p in passages for c in detect_pii(p['text'])})
    _flow_step(flow_trace,'PII pattern scan','app.guardrails.detect_pii (Python regex checks)','scanned',pii_start,detail=f"Detected categories: {', '.join(pii_detected) if pii_detected else 'none'}; detection only, no redaction")
    _flow_step(flow_trace,'Citations and response','FastAPI response serializer','complete',detail=f"{len(passages)} source citations prepared")
    return {'answer':text,'sources':[{'document':p['document'],'document_id':p.get('document_id'),'chunk_id':p.get('chunk_id',p.get('id')),'page':p['page'],'kind':p['kind'],'content_type':p.get('content_type',p['kind']),'caption':p.get('caption',''),'section':p.get('section',''),'latex':p.get('latex',''),'score':round(p['score'],3),'excerpt':p['text'],'asset_name':Path(p['asset_path']).name if p.get('asset_path') else None} for p in passages], 'mode':mode,'agent_trace':agent_trace,'flow_trace':flow_trace,'debug_trace':debug_trace.events if debug_trace else [],'pii_detected':pii_detected,'session_id':session['id'],'session_turns':session['turns']+1,'latency_seconds':round(time.perf_counter()-start,3)}
