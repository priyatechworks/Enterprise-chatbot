import requests
from pathlib import Path
import streamlit as st

st.set_page_config(page_title='Enterprise Graph RAG',page_icon='📚',layout='wide')
API=st.sidebar.text_input('Backend URL','http://localhost:8000')
# Request-flow panel is collapsed by default. Set this False to remove it from the UI.
SHOW_REQUEST_FLOW_PANEL=True

title_col,flow_col=st.columns([0.84,0.16],vertical_alignment='center')
with title_col:
 st.title('📚 Enterprise Graph RAG Assistant')
 st.caption('Docling PDF understanding · Nomic text/image embeddings · Neo4j graph retrieval · role-filtered answers')
flow_slot=flow_col.empty()

with st.sidebar:
 st.subheader('Demo sign-in')
 username=st.selectbox('Account',['employee','admin'],index=0)
 password=st.text_input('Password',type='password',value='')
 if st.button('Sign in'):
  try:
   r=requests.post(f'{API}/login',json={'username':username,'password':password},timeout=10);r.raise_for_status();data=r.json()
   st.session_state['token']=data['access_token'];st.session_state['role']=data['role'];st.session_state['session_id']=data['session_id'];st.session_state['messages']=[]
  except Exception as e:
   response=getattr(e,'response',None)
   try:detail=response.json().get('detail') if response is not None else None
   except Exception:detail=None
   st.error(f'Sign-in failed: {detail or type(e).__name__}')
 token=st.session_state.get('token','');headers={'Authorization':f'Bearer {token}'}
 if token:
  st.caption(f"Signed in as {st.session_state['role']} · session {st.session_state.get('session_id','')[:8]}")
  if st.button('New chat'):
   try:
    r=requests.post(f'{API}/sessions',headers=headers,timeout=10);r.raise_for_status();st.session_state['session_id']=r.json()['session_id'];st.session_state['messages']=[];st.rerun()
   except Exception as e:st.error(f'Could not start session: {e}')
  if st.button('Sign out'):
   for key in ('token','role','session_id','messages'):st.session_state.pop(key,None)
   st.rerun()
 if st.session_state.get('role')=='Admin':
  st.subheader('Bulk document ingestion');mode=st.radio('Input',['Upload multiple PDFs','Import sample_docs folder'])
 else:mode=None;st.caption('Only Admin may ingest documents.')
 if mode=='Upload multiple PDFs':
  classification=st.selectbox('Document access',['Admin-only','Public'],index=0,help='Admin-only is the default. Public documents are available to Admin and Employee accounts.')
  uploads=st.file_uploader('Select PDFs',type=['pdf'],accept_multiple_files=True)
  if st.button('Index selected PDFs',disabled=not uploads):
   try:
    files=[('files',(u.name,u.getvalue(),'application/pdf')) for u in uploads]
    with st.spinner('Parsing, chunking and indexing batch...'):r=requests.post(f'{API}/ingest/upload',files=files,data={'classification':classification},headers=headers,timeout=1800);r.raise_for_status()
    st.json(r.json())
   except Exception as e:
    response=getattr(e,'response',None)
    try:detail=response.json().get('detail') if response is not None else None
    except Exception:detail=None
    st.error(f'Ingestion failed: {detail or type(e).__name__}')
 elif mode=='Import sample_docs folder' and st.button('Import all sample PDFs'):
  try:
   with st.spinner('Parsing and indexing batch...'):r=requests.post(f'{API}/ingest/folder',headers=headers,timeout=1800);r.raise_for_status()
   st.json(r.json())
  except Exception as e:
   response=getattr(e,'response',None)
   try:detail=response.json().get('detail') if response is not None else None
   except Exception:detail=None
   st.error(f'Ingestion failed: {detail or type(e).__name__}')
 st.divider();st.subheader('Documents available to your role')
 if token:
  try:
   docs=requests.get(f'{API}/documents',headers=headers,timeout=15).json()
   if docs:st.dataframe(docs,use_container_width=True,hide_index=True)
   else:st.caption('No permitted documents indexed')
  except Exception:st.caption('Start Neo4j and the backend to view documents')
 st.caption('Fictional sample data only. Do not use real confidential data in this demo.')

def show_sources(sources,headers):
 for source in sources:
  content_type=source.get('content_type',source.get('kind','text'))
  st.markdown(f"**{source['document']} · page {source['page']} · {content_type} · {source['score']}**")
  if source.get('caption'):st.caption(f"Caption: {source['caption']}")
  if source.get('section'):st.caption(f"Section: {source['section']}")
  if content_type=='formula' and source.get('latex'):
   formula=source['latex'].replace('$$','').replace('$','').strip()
   if formula:st.latex(formula)
   else:st.markdown(source['excerpt'])
  else:
   # Table chunks are retained as Docling markdown so Streamlit can render tabular rows.
   st.markdown(source['excerpt'])
  if source.get('asset_name'):
   try:
    response=requests.get(f"{API}/assets/{source['asset_name']}",headers=headers,timeout=15)
    response.raise_for_status()
    st.image(response.content,caption=f"{source['document']} · page {source['page']}")
   except Exception:
    st.caption('Figure asset could not be displayed')

def show_debug_trace(events):
 if not events:return
 final_event=next((event for event in reversed(events) if event.get('stage')=='agent_merge_final'),{})
 final_items=(final_event.get('data') or {}).get('items',[])
 plan_event=next((event for event in events if event.get('stage')=='query_plan'),{})
 plan=(plan_event.get('data') or {}).get('plan',{})
 context_event=next((event for event in events if event.get('stage')=='answer_context'),{})
 context_data=context_event.get('data') or {}
 llm_event=next((event for event in reversed(events) if event.get('stage')=='llm_request'),{})
 details=final_event.get('details') or {}
 with st.expander('Developer console · inspect the complete RAG pipeline',expanded=True):
  st.caption('Testing mode. This panel shows what each component received, returned and passed to the next component. Disable RAG_DEBUG_TRACE before a normal demo or production use.')
  c1,c2,c3,c4=st.columns(4)
  c1.metric('Final chunks',len(final_items),'maximum 3')
  c2.metric('Requested types',len(details.get('requested_types',[])))
  c3.metric('Diagram images sent',sum(1 for item in context_data.get('visual_inputs',[]) if item.get('attached_to_openai_request')))
  c4.metric('LLM model',llm_event.get('details',{}).get('model','not called'))
  tabs=st.tabs(['Pipeline','Selected chunks','Diagram images','LLM handoff','Raw events'])
  with tabs[0]:
   st.markdown('**1. Question → modality planner**')
   st.write('Planner:',plan.get('planner','not recorded'))
   if plan.get('planner_reason'):st.caption(plan['planner_reason'])
   if plan.get('tasks'):st.json(plan['tasks'],expanded=True)
   st.markdown('**2. Separate retrieval passes → fusion → top three**')
   if details:
    st.write({'requested_types':details.get('requested_types',[]),'selected_types':details.get('selected_types',[]),'types_without_evidence':details.get('types_without_evidence',[]),'unique_candidates':details.get('unique_items_before_top_k'),'final_chunks':details.get('returned')})
   st.markdown('**3. Component timeline and duration**')
   for index,event in enumerate(events,1):
    stage=event.get('stage','stage')
    duration=f" · {event['duration_ms']} ms" if event.get('duration_ms') is not None else ''
    status=event.get('status','unknown')
    icon='✅' if status in ('complete','passed','loaded','called') else ('⚠️' if status in ('warning','skipped','fallback','abstained') else ('❌' if status=='failed' else '•'))
    st.markdown(f"**{index}. {icon} {stage}**{duration} · +{event.get('elapsed_from_request_start_ms',0)} ms")
    st.caption(f"{event.get('component','')} · {status} · {event.get('details',{})}")
  with tabs[1]:
   if not final_items:st.info('No final evidence was selected. Check the retrieval and evidence-gate events.')
   for item in final_items:
    kind=item.get('content_type','text')
    with st.container(border=True):
     st.markdown(f"**#{item.get('rank','?')} · {kind.upper()} · {item.get('document','')} · page {item.get('page','?')}**")
     st.caption(f"Final score: {item.get('final_score',item.get('retrieval_score'))} · RRF: {item.get('rrf_score')} · graph expanded: {item.get('graph_expanded',False)}")
     if item.get('retrieved_by'):st.caption('Why retrieved: '+str(item['retrieved_by']))
     if kind=='formula' and item.get('latex'):st.latex(str(item['latex']).replace('$$','').replace('$','').strip())
     elif item.get('text'):st.markdown(str(item['text']))
     if kind=='table' and item.get('table_json'):st.json(item['table_json'],expanded=False)
     asset=item.get('asset_path')
     if asset and Path(asset).is_file():st.image(asset,caption=f"Retrieved figure · {item.get('document')} · page {item.get('page')}")
     st.caption(f"Asset path: {asset or 'none'}")
     st.caption(f"Extracted record path: {item.get('file_path') or 'not recorded'}")
  with tabs[2]:
   visuals=context_data.get('visual_inputs',[])
   if not visuals:st.info('No figure chunk was selected for the final answer.')
   for visual in visuals:
    with st.container(border=True):
     st.write({'document':visual.get('document'),'page':visual.get('page'),'asset_path':visual.get('asset_path'),'attached_to_openai_request':visual.get('attached_to_openai_request'),'warning':visual.get('warning')})
     if visual.get('asset_path') and Path(visual['asset_path']).is_file():st.image(visual['asset_path'],caption='Image input prepared for the OpenAI request')
  with tabs[3]:
   if context_data:
    st.markdown('**Exact text/LaTeX/table context sent to the answer model**')
    st.code(context_data.get('context_sent_to_model',''),language='text')
    st.markdown('**Visual image inputs**')
    st.json(context_data.get('visual_inputs',[]),expanded=True)
   else:st.info('No answer context was prepared. Check whether retrieval returned evidence.')
   if llm_event:
    st.markdown('**OpenAI response / request status**')
    st.json({'status':llm_event.get('status'),'details':llm_event.get('details'),'answer':(llm_event.get('data') or {}).get('answer')},expanded=True)
  with tabs[4]:
   for index,event in enumerate(events,1):
    st.markdown(f"**Event {index}: {event.get('stage','')}**")
    st.json(event,expanded=False)


if 'messages' not in st.session_state:st.session_state.messages=[]
for m in st.session_state.messages:
 with st.chat_message(m['role']):
  st.markdown(m['content'])
  if m.get('sources'):
   with st.expander('Evidence and sources'):show_sources(m['sources'],{'Authorization':f"Bearer {st.session_state.get('token','')}"})
  if m.get('debug_trace'):show_debug_trace(m['debug_trace'])
q=st.chat_input('Ask about documents')
if q:
 st.session_state.messages.append({'role':'user','content':q})
 with st.chat_message('user'):st.markdown(q)
 with st.chat_message('assistant'):
  try:
   if not st.session_state.get('token'):raise RuntimeError('Sign in first (employee / shared demo password)')
   payload={'question':q,'top_k':3,'session_id':st.session_state.get('session_id')}
   with st.spinner('Searching authorized graph evidence...'):
    r=requests.post(f'{API}/ask',json=payload,headers={'Authorization':f"Bearer {st.session_state['token']}"},timeout=180);r.raise_for_status();d=r.json()
   st.session_state['session_id']=d['session_id'];st.session_state['last_flow_trace']=d.get('flow_trace',[]);st.session_state['last_flow_summary']=f"{d['mode']} · {d['latency_seconds']} seconds"
   st.markdown(d['answer']);st.caption(f"{d['mode']} · turn {d.get('session_turns','?')} · {d['latency_seconds']} seconds")
   if d['sources']:
    with st.expander('Evidence and sources',expanded=True):show_sources(d['sources'],{'Authorization':f"Bearer {st.session_state['token']}"})
   debug_events=d.get('debug_trace',[])
   if debug_events:show_debug_trace(debug_events)
   st.session_state.messages.append({'role':'assistant','content':d['answer'],'sources':d['sources'],'debug_trace':debug_events})
  except Exception as e:
   detail=None
   response=getattr(e,'response',None)
   if response is not None:
    try:detail=response.json().get('detail')
    except Exception:detail=None
   if isinstance(detail,dict):
    st.session_state['last_flow_trace']=detail.get('flow_trace',[{'step':'FastAPI request','component':'HTTP API','status':'failed','detail':detail.get('message','Request failed')}])
    if detail.get('debug_trace'):show_debug_trace(detail['debug_trace'])
    st.error(f"Backend request failed: {detail.get('message','Request failed')}")
   else:
    st.session_state['last_flow_trace']=[{'step':'Browser to backend request','component':'Streamlit requests.post → FastAPI /ask','status':'failed','detail':f'{type(e).__name__}: {e}'}]
    st.error(f'Backend request failed: {e}')

# === Hideable request-flow panel. It shows the latest completed/failed chat call. ===
if SHOW_REQUEST_FLOW_PANEL:
 with flow_slot.popover('🔎 Request flow',use_container_width=True):
  st.caption('Latest chat request · components and timings · no prompt text or secrets shown')
  trace=st.session_state.get('last_flow_trace',[])
  if st.session_state.get('last_flow_summary'):
   st.caption(st.session_state['last_flow_summary'])
  if not trace:
   st.info('Ask a question to see which components run.')
  else:
   icons={'passed':'✅','complete':'✅','loaded':'✅','scanned':'✅','called':'🤖','fallback':'↪️','abstained':'⏭️','skipped':'⏭️','blocked':'🛑','failed':'❌'}
   for index,item in enumerate(trace,1):
    icon=icons.get(item.get('status'),'•')
    elapsed=f" · {item['duration_ms']} ms" if item.get('duration_ms') is not None else ''
    st.markdown(f"**{index}. {icon} {item.get('step','Step')}**{elapsed}")
    st.caption(f"{item.get('component','') } · status: {item.get('status','unknown')}")
    if item.get('detail'):st.caption(item['detail'])
