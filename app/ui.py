import requests
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
 with st.expander('Developer trace · every retrieval step',expanded=True):
  st.caption('Local debugging only. Includes retrieved text, scores, model/DB steps, file paths and timings. Review this before changing the UI to a normal chatbot.')
  for index,event in enumerate(events,1):
   duration=f" · {event['duration_ms']} ms" if event.get('duration_ms') is not None else ''
   elapsed=f" · +{event.get('elapsed_from_request_start_ms',0)} ms"
   st.markdown(f"**{index}. {event.get('stage','Stage')}**{duration}{elapsed}")
   st.caption(f"{event.get('component','')} · {event.get('status','unknown')} · {event.get('timestamp','')}")
   if event.get('details'):st.caption(str(event['details']))
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
   payload={'question':q,'session_id':st.session_state.get('session_id')}
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
