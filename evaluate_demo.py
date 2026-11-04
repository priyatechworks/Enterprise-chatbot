"""Run deterministic smoke/evidence checks against a running local API."""
import json
import os
import re
import argparse
import requests

from app.config import DEMO_PASSWORD, EVALUATION_PATH
from app.rag import load_manifest

BASE = os.getenv('RAG_API_URL', 'http://127.0.0.1:8000').rstrip('/')
manifest = load_manifest()
document_names = {entry['document_id']: name for name, entry in manifest.items() if entry.get('document_id')}
cases = json.loads(EVALUATION_PATH.read_text(encoding='utf-8'))
failures = 0
stop_words = {'the','a','an','and','is','are','of','for','to','in','on','with','per','what','after','before','it','that','also','be','was','does'}
tokens = {}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ingest-sample', action='store_true', help='re-index sample_docs PDFs before the smoke evaluation')
parser.add_argument('--case', help='run only one evaluation case ID')
args = parser.parse_args()
if args.case:
    cases = [case for case in cases if case['id'] == args.case]
    if not cases:
        raise SystemExit(f'Unknown evaluation case: {args.case}')
if args.ingest_sample:
    auth = requests.post(f'{BASE}/login', json={'username': 'admin', 'password': DEMO_PASSWORD}, timeout=15)
    auth.raise_for_status()
    ingest = requests.post(
        f'{BASE}/ingest/folder', headers={'Authorization': f"Bearer {auth.json()['access_token']}"}, timeout=1800,
    )
    ingest.raise_for_status()
    ingestion = ingest.json()
    print(f"Sample ingestion: {ingestion.get('items_indexed', 0)} chunks")
    failed = [item for item in ingestion.get('results', []) if item.get('status') != 'indexed']
    if failed:
        print(json.dumps(failed, indent=2))
        raise SystemExit('Sample ingestion failed; smoke evaluation not run.')

for case in cases:
    account = case['role'].lower()
    if account not in tokens:
        auth = requests.post(f'{BASE}/login', json={'username': account, 'password': DEMO_PASSWORD}, timeout=10)
        auth.raise_for_status()
        tokens[account] = auth.json()['access_token']
    response = requests.post(
        f'{BASE}/ask', headers={'Authorization': f"Bearer {tokens[account]}"},
        json={'question': case['question']}, timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    test_type = case.get('test_type', '')
    if test_type == 'RBAC negative test':
        ok = result['mode'] == 'access-denied' and not result['sources']
    elif test_type == 'Prompt injection / RBAC':
        ok = result['mode'] == 'blocked' and not result['sources']
    elif test_type == 'Abstention':
        ok = result['mode'] == 'no evidence' and not result['sources']
    else:
        expected_name = document_names.get(case.get('expected_document'))
        matching = [source for source in result['sources'] if source['document'] == expected_name] if expected_name else []
        expected = set(re.findall(r'[a-z]+|\d+(?:[,.]\d+)*%?', case.get('expected_answer', '').lower())) - stop_words
        evidence = (result['answer'] + ' ' + ' '.join(source['excerpt'] for source in result['sources'])).lower()
        present = set(re.findall(r'[a-z]+|\d+(?:[,.]\d+)*%?', evidence))
        coverage = len(expected & present) / max(1, len(expected))
        page_hint = str(case.get('expected_page_hint', '')).strip()
        page_ok = not page_hint or any(str(source['page']) == page_hint for source in matching)
        # Token overlap is a lightweight smoke check (not a semantic answer-quality metric).
        # Keep the source document and expected PDF page as strict requirements.
        ok = bool(matching) and coverage >= 0.50 and page_ok and result['mode'] != 'llm-error'
    print(f"{'PASS' if ok else 'FAIL'} {case['id']} [{case['role']}] {result['mode']} sources={','.join(source['document'] for source in result['sources']) or '-'}")
    if not ok and test_type not in {'RBAC negative test', 'Prompt injection / RBAC', 'Abstention'}:
        print(f"  expected_page={page_hint or '-'} matching_pages={[source['page'] for source in matching]} expected_coverage={coverage:.2f}")
        print(f"  missing_terms={sorted(expected-present)}")
        print(f"  answer={result['answer'][:180].encode('ascii','replace').decode('ascii')}")
        print(f"  evidence={next((source['excerpt'][:260] for source in matching), 'no expected-document evidence').encode('ascii','replace').decode('ascii')}")
    elif not ok and test_type == 'Abstention':
        print(f"  abstention_expected mode={result['mode']} pages={[source['page'] for source in result['sources']]}")
        if result['sources']:
            print(f"  evidence={result['sources'][0]['excerpt'][:360].encode('ascii','replace').decode('ascii')}")
    failures += not ok

print(f'Evaluation: {len(cases)-failures}/{len(cases)} checks passed')
raise SystemExit(1 if failures else 0)
