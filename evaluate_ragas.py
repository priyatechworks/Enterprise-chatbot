"""Optional answer-faithfulness and retrieval-context recall evaluation via Ragas."""
import asyncio
import json
import os
import requests

from app.config import DEMO_PASSWORD, EVALUATION_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, ROOT

BASE = os.getenv('RAG_API_URL', 'http://127.0.0.1:8000').rstrip('/')
OUTPUT = ROOT / 'outputs' / 'ragas_evaluation.json'


async def main():
    if not (LLM_API_KEY and LLM_BASE_URL and LLM_MODEL):
        raise SystemExit('Configure LLM_BASE_URL, LLM_API_KEY and LLM_MODEL before running Ragas evaluation.')
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextRecall, Faithfulness
    except ImportError as exc:
        raise SystemExit('Install optional evaluation dependencies with: python -m pip install -r requirements-eval.txt') from exc

    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    judge = llm_factory(LLM_MODEL, client=client)
    faithfulness = Faithfulness(llm=judge)
    context_recall = ContextRecall(llm=judge)
    rows = json.loads(EVALUATION_PATH.read_text(encoding='utf-8'))
    results = []
    auth = requests.post(
        f'{BASE}/login', json={'username': 'employee', 'password': DEMO_PASSWORD}, timeout=15,
    )
    auth.raise_for_status()
    headers = {'Authorization': f"Bearer {auth.json()['access_token']}"}

    for case in rows:
        if not case.get('expected_answer') or not case.get('expected_document'):
            continue
        answer = requests.post(
            f'{BASE}/ask', headers=headers,
            json={'question': case['question']}, timeout=180,
        )
        answer.raise_for_status()
        payload = answer.json()
        contexts = [source['excerpt'] for source in payload.get('sources', [])]
        faith = await faithfulness.ascore(
            user_input=case['question'], response=payload['answer'], retrieved_contexts=contexts,
        )
        recall = await context_recall.ascore(
            user_input=case['question'], retrieved_contexts=contexts, reference=case['expected_answer'],
        )
        results.append({
            'id': case['id'], 'faithfulness': float(faith.value),
            'context_recall': float(recall.value), 'mode': payload['mode'],
        })
        print(f"{case['id']}: faithfulness={faith.value:.3f}, context_recall={recall.value:.3f}")

    summary = {
        'samples': len(results),
        'mean_faithfulness': sum(row['faithfulness'] for row in results) / max(1, len(results)),
        'mean_context_recall': sum(row['context_recall'] for row in results) / max(1, len(results)),
        'results': results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'results'}, indent=2))
    print(f'Saved Ragas results to {OUTPUT}')


if __name__ == '__main__':
    asyncio.run(main())
