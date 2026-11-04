"""Bounded query planner and content-type router for multimodal RAG retrieval."""
import json
import re
import requests

from app.config import AGENT_PLANNER_ENABLED, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

VALID_TYPES = {"text", "table", "figure", "formula"}


def heuristic_plan(question: str) -> dict:
    lower = question.lower()
    kinds = {"text"}
    if re.search(r"\b(table|tabular|results?|benchmark|bleu|accuracy|score|comparison|performance|metrics?)\b", lower):
        kinds.add("table")
    if re.search(r"\b(figure|diagram|image|visual|architecture|flowchart|illustration|show me|display)\b", lower):
        kinds.add("figure")
    if re.search(r"\b(formula|equation|mathematical|math|derive|scaled dot|softmax|q\s*[,/]\s*k)\b", lower):
        kinds.add("formula")
    return {
        "tasks": [{"query": question.strip(), "content_types": [kind]}
                  for kind in ("text", "table", "figure", "formula") if kind in kinds],
        "requested_types": sorted(kinds),
        "planner": "heuristic",
    }


def _normalise_plan(payload: dict, question: str) -> dict:
    tasks = []
    for item in payload.get("tasks", [])[:4]:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or question).strip()[:1000]
        kinds = [str(kind).lower() for kind in item.get("content_types", [])
                 if str(kind).lower() in VALID_TYPES]
        if query and kinds:
            tasks.append({"query": query, "content_types": list(dict.fromkeys(kinds))})
    if not tasks:
        return heuristic_plan(question)
    requested = sorted({kind for task in tasks for kind in task["content_types"]})
    if "text" not in requested:
        tasks.insert(0, {"query": question, "content_types": ["text"]})
        requested.append("text")
    return {"tasks": tasks[:4], "requested_types": requested, "planner": "llm"}


def plan_query(question: str) -> dict:
    """Use an LLM JSON plan when configured; always fall back to deterministic routing."""
    fallback = heuristic_plan(question)
    if not (AGENT_PLANNER_ENABLED and LLM_API_KEY and LLM_BASE_URL and LLM_MODEL):
        return fallback
    system_prompt = (
        "You are a query planner for a research-paper chatbot. Do not answer the question. "
        "Return JSON only: {tasks:[{query:string,content_types:[text|table|figure|formula]}]}. "
        "Always include text. Add table for requested numeric/table evidence, figure for requested "
        "diagram/visual/architecture and formula for requested mathematical expressions. "
        "Use at most four focused tasks. Do not invent facts."
    )
    payload = {"model": LLM_MODEL, "temperature": 0, "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]}
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions", json=payload,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"}, timeout=20,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"^\s*```(?:json)?|```\s*$", "", str(raw).strip(), flags=re.I)
        return _normalise_plan(json.loads(raw), question)
    except Exception:
        return fallback
