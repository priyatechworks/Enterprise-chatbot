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
    if re.search(r"\b(figures?|diagrams?|images?|visuals?|architecture|flowcharts?|illustrations?|show me|display)\b", lower):
        kinds.add("figure")
    if re.search(r"\b(formulas?|equations?|mathematical|math|derive|scaled dot|softmax|q\s*[,/]\s*k)\b", lower):
        kinds.add("formula")
    return {
        "tasks": [{"query": question.strip(), "content_types": [kind]}
                  for kind in ("text", "table", "figure", "formula") if kind in kinds],
        "requested_types": sorted(kinds),
        "planner": "heuristic",
    }


def _normalise_plan(payload: dict, question: str) -> dict:
    """Expand multi-type plans into independent, modality-specific searches."""
    tasks = []
    seen = set()
    for item in payload.get("tasks", [])[:8]:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or question).strip()[:1000]
        kinds = [str(kind).lower() for kind in item.get("content_types", [])
                 if str(kind).lower() in VALID_TYPES]
        for kind in kinds:
            key = (query, kind)
            if query and key not in seen:
                tasks.append({"query": query, "content_types": [kind]})
                seen.add(key)
    if not tasks:
        return heuristic_plan(question)
    if not any(task["content_types"] == ["text"] for task in tasks):
        tasks.insert(0, {"query": question.strip(), "content_types": ["text"]})
    tasks = tasks[:4]
    requested = []
    for task in tasks:
        kind = task["content_types"][0]
        if kind not in requested:
            requested.append(kind)
    return {"tasks": tasks, "requested_types": requested, "planner": "llm"}

def plan_query(question: str) -> dict:
    """Use an LLM JSON plan when configured; always fall back to deterministic routing."""
    fallback = heuristic_plan(question)
    if not (AGENT_PLANNER_ENABLED and LLM_API_KEY and LLM_BASE_URL and LLM_MODEL):
        fallback["planner_reason"] = "Heuristic routing used because planner is disabled or OpenAI configuration is incomplete."
        return fallback
    system_prompt = (
        "You are a query planner for a research-paper chatbot. Do not answer the question. "
        "Return JSON only: {tasks:[{query:string,content_types:[one type only]}]}. "
        "Always include text. Add table for requested numeric/table evidence, figure for requested "
        "diagram/visual/architecture and formula for requested mathematical expressions. "
        "Create a separate task for each requested type; never group multiple types in one task. " +
        "Use at most four focused tasks. Include text and add table for numeric/table evidence, " +
        "figure for diagrams/architecture/visual explanation, and formula for equations or mathematical notation. Do not invent facts."
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
        plan = _normalise_plan(json.loads(raw), question)
        plan["planner_reason"] = "OpenAI query planner returned a validated modality plan."
        return plan
    except Exception as exc:
        fallback["planner_reason"] = f"OpenAI query planner failed ({type(exc).__name__}); deterministic routing used."
        return fallback
