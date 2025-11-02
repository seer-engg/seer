import json
import uuid
from typing import Any, Dict, List, Optional

from agents.eval_agent.deps import LANGGRAPH_CLIENT, LANGGRAPH_SYNC_CLIENT, logger
from agents.eval_agent.models import EvalReflection


async def search_eval_reflections(agent_name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Semantic search past eval reflections from the LangGraph store index."""

    results = await LANGGRAPH_CLIENT.store.search_items(
        ("eval_reflections", agent_name),
        query=query,
        limit=limit,
    )
    return list(results)


def _parse_reflection_item(item: Dict[str, Any]) -> Optional[EvalReflection]:
    try:
        value = item.get("value") if isinstance(item, dict) else item
        if not isinstance(value, dict):
            return None
        return EvalReflection.model_validate(value)
    except Exception:
        logger.debug("reflection_store: unable to parse reflection payload", exc_info=True)
        return None


async def load_recent_reflections(agent_name: str, expectations: str, limit: int = 5) -> List[EvalReflection]:
    prior_results = await search_eval_reflections(agent_name, expectations, limit=limit)
    reflections: List[EvalReflection] = []
    for item in prior_results:
        reflection = _parse_reflection_item(item)
        if reflection is not None:
            reflections.append(reflection)
    return reflections


def format_reflections_for_prompt(reflections: List[EvalReflection], limit: int = 5) -> str:
    if not reflections:
        return "[]"

    trimmed = reflections[:limit]
    digest: List[Dict[str, Any]] = []
    for reflection in trimmed:
        digest.append(
            {
                "summary": reflection.summary,
                "failure_modes": reflection.failure_modes,
                "recommended_tests": reflection.recommended_tests,
                "latest_score": reflection.latest_score,
                "attempt": reflection.attempt,
            }
        )
    return json.dumps(digest, indent=2)


def format_previous_inputs(prev_inputs: List[str], limit: int = 10) -> str:
    if not prev_inputs:
        return "[]"
    tail = prev_inputs[-limit:]
    return json.dumps(tail, indent=2)


def persist_reflection(agent_name: str, reflection: EvalReflection) -> None:
    """Store the reflection synchronously for future retrieval."""

    try:
        key = uuid.uuid4().hex
        LANGGRAPH_SYNC_CLIENT.store.put_item(
            ("eval_reflections", agent_name),
            key=key,
            value=reflection.model_dump(),
        )
        logger.info("reflection_store: stored eval reflection")
    except Exception:
        logger.warning("reflection_store: failed to store eval reflection")


