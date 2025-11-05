"""nodes for finalizing the evaluation"""
import os
import asyncio
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client

from agents.eval_agent.constants import CODEX_REMOTE_URL, LANGSMITH_CLIENT
from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from shared.schema import SandboxContext, CodexInput


logger = get_logger("eval_agent.finalize")

def _extract_branch_from_codex_response(response: Any) -> tuple[Optional[str], Dict[str, Any]]:
    branch_name: Optional[str] = None
    branch_path: Optional[str] = None

    def _walk(obj: Any, path: str = "") -> None:
        nonlocal branch_name, branch_path
        if branch_name is not None:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and key in {"branch_name", "branch", "head_branch"} and value.strip():
                    branch_name = value.strip()
                    branch_path = current_path
                    return
                _walk(value, current_path)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                current_path = f"{path}[{idx}]" if path else f"[{idx}]"
                _walk(item, current_path)

    _walk(response)
    metadata = {"branch_path": branch_path} if branch_path else {}
    return branch_name, metadata


async def _handoff_to_codex(state: EvalAgentState) -> dict:
    if not state.github_context or not state.user_context:
        raise RuntimeError("GitHub and user context are required for Codex handoff")
    if not state.dataset_context or not state.active_experiment:
        raise RuntimeError("Dataset and experiment context are required for Codex handoff")

    github_context = state.github_context
    sandbox_context = state.sandbox_context or SandboxContext(
        sandbox_id="",
        working_directory="",
        working_branch="",
    )
    user_context = state.user_context

    codex_input = CodexInput(
        github_context=github_context,
        sandbox_context=sandbox_context,
        user_context=user_context,
        dataset_context=state.dataset_context.model_copy(deep=True),
        experiment_context=state.active_experiment.model_copy(deep=True),
        dataset_examples=list(state.dataset_examples or []),
        target_agent_version=state.target_agent_version,
    )

    codex_input_payload: Dict[str, Any] = codex_input.model_dump()

    planner_request = state.messages[0].content
    planner_payload: Dict[str, Any] = {
        "request": planner_request,
        "repo_url": github_context.repo_url,
        "branch_name": github_context.branch_name,
    }
    planner_payload.update(codex_input_payload)

    logger.info("Planner payload: %s", planner_payload)

    # create a new thread for the Codex agent
    codex_sync_client = get_sync_client(url=CODEX_REMOTE_URL)
    thread = await asyncio.to_thread(codex_sync_client.threads.create)

    codex_thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
    codex_remote = RemoteGraph(
        "planner",
        url=CODEX_REMOTE_URL,
        client=LANGSMITH_CLIENT,
        sync_client=codex_sync_client,
        distributed_tracing=True,
    )

    codex_response = await asyncio.to_thread(
        codex_remote.invoke,
        planner_payload,
        codex_thread_cfg,
    )

    branch_name, metadata = _extract_branch_from_codex_response(codex_response)
    if branch_name:
        logger.info("Codex handoff response metadata: %s", metadata)
    else:
        logger.error("finalize.handoff: Codex response missing branch_name: %s", codex_response)

    return {}


def _summarize_finalize(state: EvalAgentState) -> dict:
    experiment = state.active_experiment
    dataset = state.dataset_context
    latest_score = experiment.mean_score if experiment else 0.0
    failed_cases = list(experiment.failed_results) if experiment else []

    logger.info(
        "finalize_node: attempts=%d latest_score=%.3f",
        state.attempts,
        latest_score,
    )

    user_summary = (
        f"Final evaluation complete: attempts={state.attempts}; "
        f"latest score={latest_score:.2f}. "
        f"Dataset=`{dataset.dataset_name if dataset else ''}`, Experiment=`{experiment.experiment_name if experiment else ''}`."
    )
    if failed_cases:
        user_summary += f" Escalated failing tests: {len(failed_cases)}."

    next_state: Dict[str, Any] = {
        "messages": [AIMessage(content=user_summary)],
    }

    return next_state


def build_finalize_subgraph():
    """Build the finalize subgraph."""
    builder = StateGraph(EvalAgentState)
    builder.add_node("summarize", _summarize_finalize)
    builder.add_edge("summarize", END)
    
    if os.getenv("CODEX_HANDOFF_ENABLED") == "true":
        builder.add_node("handoff", _handoff_to_codex)
        builder.add_edge(START, "handoff")
        builder.add_edge("handoff", "summarize")
    else:
        # just summarize the results
        builder.add_edge(START, "summarize")
    
    return builder.compile()
