"""nodes for finalizing the evaluation"""
import os
import asyncio
from typing import Any, Dict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client

from agents.eval_agent.constants import CODEX_REMOTE_URL, LANGSMITH_CLIENT
from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from shared.schema import (
    CodexInput,
    CodexOutput,
)


logger = get_logger("eval_agent.finalize")


async def _handoff_to_codex(state: EvalAgentState) -> dict:
    if not state.context.github_context or not state.context.user_context:
        raise RuntimeError("GitHub and user context are required for Codex handoff")
    if not state.dataset_context or not state.active_experiment:
        raise RuntimeError("Dataset and experiment context are required for Codex handoff")
    
    is_any_eval_failed = False
    for result in state.active_experiment.results:
        if not result.passed:
            is_any_eval_failed = True
            break
    
    if not is_any_eval_failed:
        logger.warning("No eval failed, skipping Codex handoff")
        return {"codex_output": None}

    codex_input = CodexInput(
        context=state.context,
        dataset_context=state.dataset_context.model_copy(deep=True),
        experiment_context=state.active_experiment.model_copy(deep=True),
        dataset_examples=list(state.dataset_examples or []),
    )

    codex_input_payload: Dict[str, Any] = codex_input.model_dump()

    codex_request = state.messages[0].content
    codex_payload: Dict[str, Any] = {
        "request": codex_request,
        "repo_url": state.context.github_context.repo_url,
        "branch_name": state.context.github_context.branch_name,
    }
    codex_payload.update(codex_input_payload)

    logger.info("Codex payload: %s", codex_payload)

    # create a new thread for the Codex agent
    codex_sync_client = get_sync_client(url=CODEX_REMOTE_URL)
    thread = await asyncio.to_thread(codex_sync_client.threads.create)

    codex_thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
    codex_remote = RemoteGraph(
        "codex",
        url=CODEX_REMOTE_URL,
        client=LANGSMITH_CLIENT,
        sync_client=codex_sync_client,
        distributed_tracing=True,
    )

    codex_response = await asyncio.to_thread(
        codex_remote.invoke,
        codex_payload,
        codex_thread_cfg,
    )

    codex_response_payload: CodexOutput = CodexOutput.model_validate(codex_response)
    logger.info("Codex response payload: %s", codex_response_payload)

    # if the target agent was updated, store the handoff and reset the state for a new round
    if codex_response_payload.target_agent_version > state.context.target_agent_version:
        # Pass the handoff object and reset the loop state
        return {
            "codex_output": codex_response_payload,
            "attempts": 0,
            "dataset_examples": [],
            "latest_results": [],
            # ADDED: Clear MCP resources for the next round
            "mcp_resources": {}, 
        }
    else:
        # Agent was not updated, clear any potential stale handoff object
        return {
            "codex_output": None,
            "mcp_resources": {}, # ADDED: Clear resources even if no update
        }


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
