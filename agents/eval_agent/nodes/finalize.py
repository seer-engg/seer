"""nodes for finalizing the evaluation"""
import os
import asyncio
from typing import Any, Dict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agents.eval_agent.constants import LANGFUSE_CLIENT
from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from shared.schema import (
    CodexInput,
    CodexOutput,
)
from shared.config import config

logger = get_logger("eval_agent.finalize")


async def _handoff_to_codex(state: EvalAgentState) -> dict:
    # Runtime check: skip handoff if disabled (even if node was added to graph)
    # Check both config and env var directly (env var takes precedence)
    env_var = os.environ.get("CODEX_HANDOFF_ENABLED", "").lower()
    codex_disabled = (
        not config.codex_handoff_enabled 
        or env_var in ("false", "0", "no", "off")
    )
    
    if codex_disabled:
        logger.info(
            f"⏭️  Codex handoff disabled at runtime - skipping handoff "
            f"(config={config.codex_handoff_enabled}, env_var={env_var})"
        )
        return {"codex_output": None}
    
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
    codex_sync_client = get_sync_client(url=config.codex_remote_url)
    thread = await asyncio.to_thread(codex_sync_client.threads.create)

    codex_thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
    
    # Generate deterministic trace ID from thread ID for distributed tracing
    trace_id = None
    langfuse_handler = None
    if LANGFUSE_CLIENT:
        trace_id = Langfuse.create_trace_id(seed=thread["thread_id"])
        # Initialize CallbackHandler with public_key (required for proper tracing)
        langfuse_handler = CallbackHandler(
            public_key=config.langfuse_public_key
        ) if config.langfuse_public_key else CallbackHandler()
        # Add trace context and metadata to config
        codex_thread_cfg["metadata"] = codex_thread_cfg.get("metadata", {})
        codex_thread_cfg["metadata"]["langfuse_trace_id"] = trace_id
        codex_thread_cfg["metadata"]["project_name"] = config.project_name  # Add project_name for filtering
    
    codex_remote = RemoteGraph(
        "codex",
        url=config.codex_remote_url,
        sync_client=codex_sync_client,
    )

    # Wrap invocation with Langfuse trace context if available
    if LANGFUSE_CLIENT and trace_id:
        with LANGFUSE_CLIENT.start_as_current_observation(
            as_type="span",
            name="codex-remote-invocation",
            trace_context={"trace_id": trace_id}
        ) as span:
            span.update_trace(input=codex_payload)
            # Pass metadata via config to ensure it's attached to the root trace
            invoke_config = {**codex_thread_cfg}
            if langfuse_handler:
                invoke_config["callbacks"] = [langfuse_handler]
            # Metadata is already in codex_thread_cfg["metadata"] from above
            codex_response = await asyncio.to_thread(
                codex_remote.invoke,
                codex_payload,
                invoke_config,
            )
            span.update_trace(output=codex_response)
    else:
        codex_response = await asyncio.to_thread(
            codex_remote.invoke,
            codex_payload,
            codex_thread_cfg,
        )

    codex_response: CodexOutput = CodexOutput.model_validate(codex_response)
    logger.info("Codex response: %s", codex_response)

    # if the target agent was updated, store the handoff and reset the state for a new round
    if codex_response.updated_context.target_agent_version > state.context.target_agent_version:
        # Pass the handoff object and reset the loop state
        return {
            "codex_output": codex_response,
            "attempts": 0,
            "dataset_examples": [],
            "latest_results": [],
            # Clear MCP resources for the next round via context
            "context": state.context.model_copy(update={"mcp_resources": {}}),
        }
    else:
        # Agent was not updated, clear any potential stale handoff object
        return {
            "codex_output": None,
            # Clear resources even if no update via context
            "context": state.context.model_copy(update={"mcp_resources": {}}),
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
    
    # Strategic logging: Log config and decision
    codex_enabled = config.codex_handoff_enabled
    logger.info(
        f"🔧 Building finalize subgraph: CODEX_HANDOFF_ENABLED={codex_enabled}, "
        f"env_var={os.environ.get('CODEX_HANDOFF_ENABLED', 'not_set')}"
    )
    
    if codex_enabled:
        logger.info("📤 Codex handoff ENABLED - will attempt handoff after evaluation")
        builder.add_node("handoff", _handoff_to_codex)
        builder.add_edge(START, "handoff")
        builder.add_edge("handoff", "summarize")
    else:
        logger.info("⏭️  Codex handoff DISABLED - skipping handoff, summarizing results only")
        # just summarize the results
        builder.add_edge(START, "summarize")
    
    return builder.compile()
