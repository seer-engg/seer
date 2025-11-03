import os
import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client

from shared.schema import GithubContext, SandboxContext, TestingContext, UserContext, TestResult, CodexInput, CodexOutput
from agents.eval_agent.constants import (
    CODEX_REMOTE_URL,
    LANGSMITH_CLIENT,
    logger,
)
from agents.eval_agent.models import EvalAgentState, RunContext


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


def _prepare_finalize_context(state: EvalAgentState) -> dict:
    run_ctx = state.run or RunContext()

    codex_thread_id = state.codex_thread_id
    failed_cases = list(run_ctx.last_failed_cases or [])
    structured_test_results: List[TestResult] = []
    for case in failed_cases:
        structured_test_results.append(
            TestResult(
                input_sent=str(case.get("input", "")),
                actual_output=str(case.get("actual_output", "")),
                expected_behavior=str(case.get("expected_output") or case.get("expected_behavior", "")),
                passed=False,
                score=float(case.get("score", 0.0)),
                judge_reasoning=str(case.get("judge_comment", "")),
            )
        )

    codex_response_payload = state.codex_response

    testing_context = TestingContext(test_results=structured_test_results)
    testing_context.test_cases = state.test_cases

    return {
        "testing_context": testing_context,
        "codex_thread_id": codex_thread_id,
        "codex_response": codex_response_payload,
        "codex_followup_branch": state.codex_followup_branch,
        "pending_followup": state.pending_followup,
    }


async def _handoff_to_codex(state: EvalAgentState) -> dict:
    codex_thread_id = state.codex_thread_id
    testing_context = state.testing_context or TestingContext()
    github_context = state.github_context or GithubContext(repo_url="")
    sandbox_context = state.sandbox_context or SandboxContext(sandbox_id="", working_directory=None, working_branch=None)
    user_context = state.user_context or UserContext(user_expectation="")

    # Prepare typed CodexInput and default CodexOutput for downstream state
    github_ctx_for_input = github_context or GithubContext(repo_url="")
    sandbox_ctx_for_input = sandbox_context or SandboxContext(
        sandbox_id="",
        working_directory=None,
        working_branch=None,
    )
    user_ctx_for_input = user_context or UserContext(user_expectation="")
    testing_ctx_for_input = testing_context or TestingContext()
    codex_input = CodexInput(
        github_context=github_ctx_for_input,
        sandbox_context=sandbox_ctx_for_input,
        user_context=user_ctx_for_input,
        testing_context=testing_ctx_for_input,
    )
    codex_output: CodexOutput = CodexOutput(agent_updated=False, new_branch_name=None)

    pending_followup = state.pending_followup
    codex_followup_branch = state.codex_followup_branch

    codex_input_payload: Dict[str, Any] = codex_input.model_dump()

    planner_request = (
        f"Address failing evaluations for agent '{github_context.repo_url}'. "
        "See the attached summary for context and required fixes."
    )
    planner_payload: Dict[str, Any] = {
        "request": planner_request,
        "repo_url": github_context.repo_url,
        "branch_name": github_context.branch_name,
    }
    planner_payload.update(codex_input_payload)

    logger.info("Planner payload: %s", planner_payload)

    codex_sync_client = get_sync_client(url=CODEX_REMOTE_URL)
    if not codex_thread_id:
        thread = await asyncio.to_thread(codex_sync_client.threads.create)
        codex_thread_id = thread["thread_id"]

    codex_thread_cfg = {"configurable": {"thread_id": codex_thread_id}}
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

    branch_name = _extract_branch_from_codex_response(codex_response)
    if branch_name:
        codex_followup_branch = branch_name
        pending_followup = True
        codex_output = CodexOutput(agent_updated=True, new_branch_name=branch_name)
    else:
        logger.error("finalize.handoff: Codex response missing branch_name: %s", codex_response)

    return {
        "codex_thread_id": codex_thread_id,
        "codex_request": codex_input.model_dump(),
        "codex_response": codex_output.model_dump(),
        "pending_followup": pending_followup,
        "codex_followup_branch": codex_followup_branch,
    }


def _summarize_finalize(state: EvalAgentState) -> dict:
    run_ctx = state.run or RunContext()
    attempts = len(run_ctx.score_history or [])
    latest_score = run_ctx.score
    average_score = (sum(run_ctx.score_history or []) / attempts) if attempts else run_ctx.score
    codex_thread_id = state.codex_thread_id
    failed_cases = list(run_ctx.last_failed_cases or [])
    codex_response_payload = state.codex_response

    logger.info(
        "finalize_node: attempts=%d latest_score=%.3f average_score=%.3f",
        attempts,
        latest_score,
        average_score,
    )

    user_summary = (
        f"Final evaluation complete: attempts={max(attempts, run_ctx.attempts)}; "
        f"average score={average_score:.2f} (0–1), latest={latest_score:.2f}. "
        f"Dataset=`{run_ctx.dataset_name}`, Experiment=`{run_ctx.experiment_name}`."
    )
    if codex_thread_id:
        user_summary += f" Codex thread ID: {codex_thread_id}."
    if failed_cases:
        user_summary += f" Escalated failing tests: {len(failed_cases)}."

    pending_followup = state.pending_followup
    codex_branch = state.codex_followup_branch
    if codex_branch:
        user_summary += f" Codex branch: {codex_branch}."
        if pending_followup:
            user_summary += " Follow-up evaluation scheduled."

    codex_response_value = (
        codex_response_payload.model_dump() if hasattr(codex_response_payload, "model_dump") else codex_response_payload
    )

    next_state: Dict[str, Any] = {
        "messages": [AIMessage(content=user_summary)],
        "codex_thread_id": codex_thread_id,
        "codex_response": codex_response_value,
        "pending_followup": pending_followup,
        "codex_followup_branch": codex_branch,
    }

    run_updates: Dict[str, Any] = {}

    if pending_followup:
        run_updates["current_thread_id"] = None

    if run_updates:
        next_state["run"] = run_ctx.model_copy(update=run_updates)

    return next_state


def build_finalize_subgraph():
    """Build the finalize subgraph."""
    builder = StateGraph(EvalAgentState)
    builder.add_node("summarize", _summarize_finalize)
    builder.add_edge("summarize", END)
    
    if os.getenv("CODEX_HANDOFF_ENABLED") == "true":
        builder.add_node("prepare", _prepare_finalize_context)
        builder.add_node("handoff", _handoff_to_codex)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "handoff")
        builder.add_edge("handoff", "summarize")
    else:
        # just summarize the results
        builder.add_edge(START, "summarize")
    
    return builder.compile()
