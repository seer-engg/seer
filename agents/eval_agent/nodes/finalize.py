import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client

from agents.codex.common.state import GithubContext, SandboxContext, TestingContext, UserContext
from agents.eval_agent.deps import (
    CODEX_REMOTE_URL,
    LANGSMITH_CLIENT,
    MAX_ATTEMPTS,
    MIN_ATTEMPTS,
    PASS_THRESHOLD,
    logger,
)
from agents.eval_agent.models import DeploymentContext, EvalAgentState, RunContext, TargetAgentConfig, TestResult


def should_continue(state: EvalAgentState) -> Literal["reflect", "finalize"]:
    """Decide whether the eval loop should continue reflecting or finalize."""

    run_ctx = state.run or RunContext()

    if run_ctx.attempts < MIN_ATTEMPTS:
        return "reflect"
    if run_ctx.score >= PASS_THRESHOLD or run_ctx.attempts >= MAX_ATTEMPTS:
        return "finalize"
    return "reflect"


def _build_codex_handoff_message(
    cfg: TargetAgentConfig,
    deployment_url: str,
    expectations: str,
    dataset_name: str,
    experiment_name: str,
    total_tests: int,
    passed_count: int,
    failed_cases: List[Dict[str, Any]],
    latest_score: float,
    aggregate_score: float,
) -> str:
    def _truncate(text: str, limit: int = 400) -> str:
        if text is None:
            return ""
        text = str(text)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}… (truncated {len(text)} chars)"

    lines: List[str] = []
    lines.append("Seer Eval Agent → Codex Handoff")
    lines.append("")
    lines.append("## Agent Deployment")
    lines.append(f"- Graph Name: {cfg.graph_name}")
    lines.append(f"- Repository: {cfg.repo_url} (branch: {cfg.branch_name})")
    lines.append(f"- Deployment URL: {deployment_url}")
    if dataset_name:
        lines.append(f"- Dataset: {dataset_name}")
    if experiment_name:
        lines.append(f"- Experiment: {experiment_name}")

    lines.append("")
    lines.append("## User Expectations")
    lines.append(expectations.strip() or "(none provided)")

    lines.append("")
    lines.append("## Evaluation Summary")
    lines.append(f"- Total tests: {total_tests}")
    lines.append(f"- Passed: {passed_count}")
    lines.append(f"- Failed: {len(failed_cases)}")
    lines.append(f"- Latest mean correctness: {latest_score:.2f}")
    lines.append(f"- Aggregate correctness: {aggregate_score:.2f}")

    lines.append("")
    lines.append("## Failed Test Details")
    if not failed_cases:
        lines.append("All tests passed. No remediation required.")
    else:
        for idx, case in enumerate(failed_cases, start=1):
            expectation_ref = case.get("expectation_ref") or f"Test {idx}"
            lines.append(f"{idx}. Expectation: {expectation_ref}")
            lines.append(f"   - Input: {_truncate(case.get('input'))}")
            lines.append(f"   - Expected: {_truncate(case.get('expected_output'))}")
            lines.append(f"   - Actual: {_truncate(case.get('actual_output')) or '(empty)'}")
            lines.append(f"   - Success Criteria: {_truncate(case.get('success_criteria'), 300)}")
            lines.append(f"   - Score: {case.get('score', 0.0):.2f}")
            judge_comment = case.get("judge_comment") or "(no comment)"
            lines.append(f"   - Judge Comment: {_truncate(judge_comment, 400)}")

    lines.append("")
    lines.append("## Requested Outcome")
    lines.append(
        "Investigate the repository, reproduce the failing evaluations, and propose code changes "
        "that satisfy the expectations while keeping passing behavior intact."
    )

    return "\n".join(lines)


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
    payload = {
        "attempts": run_ctx.attempts,
        "score": run_ctx.score,
        "dataset_name": run_ctx.dataset_name,
        "experiment_name": run_ctx.experiment_name,
    }
    score_history = list(run_ctx.score_history or [])
    attempts = len(score_history)
    average_score = (sum(score_history) / attempts) if attempts else run_ctx.score
    latest_score = score_history[-1] if attempts else run_ctx.score

    deployment = state.deployment or DeploymentContext()
    deployment_url = deployment.url
    deployment_metadata = dict(deployment.metadata)
    codex_thread_id = state.codex_thread_id
    failed_cases = list(run_ctx.accumulated_failed_cases or run_ctx.last_failed_cases or [])
    structured_test_results: List[TestResult] = []
    for idx, case in enumerate(failed_cases, start=1):
        expectation_ref = case.get("expectation_ref") or f"test-{idx}"
        structured_test_results.append(
            TestResult(
                test_case_id=str(expectation_ref),
                input_sent=str(case.get("input", "")),
                actual_output=str(case.get("actual_output", "")),
                expected_behavior=str(case.get("expected_output") or case.get("expected_behavior", "")),
                passed=False,
                score=float(case.get("score", 0.0)),
                judge_reasoning=str(case.get("judge_comment", "")),
            )
        )

    metadata = dict(run_ctx.accumulated_metadata or run_ctx.last_metadata or {})
    failed_cases_count = len(failed_cases)
    codex_status = "skipped (no failing tests)" if failed_cases_count == 0 else "pending"

    cfg = state.target_agent_config
    codex_request_payload = state.codex_request
    codex_response_payload = state.codex_response

    resolved_repo_url = (cfg.repo_url if cfg else None) or deployment.repo_url or deployment_metadata.get("repo_url")
    github_context = GithubContext(repo_url=resolved_repo_url) if resolved_repo_url else None

    resolved_sandbox_id = deployment.sandbox_id or deployment_metadata.get("sandbox_id")
    resolved_sandbox_dir = deployment.sandbox_repo_dir or deployment_metadata.get("sandbox_repo_dir")
    resolved_sandbox_branch = deployment.sandbox_branch or deployment_metadata.get("sandbox_branch") or deployment_metadata.get("branch_name")
    sandbox_context = (
        SandboxContext(
            sandbox_session_id=resolved_sandbox_id,
            working_directory=resolved_sandbox_dir,
            working_branch=resolved_sandbox_branch,
        )
        if resolved_sandbox_id
        else None
    )

    user_expectations = cfg.expectations if cfg else None
    user_context = UserContext(user_expectation=user_expectations) if user_expectations else None

    testing_context = TestingContext(test_results=structured_test_results)

    deployment_metadata.update(
        {
            "deployment_url": deployment_url,
            "repo_url": resolved_repo_url,
            "sandbox_id": resolved_sandbox_id,
            "sandbox_repo_dir": resolved_sandbox_dir,
            "sandbox_branch": resolved_sandbox_branch,
        }
    )

    finalize_context = {
        "payload": payload,
        "score_history": score_history,
        "attempts": attempts,
        "average_score": average_score,
        "latest_score": latest_score,
        "deployment_url": deployment_url,
        "deployment_metadata": deployment_metadata,
        "codex_thread_id": codex_thread_id,
        "failed_cases": failed_cases,
        "failed_cases_count": failed_cases_count,
        "metadata": metadata,
        "codex_status": codex_status,
        "codex_request_payload": codex_request_payload,
        "codex_response_payload": codex_response_payload,
        "github_context": github_context,
        "sandbox_context": sandbox_context,
        "user_context": user_context,
        "testing_context": testing_context,
        "handoff_required": failed_cases_count > 0 and cfg is not None,
    }

    updated_deployment = deployment.model_copy(
        update={
            "url": deployment_url,
            "repo_url": resolved_repo_url or deployment.repo_url,
            "sandbox_id": resolved_sandbox_id,
            "sandbox_repo_dir": resolved_sandbox_dir,
            "sandbox_branch": resolved_sandbox_branch,
            "metadata": deployment_metadata,
        }
    )

    return {
        "finalize_context": finalize_context,
        "deployment": updated_deployment,
        "codex_thread_id": codex_thread_id,
        "codex_request": codex_request_payload,
        "codex_response": codex_response_payload,
        "codex_followup_branch": state.codex_followup_branch,
        "codex_followup_metadata": state.codex_followup_metadata,
        "pending_followup": state.pending_followup,
    }


async def _handoff_to_codex(state: EvalAgentState) -> dict:
    context = dict(getattr(state, "finalize_context", {}) or {})
    if not context:
        return {}

    cfg = state.target_agent_config
    deployment = state.deployment or DeploymentContext()
    run_ctx = state.run or RunContext()
    deployment_metadata = dict(deployment.metadata)
    metadata = context.get("metadata", {})
    failed_cases = list(context.get("failed_cases", []))
    codex_status = context.get("codex_status", "pending")
    codex_thread_id = state.codex_thread_id
    codex_request_payload = context.get("codex_request_payload", state.codex_request)
    codex_response_payload = context.get("codex_response_payload", state.codex_response)
    testing_context: TestingContext = context.get("testing_context", TestingContext())
    github_context: Optional[GithubContext] = context.get("github_context")
    sandbox_context: Optional[SandboxContext] = context.get("sandbox_context")
    user_context: Optional[UserContext] = context.get("user_context")
    deployment_url = context.get("deployment_url", deployment.url)

    pending_followup = state.pending_followup
    codex_followup_branch = state.codex_followup_branch
    codex_followup_metadata = dict(getattr(state, "codex_followup_metadata", {}) or {})

    if context.get("handoff_required") and cfg is not None:
        dataset_name = metadata.get("dataset_name", run_ctx.dataset_name)
        experiment_name = metadata.get("experiment_name", run_ctx.experiment_name)
        total_tests = metadata.get("total_tests", len(state.test_cases))
        passed_tests = metadata.get("passed_tests", total_tests - len(failed_cases))
        latest_score_value = metadata.get("latest_score", run_ctx.score)
        aggregate_score_value = metadata.get("aggregate_score", context.get("average_score", run_ctx.score))
        deployment_url = metadata.get("target_url", deployment_url or cfg.url)

        codex_message = _build_codex_handoff_message(
            cfg=cfg,
            deployment_url=deployment_url or "",
            expectations=cfg.expectations,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            total_tests=total_tests,
            passed_count=passed_tests,
            failed_cases=failed_cases,
            latest_score=latest_score_value,
            aggregate_score=aggregate_score_value,
        )

        handoff_timestamp = datetime.now(timezone.utc)
        context_payload: Dict[str, Any] = {}
        if github_context:
            context_payload["github_context"] = github_context.model_dump()
        if sandbox_context:
            context_payload["sandbox_context"] = sandbox_context.model_dump()
        if user_context:
            context_payload["user_context"] = user_context.model_dump()
        context_payload["testing_context"] = testing_context.model_dump()

        codex_request_payload = {
            "message": codex_message,
            "graph_name": cfg.graph_name,
            "handoff_at": handoff_timestamp.isoformat(),
            **context_payload,
        }
        if deployment_url:
            codex_request_payload["deployment_url"] = deployment_url

        planner_request = (
            f"Address failing evaluations for agent '{cfg.graph_name}'. "
            "See the attached summary for context and required fixes."
        )
        planner_payload: Dict[str, Any] = {
            "request": planner_request,
            "repo_url": cfg.repo_url,
            "branch_name": cfg.branch_name,
            "setup_script": cfg.setup_script,
            "messages": [{"role": "user", "content": codex_message}],
            "deployment_url": deployment_url,
            "dataset_name": dataset_name,
            "experiment_name": experiment_name,
        }
        planner_payload.update(context_payload)
        repo_path = deployment.sandbox_repo_dir or deployment_metadata.get("sandbox_repo_dir")
        if repo_path:
            planner_payload["repo_path"] = repo_path
        metadata_repo_path = metadata.get("repo_path")
        if metadata_repo_path and "repo_path" not in planner_payload:
            planner_payload["repo_path"] = metadata_repo_path
        if "repo_path" not in planner_payload:
            planner_payload["repo_path"] = "/tmp/codex"
        if deployment.sandbox_id:
            planner_payload["sandbox_session_id"] = deployment.sandbox_id

        logger.info("Planner payload: %s", planner_payload)

        try:
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

            codex_response_payload = {
                "thread_id": codex_thread_id,
                "response": codex_response,
                "planner_request": planner_payload,
            }
            branch_name, branch_meta = _extract_branch_from_codex_response(codex_response)
            if branch_name:
                codex_followup_branch = branch_name
                codex_followup_metadata.update(branch_meta)
                pending_followup = True
                deployment_metadata["codex_last_handoff_at"] = handoff_timestamp.isoformat()
                deployment_metadata["codex_branch"] = branch_name
                failed_cases = []
                metadata.clear()
                codex_status = "ok"
            else:
                codex_status = "error: Codex response missing branch_name"
                logger.error("finalize.handoff: Codex response missing branch_name: %s", codex_response)
        except Exception as codex_error:
            logger.error("finalize.handoff: failed to hand off to Codex: %s", codex_error, exc_info=True)
            codex_status = f"error: {codex_error}"

    context.update(
        {
            "codex_status": codex_status,
            "failed_cases": failed_cases,
            "metadata": metadata,
            "codex_thread_id": codex_thread_id,
            "codex_request_payload": codex_request_payload,
            "codex_response_payload": codex_response_payload,
            "deployment_url": deployment_url,
        }
    )

    deployment_metadata["deployment_url"] = deployment_url

    updated_deployment = deployment.model_copy(
        update={
            "url": deployment_url,
            "metadata": deployment_metadata,
        }
    )

    run_updates = {
        "accumulated_failed_cases": context.get("failed_cases", failed_cases),
        "accumulated_metadata": metadata,
    }

    updated_run = run_ctx.model_copy(update=run_updates)

    return {
        "finalize_context": context,
        "deployment": updated_deployment,
        "codex_thread_id": codex_thread_id,
        "codex_request": codex_request_payload,
        "codex_response": codex_response_payload,
        "pending_followup": pending_followup,
        "codex_followup_branch": codex_followup_branch,
        "codex_followup_metadata": codex_followup_metadata,
        "run": updated_run,
    }


def _summarize_finalize(state: EvalAgentState) -> dict:
    context = dict(getattr(state, "finalize_context", {}) or {})
    run_ctx = state.run or RunContext()
    payload = context.get("payload", {})
    attempts = context.get("attempts", len(run_ctx.score_history or []))
    latest_score = context.get("latest_score", run_ctx.score)
    average_score = context.get("average_score", run_ctx.score)
    deployment = state.deployment or DeploymentContext()
    deployment_url = context.get("deployment_url", deployment.url)
    deployment_metadata = dict(deployment.metadata)
    codex_thread_id = state.codex_thread_id
    codex_status = context.get("codex_status", "skipped (no failing tests)")
    failed_cases = list(context.get("failed_cases", []))
    metadata = context.get("metadata", {})
    codex_request_payload = context.get("codex_request_payload", state.codex_request)
    codex_response_payload = context.get("codex_response_payload", state.codex_response)

    logger.info(
        "finalize_node: attempts=%d latest_score=%.3f average_score=%.3f payload=%s",
        attempts,
        latest_score,
        average_score,
        payload,
    )

    user_summary = (
        f"Final evaluation complete: attempts={max(attempts, run_ctx.attempts)}; "
        f"average score={average_score:.2f} (0–1), latest={latest_score:.2f}. "
        f"Dataset=`{run_ctx.dataset_name}`, Experiment=`{run_ctx.experiment_name}`."
    )
    if deployment_url:
        user_summary += f" Deployment URL: {deployment_url}."
    user_summary += f" Codex handoff status: {codex_status}."
    if codex_thread_id:
        user_summary += f" Codex thread ID: {codex_thread_id}."
    if failed_cases:
        user_summary += f" Escalated failing tests: {len(failed_cases)}."

    pending_followup = state.pending_followup
    codex_branch = state.codex_followup_branch
    codex_followup_metadata = dict(getattr(state, "codex_followup_metadata", {}) or {})
    if codex_branch:
        user_summary += f" Codex branch: {codex_branch}."
        if pending_followup:
            user_summary += " Follow-up evaluation scheduled."

    deployment_metadata["codex_handoff_status"] = codex_status
    if codex_branch:
        deployment_metadata["branch_name"] = codex_branch
        deployment_metadata["codex_branch"] = codex_branch
        if codex_followup_metadata:
            deployment_metadata["codex_branch_metadata"] = codex_followup_metadata

    deployment_metadata["deployment_url"] = deployment_url

    next_state: Dict[str, Any] = {
        "messages": [AIMessage(content=user_summary)],
        "codex_thread_id": codex_thread_id,
        "codex_request": codex_request_payload,
        "codex_response": codex_response_payload,
        "pending_followup": pending_followup,
        "codex_followup_branch": codex_branch,
        "codex_followup_metadata": codex_followup_metadata,
        "finalize_context": {},
    }

    deployment_updates: Dict[str, Any] = {
        "metadata": deployment_metadata,
        "url": deployment_url,
    }

    run_updates: Dict[str, Any] = {}

    if pending_followup:
        deployment_updates.update(
            {
                "sandbox_id": None,
                "sandbox_repo_dir": None,
                "sandbox_branch": codex_branch or deployment.sandbox_branch,
                "url": None,
            }
        )
        run_updates["current_thread_id"] = None

    updated_deployment = deployment.model_copy(update=deployment_updates)
    next_state["deployment"] = updated_deployment

    if run_updates:
        next_state["run"] = run_ctx.model_copy(update=run_updates)

    return next_state


def build_finalize_subgraph():
    builder = StateGraph(EvalAgentState)
    builder.add_node("prepare", _prepare_finalize_context)
    builder.add_node("handoff", _handoff_to_codex)
    builder.add_node("summarize", _summarize_finalize)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "handoff")
    builder.add_edge("handoff", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()


def finalize_router(state: EvalAgentState) -> Literal["followup", "end"]:
    return "followup" if state.pending_followup else "end"



