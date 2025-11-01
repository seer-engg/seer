import asyncio
import json
import os
import uuid
import requests
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_client, get_sync_client
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.types import EvaluatorResult
from e2b import AsyncSandbox

from agents.eval_agent.models import (
    AgentSpec,
    EvalReflection,
    EvalAgentState,
    GeneratedTestCase,
    GeneratedTests,
    TargetAgentConfig,
    TestResult,
)
from agents.codex.common.state import GithubContext, SandboxContext, UserContext, TestingContext
from agents.eval_agent.prompts import (
    EVAL_AGENT_SPEC_PROMPT,
    EVAL_AGENT_TEST_GEN_PROMPT,
)
from shared.logger import get_logger
from shared.llm import get_llm
from sandbox import TARGET_AGENT_COMMAND, TARGET_AGENT_PORT, initialize_e2b_sandbox, setup_project, deploy_server_and_confirm_ready

logger = get_logger("eval_agent")

_LANGGRAPH_CLIENT = get_client(url="http://127.0.0.1:8002")
_LANGGRAPH_SYNC_CLIENT = get_sync_client(url="http://127.0.0.1:8002")
_LANGSMITH_CLIENT = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
PASS_THRESHOLD = 0.99
MAX_ATTEMPTS = 3
MIN_ATTEMPTS = 2
_CODEX_REMOTE_URL = os.getenv("CODEX_REMOTE_URL", "http://127.0.0.1:8003")

# Use a slightly higher temperature for test generation to encourage diversity
_LLM = get_llm(temperature=0.2)
_CORRECTNESS_EVALUATOR = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:gpt-4.1-mini",
    feedback_key="correctness",
)


async def search_eval_reflections(agent_name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Semantic search past eval reflections from the LangGraph store index.

    Returns a list of items with at least `value` containing the stored reflection payload.
    """
    # Use SDK's search_items API (supports text/vector search when index configured)
    results = await _LANGGRAPH_CLIENT.store.search_items(
        ("eval_reflections", agent_name),
        query=query,
        limit=limit,
    )
    return list(results)


async def upsert_eval_reflection(agent_name: str, reflection: EvalReflection) -> bool:
    """Upsert a reflection document into the store for future retrieval."""
    key = uuid.uuid4().hex
    value = reflection.model_dump()
    await _LANGGRAPH_CLIENT.store.put_item(("eval_reflections", agent_name), key=key, value=value)
    return True




async def _ensure_target_agent_config(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is not None and cfg.repo_url:
        return {"target_agent_config": cfg}

    last_human = None
    for msg in reversed(state.messages or []):
        try:
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                last_human = msg
                break
        except Exception:
            continue
    if last_human is None:
        raise ValueError("Missing target_agent_config and no human message to extract from")

    instruction = (
        "Extract the following fields from the user's latest message about the target agent:\n"
        "- graph_name: the LangGraph graph name to evaluate (NOT the assistant/thread id).\n"
        "- repo_url: canonical Git URL (https preferred) for the agent's source repository.\n"
        "- expectations: verbatim expectations text describing desired behavior.\n"
        "- url: fully-qualified deployment URL if the agent is already running; leave empty/null otherwise.\n"
        "- branch_name: Git branch to evaluate (default to 'main' if unspecified).\n"
        "- setup_script: single-line shell command to prepare the project inside the sandbox before launch (default 'pip install -e .')."
    )

    extractor = _LLM.with_structured_output(TargetAgentConfig)
    cfg = await extractor.ainvoke(f"{instruction}\n\nUSER:\n{last_human.content}")
    return {"target_agent_config": cfg}


async def _provision_target_agent(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_provision_target_agent requires target_agent_config to be set")

    repo_url = (cfg.repo_url or "").strip()
    branch_name = (cfg.branch_name or "main").strip() or "main"
    setup_script = (cfg.setup_script or "pip install -e .").strip() or "pip install -e ."

    deployment_url = cfg.url or state.deployment_url
    sandbox_id = state.sandbox_id
    sandbox_repo_dir = state.sandbox_repo_dir
    sandbox_branch = state.sandbox_branch or branch_name
    deployment_metadata = dict(getattr(state, "deployment_metadata", {}) or {})

    if not deployment_url:
        if not repo_url:
            raise ValueError("TargetAgentConfig.repo_url is required when no deployment URL is provided")

        github_token = os.getenv("GITHUB_TOKEN")
        logger.info(
            "plan.provision: provisioning sandbox (repo=%s branch=%s existing_sandbox=%s)",
            repo_url,
            branch_name,
            sandbox_id or "<new>",
        )

        sbx, repo_dir, resolved_branch = await initialize_e2b_sandbox(
            repo_url=repo_url,
            branch_name=branch_name,
            github_token=github_token,
        )
        sandbox_repo_dir = repo_dir
        sandbox_branch = resolved_branch or branch_name
        sandbox_id = sbx.sandbox_id

        await setup_project(sandbox_id, repo_dir, setup_script)

        sandbox, _ = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND, sb=sbx, cwd=repo_dir)

        deployment_url = sandbox.get_host(TARGET_AGENT_PORT)
        if not deployment_url.startswith("http"):
            deployment_url = f"https://{deployment_url}"

        deployment_metadata.update(
            {
                "sandbox_id": sandbox_id,
                "sandbox_repo_dir": repo_dir,
                "sandbox_branch": sandbox_branch,
                "setup_script": setup_script,
                "last_boot_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        logger.info("plan.provision: sandbox ready at %s", deployment_url)
    else:
        deployment_metadata.setdefault("last_seen_at", datetime.now(timezone.utc).isoformat())
        logger.info("plan.provision: reusing deployment url %s", deployment_url)

    cfg = cfg.model_copy(update={"url": deployment_url, "branch_name": sandbox_branch, "setup_script": setup_script})

    deployment_metadata["deployment_url"] = deployment_url
    if repo_url:
        deployment_metadata["repo_url"] = repo_url
    deployment_metadata["branch_name"] = sandbox_branch

    return {
        "target_agent_config": cfg,
        "deployment_url": deployment_url,
        "sandbox_id": sandbox_id,
        "sandbox_repo_dir": sandbox_repo_dir,
        "sandbox_branch": sandbox_branch,
        "deployment_metadata": deployment_metadata,
    }


async def _generate_eval_plan(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_generate_eval_plan requires target_agent_config to be set")

    agent_name = cfg.graph_name
    expectations = cfg.expectations
    repo_url_hint = cfg.repo_url or "(missing repo URL)"
    branch_hint = cfg.branch_name or "main"
    deployment_url_hint = cfg.url or state.deployment_url or "Pending sandbox deployment"

    prior_results = await search_eval_reflections(agent_name, expectations, limit=5)
    prior_reflections = []
    for item in prior_results:
        try:
            value = item.get("value") if isinstance(item, dict) else item
            if isinstance(value, dict):
                prior_reflections.append(value.get("summary") or json.dumps(value))
        except Exception:
            continue
    reflections_text = "\n- ".join(prior_reflections) if prior_reflections else "(none)"

    spec_prompt = EVAL_AGENT_SPEC_PROMPT.format(
        expectations=expectations,
        agent_name=agent_name,
        agent_repo=repo_url_hint,
        agent_branch=branch_hint,
        deployment_url=deployment_url_hint,
    )
    spec_llm = _LLM.with_structured_output(AgentSpec)
    spec_obj: AgentSpec = await spec_llm.ainvoke(spec_prompt)

    prev_inputs = state.previous_inputs or []
    prev_inputs_text = "\n- ".join(prev_inputs) if prev_inputs else "(none)"

    augmented_prompt = EVAL_AGENT_TEST_GEN_PROMPT.format(
        spec_json=spec_obj,
        reflections_text=reflections_text,
        prev_inputs_text=prev_inputs_text,
    )

    generated_runnable = _LLM.with_structured_output(GeneratedTests)
    generated: GeneratedTests = await generated_runnable.ainvoke(augmented_prompt)
    # assert len(generated.test_cases) == 5, f"Generated {len(generated.test_cases)} test cases, expected 5"

    test_cases: list[GeneratedTestCase] = []
    for tc in generated.test_cases:
        test_cases.append(GeneratedTestCase(
            expectation_ref=tc.expectation_ref,
            input_message=tc.input_message,
            expected_behavior=tc.expected_behavior,
            success_criteria=tc.success_criteria,
            expected_output=getattr(tc, "expected_output", None) or "",
        ))

    new_prev = list(prev_inputs)
    new_prev.extend([tc.input_message for tc in test_cases])

    logger.info("plan.generate: produced %d tests (agent=%s)", len(test_cases), agent_name)
    return {
        "test_cases": test_cases,
        "previous_inputs": new_prev,
        "dataset_name": "",
        "experiment_name": "",
    }


def _get_plan_subgraph():
    builder = StateGraph(EvalAgentState)
    builder.add_node("ensure-config", _ensure_target_agent_config)
    builder.add_node("provision-target", _provision_target_agent)
    builder.add_node("generate-tests", _generate_eval_plan)

    builder.add_edge(START, "ensure-config")
    builder.add_edge("ensure-config", "provision-target")
    builder.add_edge("provision-target", "generate-tests")
    builder.add_edge("generate-tests", END)

    return builder.compile()


async def plan_node(state: EvalAgentState) -> dict:
    """Execute the compiled plan subgraph."""
    plan_subgraph = _get_plan_subgraph()
    return await plan_subgraph.ainvoke(state)


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


async def _prepare_run_context(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_prepare_run_context requires target_agent_config to be set")

    target_graph_name = cfg.graph_name
    target_url = cfg.url or state.deployment_url
    if not target_url:
        raise ValueError("_prepare_run_context requires a deployment URL")

    dataset_name = state.dataset_name
    if not dataset_name:
        agent_id = target_graph_name.replace("/", "_")
        date_tag = datetime.now().strftime("%Y%m%d")
        dataset_name = f"seer_eval_{agent_id}_{date_tag}"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = state.experiment_name or f"seer-local-eval-{target_graph_name}-{timestamp}"

    sync_client = get_sync_client(url=target_url)
    thread = await asyncio.to_thread(sync_client.threads.create)
    thread_id = thread["thread_id"]

    deployment_metadata = dict(getattr(state, "deployment_metadata", {}) or {})
    deployment_metadata.setdefault("deployment_url", target_url)
    deployment_metadata.setdefault("repo_url", cfg.repo_url)
    deployment_metadata.setdefault("branch_name", cfg.branch_name)

    logger.info(
        "run.prepare: dataset=%s experiment=%s thread=%s",
        dataset_name,
        experiment_name,
        thread_id,
    )

    return {
        "dataset_name": dataset_name,
        "experiment_name": experiment_name,
        "current_run_thread_id": thread_id,
        "deployment_metadata": deployment_metadata,
        "last_failed_cases": [],
        "last_run_results": [],
        "last_run_metadata": {
            "target_graph_name": target_graph_name,
            "target_url": target_url,
            "experiment_prepared_at": datetime.now(timezone.utc),
        },
    }


async def _execute_test_cases(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_execute_test_cases requires target_agent_config to be set")

    target_url = cfg.url or state.deployment_url
    if not target_url:
        raise ValueError("_execute_test_cases requires a deployment URL")

    thread_id = state.current_run_thread_id
    sync_client = get_sync_client(url=target_url)
    if not thread_id:
        thread = sync_client.threads.create()
        thread_id = thread["thread_id"]

    thread_cfg = {"configurable": {"thread_id": thread_id}}
    remote_graph = RemoteGraph(
        cfg.graph_name,
        url=target_url,
        client=_LANGSMITH_CLIENT,
        sync_client=sync_client,
        distributed_tracing=True,
    )

    results_payload: list[dict] = []
    failed_cases: list[dict] = []
    scores: list[float] = []
    passed_count = 0
    total_tests = len(state.test_cases)

    experiment_start_time = datetime.now(timezone.utc)
    earliest_start = experiment_start_time
    latest_end = experiment_start_time

    for tc in state.test_cases:
        question = tc.input_message
        expected = tc.expected_output or tc.expected_behavior
        row_id = uuid.uuid4().hex

        run_start = datetime.now(timezone.utc)
        try:
            result = await asyncio.to_thread(
                remote_graph.invoke,
                {"messages": [{"role": "user", "content": question}]},
                thread_cfg,
            )
            answer = result.get("messages", [{}])[-1].get("content", "")
        except Exception as invoke_error:
            logger.error("run.execute: error invoking remote graph: %s", invoke_error)
            answer = ""
        run_end = datetime.now(timezone.utc)

        try:
            eval_result: EvaluatorResult = await asyncio.to_thread(
                _CORRECTNESS_EVALUATOR,
                inputs={"question": question},
                outputs={"answer": answer},
                reference_outputs={"answer": expected},
            )
            score = float(eval_result.get("score", 0.0))
            evaluator_comment = eval_result.get("comment", "")
        except Exception as eval_error:
            logger.error("run.execute: error running correctness evaluator: %s", eval_error)
            score = 0.0
            evaluator_comment = f"Evaluation error: {eval_error}"

        scores.append(score)
        if score >= PASS_THRESHOLD:
            passed_count += 1
        else:
            failed_cases.append(
                {
                    "expectation_ref": tc.expectation_ref,
                    "input": question,
                    "expected_output": expected,
                    "actual_output": answer,
                    "success_criteria": tc.success_criteria,
                    "score": score,
                    "judge_comment": evaluator_comment,
                }
            )

        results_payload.append(
            {
                "row_id": row_id,
                "inputs": {"question": question},
                "expected_outputs": {"answer": expected},
                "actual_outputs": {"answer": answer},
                "evaluation_scores": [
                    {
                        "key": "correctness",
                        "score": score,
                        "comment": evaluator_comment,
                    }
                ],
                "start_time": run_start,
                "end_time": run_end,
                "run_name": cfg.graph_name,
                "run_metadata": {
                    "expectation_ref": tc.expectation_ref,
                    "success_criteria": tc.success_criteria,
                },
            }
        )

        if run_start < earliest_start:
            earliest_start = run_start
        if run_end > latest_end:
            latest_end = run_end

    logger.info(
        "run.execute: completed %d tests (failures=%d)",
        total_tests,
        len(failed_cases),
    )

    metadata = dict(state.last_run_metadata or {})
    metadata.update(
        {
            "scores": scores,
            "passed_count": passed_count,
            "total_tests": total_tests,
            "experiment_start_time": experiment_start_time,
            "earliest_start": earliest_start,
            "latest_end": latest_end,
        }
    )

    accumulated_cases = list(state.accumulated_failed_cases or [])
    accumulated_context = dict(state.accumulated_run_context or {})
    if failed_cases:
        accumulated_cases.extend(failed_cases)
        accumulated_context.update(
            {
                "dataset_name": state.dataset_name,
                "experiment_name": state.experiment_name,
                "target_url": metadata.get("target_url", state.deployment_url or cfg.url),
                "latest_score": metadata.get("scores", [0])[-1] if metadata.get("scores") else state.score,
                "aggregate_score": metadata.get("aggregate_score", state.score),
                "total_tests": metadata.get("total_tests", total_tests),
                "passed_tests": metadata.get("passed_count", passed_count),
            }
        )

    return {
        "current_run_thread_id": thread_id,
        "last_run_results": results_payload,
        "last_failed_cases": failed_cases,
        "last_run_metadata": metadata,
        "accumulated_failed_cases": accumulated_cases,
        "accumulated_run_context": accumulated_context,
    }


async def _upload_run_results(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_upload_run_results requires target_agent_config to be set")

    results_payload = list(state.last_run_results or [])
    metadata = dict(state.last_run_metadata or {})
    failed_cases = list(state.last_failed_cases or [])
    scores: list[float] = list(metadata.get("scores", []))
    passed_count = metadata.get("passed_count", 0)
    total_tests = metadata.get("total_tests", len(state.test_cases))

    mean_score = round(sum(scores) / max(len(scores), 1), 4)
    earliest_start = metadata.get("earliest_start", datetime.now(timezone.utc))
    latest_end = metadata.get("latest_end", earliest_start)
    experiment_end_time = max(latest_end, datetime.now(timezone.utc))

    score_history = list(state.score_history or [])
    score_history.append(float(mean_score))
    aggregate_score = round(sum(score_history) / len(score_history), 4)

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY environment variable is required for experiment upload.")

    endpoint_base = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
    endpoint = f"{endpoint_base.rstrip('/')}/api/v1/datasets/upload-experiment"

    serialised_results = []
    for row in results_payload:
        serialised_results.append(
            {
                **row,
                "start_time": row["start_time"].isoformat(),
                "end_time": row["end_time"].isoformat(),
            }
        )

    upload_body = {
        "experiment_name": state.experiment_name,
        "experiment_description": "Evaluation uploaded by Seer eval_agent run_node.",
        "dataset_name": state.dataset_name,
        "experiment_start_time": earliest_start.isoformat(),
        "experiment_end_time": experiment_end_time.isoformat(),
        "experiment_metadata": {
            "target_graph_name": cfg.graph_name,
            "target_url": metadata.get("target_url", state.deployment_url),
            "attempt": len(score_history),
        },
        "summary_experiment_scores": [
            {
                "key": "mean_correctness",
                "score": mean_score,
                "comment": "Average correctness score across generated tests.",
            },
            {
                "key": "aggregate_correctness",
                "score": aggregate_score,
                "comment": "Rolling average correctness score across eval attempts.",
            },
        ],
        "results": serialised_results,
    }

    response = await asyncio.to_thread(
        requests.post,
        endpoint,
        json=upload_body,
        headers={"x-api-key": api_key},
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(
            f"LangSmith upload failed with status {response.status_code}: {response.text}"
        )

    response_data = response.json()
    uploaded_dataset_name = response_data.get("dataset", {}).get("name", state.dataset_name)
    uploaded_experiment_name = response_data.get("experiment", {}).get("name", state.experiment_name)

    logger.info(
        "run.upload: uploaded experiment=%s dataset=%s latest=%.3f aggregate=%.3f rows=%d",
        uploaded_experiment_name,
        uploaded_dataset_name,
        mean_score,
        aggregate_score,
        len(results_payload),
    )

    tool_payload = {
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "latest_score": mean_score,
        "average_score": aggregate_score,
        "rows": len(results_payload),
        "total_tests": total_tests,
        "failed_tests": len(failed_cases),
        "passed_tests": passed_count,
        "deployment_url": metadata.get("target_url", state.deployment_url),
        "codex_handoff_status": "pending",
    }

    last_metadata = {
        **metadata,
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "latest_score": mean_score,
        "aggregate_score": aggregate_score,
        "failed_tests": len(failed_cases),
        "passed_tests": passed_count,
        "total_tests": total_tests,
        "handoff_ready_at": datetime.now(timezone.utc),
    }

    deployment_metadata = dict(getattr(state, "deployment_metadata", {}) or {})
    deployment_metadata.update({
        "deployment_url": metadata.get("target_url", state.deployment_url),
        "repo_url": cfg.repo_url,
        "branch_name": cfg.branch_name,
        "last_upload_at": datetime.now(timezone.utc).isoformat(),
        "codex_handoff_status": "pending",
    })

    tool_message = ToolMessage(content=json.dumps(tool_payload), tool_call_id="run_node")

    return {
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "score": float(mean_score),
        "score_history": score_history,
        "messages": [tool_message],
        "deployment_metadata": deployment_metadata,
        "last_run_metadata": last_metadata,
        "codex_request": None,
        "codex_response": None,
        "accumulated_run_context": {
            **state.accumulated_run_context,
            "dataset_name": uploaded_dataset_name,
            "experiment_name": uploaded_experiment_name,
            "latest_score": mean_score,
            "aggregate_score": aggregate_score,
            "total_tests": total_tests,
            "passed_tests": passed_count,
        },
    }


def _get_run_subgraph():
    builder = StateGraph(EvalAgentState)
    builder.add_node("prepare", _prepare_run_context)
    builder.add_node("execute", _execute_test_cases)
    builder.add_node("upload", _upload_run_results)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "execute")
    builder.add_edge("execute", "upload")
    builder.add_edge("upload", END)

    return builder.compile()


def reflect_node(state: EvalAgentState) -> dict:
    """Summarize how tests should be improved and persist as EvalReflection."""
    cfg = state.target_agent_config
    agent_name = cfg.graph_name
    expectations = cfg.expectations

    # Build a concise reflection focused on test quality/coverage
    latest_score = (state.score_history[-1] if getattr(state, "score_history", None) else state.score)

    summary_prompt = (
        "You are a QA lead improving E2E eval tests. "
        "Given the agent name and the user's expectations, produce a short summary of improvements to future tests. "
        "Focus on edge cases, negative cases, and clarity of expected outputs.\n\n"
        f"Agent: {agent_name}\n"
        f"Expectations: {expectations}\n"
        f"Latest score: {latest_score:.3f} (attempt {state.attempts + 1})\n"
    )

    reflection_llm = _LLM.with_structured_output(EvalReflection)
    reflection: EvalReflection = reflection_llm.invoke(summary_prompt)

    # Ensure correct agent_name populated
    reflection.agent_name = agent_name

    try:
        key = uuid.uuid4().hex
        _LANGGRAPH_SYNC_CLIENT.store.put_item(
            ("eval_reflections", agent_name),
            key=key,
            value=reflection.model_dump(),
        )
        logger.info("reflect_node: stored eval reflection")
    except Exception:
        logger.warning("reflect_node: failed to store eval reflection")

    # Increment attempts and append to in-memory reflections list
    new_reflections = list(state.reflections)
    new_reflections.append(reflection)

    return {
        "attempts": state.attempts + 1,
        "reflections": new_reflections,
        "deployment_url": state.deployment_url,
        "deployment_metadata": dict(getattr(state, "deployment_metadata", {}) or {}),
        "sandbox_id": state.sandbox_id,
        "sandbox_repo_dir": state.sandbox_repo_dir,
        "sandbox_branch": state.sandbox_branch,
        "codex_thread_id": state.codex_thread_id,
        "codex_request": state.codex_request,
        "codex_response": state.codex_response,
    }


def should_continue(state: EvalAgentState) -> Literal["reflect", "finalize"]:
    # Enforce at least MIN_ATTEMPTS eval→reflect cycles before considering finalize
    if state.attempts < MIN_ATTEMPTS:
        return "reflect"
    if state.score >= PASS_THRESHOLD or state.attempts >= MAX_ATTEMPTS:
        return "finalize"
    return "reflect"


async def finalize_node(state: EvalAgentState) -> dict:
    payload = {
        "attempts": state.attempts,
        "score": state.score,
        "dataset_name": state.dataset_name,
        "experiment_name": state.experiment_name,
    }
    score_history = list(getattr(state, "score_history", []) or [])
    attempts = len(score_history)
    average_score = (sum(score_history) / attempts) if attempts else state.score
    latest_score = score_history[-1] if attempts else state.score

    deployment_url = state.deployment_url
    deployment_metadata = dict(getattr(state, "deployment_metadata", {}) or {})
    codex_thread_id = state.codex_thread_id
    failed_cases = list(state.accumulated_failed_cases or state.last_failed_cases or [])
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
    metadata = dict(state.accumulated_run_context or state.last_run_metadata or {})
    failed_cases_count = len(failed_cases)
    codex_status = "skipped (no failing tests)" if failed_cases_count == 0 else "pending"

    cfg = state.target_agent_config
    codex_request_payload = state.codex_request
    codex_response_payload = state.codex_response

    resolved_repo_url = (cfg.repo_url if cfg else None) or deployment_metadata.get("repo_url")
    github_context = GithubContext(repo_url=resolved_repo_url) if resolved_repo_url else None

    resolved_sandbox_id = state.sandbox_id or deployment_metadata.get("sandbox_id")
    resolved_sandbox_dir = state.sandbox_repo_dir or deployment_metadata.get("sandbox_repo_dir")
    resolved_sandbox_branch = state.sandbox_branch or deployment_metadata.get("branch_name")
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

    if failed_cases_count > 0 and cfg is not None:
        dataset_name = metadata.get("dataset_name", state.dataset_name)
        experiment_name = metadata.get("experiment_name", state.experiment_name)
        total_tests = metadata.get("total_tests", len(state.test_cases))
        passed_tests = metadata.get("passed_tests", total_tests - failed_cases_count)
        latest_score_value = metadata.get("latest_score", state.score)
        aggregate_score_value = metadata.get("aggregate_score", average_score)
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
        repo_path = state.sandbox_repo_dir or deployment_metadata.get("sandbox_repo_dir")
        if repo_path:
            planner_payload["repo_path"] = repo_path
        metadata_repo_path = metadata.get("repo_path")
        if metadata_repo_path and "repo_path" not in planner_payload:
            planner_payload["repo_path"] = metadata_repo_path
        if "repo_path" not in planner_payload:
            planner_payload["repo_path"] = "/tmp/codex"  # fallback to satisfy schema
        if state.sandbox_id:
            planner_payload["sandbox_session_id"] = state.sandbox_id

        logger.info(f"Planner payload: {planner_payload}")

        try:
            codex_sync_client = get_sync_client(url=_CODEX_REMOTE_URL)
            if not codex_thread_id:
                thread = await asyncio.to_thread(codex_sync_client.threads.create)
                codex_thread_id = thread["thread_id"]

            codex_thread_cfg = {"configurable": {"thread_id": codex_thread_id}}
            codex_remote = RemoteGraph(
                'planner',
                url=_CODEX_REMOTE_URL,
                client=_LANGSMITH_CLIENT,
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
            codex_status = "ok"
            deployment_metadata["codex_last_handoff_at"] = handoff_timestamp.isoformat()
            failed_cases = []
            metadata.clear()
        except Exception as codex_error:
            logger.error("finalize_node: failed to hand off to Codex: %s", codex_error, exc_info=True)
            codex_status = f"error: {codex_error}"

    logger.info(
        "finalize_node: attempts=%d latest_score=%.3f average_score=%.3f payload=%s",
        attempts,
        latest_score,
        average_score,
        payload,
    )

    user_summary = (
        f"Final evaluation complete: attempts={max(attempts, state.attempts)}; "
        f"average score={average_score:.2f} (0–1), latest={latest_score:.2f}. "
        f"Dataset=`{state.dataset_name}`, Experiment=`{state.experiment_name}`."
    )
    if deployment_url:
        user_summary += f" Deployment URL: {deployment_url}."
    user_summary += f" Codex handoff status: {codex_status}."
    if codex_thread_id:
        user_summary += f" Codex thread ID: {codex_thread_id}."
    if failed_cases_count:
        user_summary += f" Escalated failing tests: {failed_cases_count}."

    deployment_metadata["codex_handoff_status"] = codex_status

    return {
        "messages": [AIMessage(content=user_summary)],
        "deployment_url": deployment_url,
        "deployment_metadata": deployment_metadata,
        "codex_thread_id": codex_thread_id,
        "codex_request": codex_request_payload,
        "codex_response": codex_response_payload,
        "accumulated_failed_cases": failed_cases,
        "accumulated_run_context": metadata,
    }


def build_graph():
    workflow = StateGraph(EvalAgentState)
    plan_subgraph = _get_plan_subgraph()
    workflow.add_node("plan", plan_subgraph)
    workflow.add_node("run", _get_run_subgraph())
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "run")
    workflow.add_conditional_edges("run", should_continue, {"reflect": "reflect", "finalize": "finalize"})
    workflow.add_edge("reflect", "plan")
    workflow.add_edge("finalize", END)

    graph = workflow.compile(debug=True)
    logger.info("Eval Agent graph compiled successfully")
    return graph


graph = build_graph()
