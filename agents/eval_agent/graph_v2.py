import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

import requests

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_client, get_sync_client
from langsmith import Client

from agents.eval_agent.models import (
    AgentSpec,
    EvalReflection,
    EvalV2State,
    GeneratedTestCase,
    GeneratedTests,
    TargetAgentConfig,
)
from agents.eval_agent.prompts import (
    EVAL_AGENT_SPEC_PROMPT,
    EVAL_AGENT_TEST_GEN_PROMPT,
)
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.types import EvaluatorResult
from shared.logger import get_logger
from shared.llm import get_llm



logger = get_logger("eval_agent.v2")


_LANGGRAPH_CLIENT = get_client(url="http://127.0.0.1:8002")
_LANGGRAPH_SYNC_CLIENT = get_sync_client(url="http://127.0.0.1:8002")
_LANGSMITH_CLIENT = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
PASS_THRESHOLD = 0.90
MAX_ATTEMPTS = 4
MIN_ATTEMPTS = 2

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




def plan_node(state: EvalV2State) -> dict:
    """Regenerate tests using past eval reflections as context."""
    cfg = state.target_agent_config
    # If target config missing, extract from the last human message (compat with v1 agent UX)
    if cfg is None:
        # Find last human message content
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

        extractor = _LLM.with_structured_output(TargetAgentConfig)
        instruction = (
            "Extract graph_name, url, and expectations from the user's latest message.\n\n"
            "url should begin with http:// or https://"
            "IMPORTANT - graph_name is the name of the graph (NOT ASSISTANT ID WHICH IS A HEX STRING) to evaluate"
        )
        cfg = extractor.invoke(f"{instruction}\n\nUSER:\n{last_human.content}")

    agent_name = cfg.graph_name
    expectations = cfg.expectations

    # Retrieve top-K reflections and fold into test generation
    found = _LANGGRAPH_SYNC_CLIENT.store.search_items(
        ("eval_reflections", agent_name),
        query=expectations,
        limit=5,
    )
    prior_reflections = []
    for item in found:
        try:
            value = item.get("value") if isinstance(item, dict) else item
            if isinstance(value, dict):
                prior_reflections.append(value.get("summary") or json.dumps(value))
        except Exception:
            continue
    reflections_text = "\n- ".join(prior_reflections) if prior_reflections else "(none)"

    # Build AgentSpec
    spec_llm = _LLM.with_structured_output(AgentSpec)
    spec_obj: AgentSpec = spec_llm.invoke(
        EVAL_AGENT_SPEC_PROMPT.format(
            expectations=expectations,
            agent_name=agent_name,
            agent_url=cfg.url,
        )
    )

    # Compose previous inputs to discourage duplicates
    prev_inputs = state.previous_inputs or []
    prev_inputs_text = "\n- ".join(prev_inputs) if prev_inputs else "(none)"

    logger.debug(reflections_text)
    logger.debug(prev_inputs_text)
    logger.debug(spec_obj)

    # Generate tests with reflection context and anti-duplication hints
    augmented_prompt = EVAL_AGENT_TEST_GEN_PROMPT.format(
        spec_json=spec_obj, 
        reflections_text=reflections_text, 
        prev_inputs_text=prev_inputs_text
    )

    generated: GeneratedTests = _LLM.with_structured_output(GeneratedTests).invoke(augmented_prompt)
    assert len(generated.test_cases) == 5, f"Generated {len(generated.test_cases)} test cases, expected 5"

    test_cases: list[GeneratedTestCase] = []
    for idx, tc in enumerate(generated.test_cases):
        test_cases.append(GeneratedTestCase(
            expectation_ref=tc.expectation_ref,
            input_message=tc.input_message,
            expected_behavior=tc.expected_behavior,
            success_criteria=tc.success_criteria,
            expected_output=getattr(tc, "expected_output", None) or "",
        ))

    # Update previous inputs
    new_prev = list(prev_inputs)
    new_prev.extend([tc.input_message for tc in test_cases])

    logger.info(f"plan_node: generated {len(test_cases)} tests (agent={agent_name})")
    return {
        "test_cases": test_cases,
        "target_agent_config": cfg,
        "previous_inputs": new_prev,
        # ensure dataset/activity resets on each plan
        "dataset_name": "",
        "experiment_name": "",
    }


def run_node(state: EvalV2State) -> dict:
    """Execute tests, score them locally, and upload results to LangSmith via REST."""
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("run_node requires target_agent_config to be set")

    target_graph_name = cfg.graph_name
    target_url = cfg.url

    dataset_name = state.dataset_name
    experiment_name = state.experiment_name

    if not dataset_name:
        agent_id = target_graph_name.replace("/", "_")
        date_tag = datetime.now().strftime("%Y%m%d")
        dataset_name = f"seer_eval_v2_{agent_id}_{date_tag}"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = experiment_name or f"seer-local-eval-{target_graph_name}-{timestamp}"

    sync_client = get_sync_client(url=target_url)
    thread = sync_client.threads.create()
    thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
    remote_graph = RemoteGraph(
        target_graph_name,
        url=target_url,
        client=_LANGSMITH_CLIENT,
        sync_client=sync_client,
        distributed_tracing=True,
    )

    experiment_start_time = datetime.now(timezone.utc)
    earliest_start = experiment_start_time
    latest_end = experiment_start_time
    scores: list[float] = []
    results_payload: list[dict] = []

    for tc in state.test_cases:
        question = tc.input_message
        expected = tc.expected_output or tc.expected_behavior

        row_id = uuid.uuid4().hex
        run_start = datetime.now(timezone.utc)
        try:
            result = remote_graph.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=thread_cfg,
            )
            answer = result.get("messages", [{}])[-1].get("content", "")
        except Exception as invoke_error:
            logger.error("run_node: error invoking remote graph: %s", invoke_error)
            answer = ""
        run_end = datetime.now(timezone.utc)

        try:
            eval_result: EvaluatorResult = _CORRECTNESS_EVALUATOR(
                inputs={"question": question},
                outputs={"answer": answer},
                reference_outputs={"answer": expected},
            )
            score = float(eval_result.get("score", 0.0))
            evaluator_comment = eval_result.get("comment", "")
        except Exception as eval_error:
            logger.error("run_node: error running correctness evaluator: %s", eval_error)
            score = 0.0
            evaluator_comment = f"Evaluation error: {eval_error}"

        scores.append(score)

        results_payload.append({
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
            "start_time": run_start.isoformat(),
            "end_time": run_end.isoformat(),
            "run_name": target_graph_name,
            "run_metadata": {
                "expectation_ref": tc.expectation_ref,
                "success_criteria": tc.success_criteria,
            },
        })

        if run_start < earliest_start:
            earliest_start = run_start
        if run_end > latest_end:
            latest_end = run_end

    mean_score = round(sum(scores) / max(len(scores), 1), 4)
    experiment_end_time = max(latest_end, datetime.now(timezone.utc))

    score_history = list(state.score_history or [])
    score_history.append(float(mean_score))
    aggregate_score = round(sum(score_history) / len(score_history), 4)

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY environment variable is required for experiment upload.")

    api_base = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
    endpoint = f"{api_base.rstrip('/')}/api/v1/datasets/upload-experiment"

    upload_body = {
        "experiment_name": experiment_name,
        "experiment_description": "Evaluation uploaded by Seer eval_agent v2 run_node.",
        "dataset_name": dataset_name,
        "experiment_start_time": earliest_start.isoformat(),
        "experiment_end_time": experiment_end_time.isoformat(),
        "experiment_metadata": {
            "target_graph_name": target_graph_name,
            "target_url": target_url,
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
        "results": results_payload,
    }

    try:
        response = requests.post(
            endpoint,
            json=upload_body,
            headers={"x-api-key": api_key},
            timeout=60,
        )
    except Exception as request_error:
        raise RuntimeError(f"Failed to upload experiment to LangSmith: {request_error}") from request_error

    if not response.ok:
        raise RuntimeError(
            f"LangSmith upload failed with status {response.status_code}: {response.text}"
        )

    response_data = response.json()
    dataset_info = response_data.get("dataset", {})
    experiment_info = response_data.get("experiment", {})

    uploaded_dataset_name = dataset_info.get("name", dataset_name)
    uploaded_experiment_name = experiment_info.get("name", experiment_name)

    tool_payload = {
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "latest_score": mean_score,
        "average_score": aggregate_score,
        "rows": len(results_payload),
    }
    tool_message = ToolMessage(content=json.dumps(tool_payload), tool_call_id="run_node")

    logger.info(
        "run_node: uploaded experiment=%s dataset=%s latest=%.3f aggregate=%.3f rows=%d",
        uploaded_experiment_name,
        uploaded_dataset_name,
        mean_score,
        aggregate_score,
        len(results_payload),
    )

    return {
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "score": float(mean_score),
        "score_history": score_history,
        "messages": [tool_message],
    }
def reflect_node(state: EvalV2State) -> dict:
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
    }


def should_continue(state: EvalV2State) -> Literal["reflect", "finalize"]:
    # Enforce at least MIN_ATTEMPTS eval→reflect cycles before considering finalize
    if state.attempts < MIN_ATTEMPTS:
        return "reflect"
    if state.score >= PASS_THRESHOLD or state.attempts >= MAX_ATTEMPTS:
        return "finalize"
    return "reflect"


def finalize_node(state: EvalV2State) -> dict:
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
    return {"messages": [AIMessage(content=user_summary)]}


def build_graph():
    workflow = StateGraph(EvalV2State)
    workflow.add_node("plan", plan_node)
    workflow.add_node("run", run_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "run")
    workflow.add_conditional_edges("run", should_continue, {"reflect": "reflect", "finalize": "finalize"})
    workflow.add_edge("reflect", "plan")
    workflow.add_edge("finalize", END)

    graph = workflow.compile(debug=True)
    logger.info("Eval Agent V2 graph compiled successfully")
    return graph


graph = build_graph()


