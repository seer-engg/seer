import os
import json
from datetime import datetime
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client

from agents.eval_agent.models import (
    EvalV2State,
    EvalReflection,
    AgentSpec,
    GeneratedTests,
    GeneratedTestCase,
    TargetAgentConfig,
)
from agents.eval_agent.prompts import (
    EVAL_AGENT_SPEC_PROMPT,
    EVAL_AGENT_TEST_GEN_PROMPT,
)
from agents.eval_agent.graph import (
    correctness_evaluator,
)
from shared.logger import get_logger
from shared.llm import get_llm
import uuid
from typing import Any, Dict, List


from agents.eval_agent.models import EvalReflection
from shared.logger import get_logger
from langgraph_sdk import get_client
from langsmith import Client



logger = get_logger("eval_agent.v2")


_LANGGRAPH_CLIENT = get_client(url="http://127.0.0.1:8002")
_LANGGRAPH_SYNC_CLIENT = get_sync_client(url="http://127.0.0.1:8002")
_LANGSMITH_CLIENT = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
PASS_THRESHOLD = 0.90
MAX_ATTEMPTS = 10
MIN_ATTEMPTS = 3

# Use a slightly higher temperature for test generation to encourage diversity
_LLM = get_llm(temperature=0.2)


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

    print('XXX', reflections_text, flush=True)
    print('XXX', prev_inputs_text, flush=True)
    print('XXX', spec_obj, flush=True)

    # Generate tests with reflection context and anti-duplication hints
    augmented_prompt = EVAL_AGENT_TEST_GEN_PROMPT.format(
        spec_json=spec_obj, 
        reflections_text=reflections_text, 
        prev_inputs_text=prev_inputs_text
    )

    generated: GeneratedTests = _LLM.with_structured_output(GeneratedTests).invoke(augmented_prompt)
    assert len(generated.test_cases) == 3, f"Generated {len(generated.test_cases)} test cases, expected 3"

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


def _ensure_dataset(state: EvalV2State) -> str:
    if state.dataset_name:
        return state.dataset_name
    agent_id = state.target_agent_config.graph_name.replace("/", "_")
    date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"seer_eval_v2_{agent_id}_{date_tag}"
    _LANGSMITH_CLIENT.create_dataset(name)
    return name


def run_node(state: EvalV2State) -> dict:
    """Create dataset, upsert tests, evaluate target, compute mean score locally."""
    cfg = state.target_agent_config
    target_graph_name = cfg.graph_name
    target_url = cfg.url

    # Dataset + examples
    dataset_name = _ensure_dataset(state)
    dataset = _LANGSMITH_CLIENT.read_dataset(dataset_name=dataset_name)
    examples = [
        {
            "inputs": {"question": tc.input_message},
            "outputs": {"answer": (tc.expected_output or tc.expected_behavior)},
        }
        for tc in state.test_cases
    ]
    if examples:
        _LANGSMITH_CLIENT.create_examples(dataset_id=dataset.id, examples=examples)

    # Evaluate locally (compute score) using RemoteGraph
    sync_client = get_sync_client(url=target_url)
    thread = sync_client.threads.create()
    thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
    remote_graph = RemoteGraph(target_graph_name, url=target_url, client=_LANGSMITH_CLIENT, sync_client=sync_client, distributed_tracing=True)

    scores = []

    tc: GeneratedTestCase = None
    for tc in state.test_cases:
        question = tc.input_message
        expected = tc.expected_output or tc.expected_behavior
        try:
            result = remote_graph.invoke({"messages": [{"role": "user", "content": question}]}, config=thread_cfg)
            answer = result.get("messages", [{}])[-1].get("content", "")
        except Exception:
            answer = ""

        try:
            eval_result = correctness_evaluator(
                inputs={"question": question},
                outputs={"answer": answer},
                reference_outputs={"answer": expected},
            )
            score = _extract_score(eval_result)
        except Exception:
            score = 0.0

        scores.append(score)

    mean_score = sum(scores) / max(len(scores), 1)
    logger.info(f"run_node: mean_score={mean_score:.3f} (cases={len(scores)})")

    # Emit a ToolMessage for traceability
    msg = ToolMessage(content=json.dumps({"dataset": dataset_name, "mean_score": mean_score}), tool_call_id="run_node")

    return {
        "dataset_name": dataset_name,
        "score": float(mean_score),
        "messages": [msg],
    }


def _extract_score(eval_result) -> float:
    try:
        if isinstance(eval_result, (int, float)):
            return float(eval_result)
        if isinstance(eval_result, dict):
            if "score" in eval_result:
                return float(eval_result.get("score", 0.0))
            inner = eval_result.get("correctness") or {}
            if isinstance(inner, dict) and "score" in inner:
                return float(inner.get("score", 0.0))
        return 0.0
    except Exception:
        return 0.0


def reflect_node(state: EvalV2State) -> dict:
    """Summarize how tests should be improved and persist as EvalReflection."""
    cfg = state.target_agent_config
    agent_name = cfg.graph_name
    expectations = cfg.expectations

    # Build a concise reflection focused on test quality/coverage
    summary_prompt = (
        "You are a QA lead improving E2E eval tests. "
        "Given the agent name and the user's expectations, produce a short summary of improvements to future tests. "
        "Focus on edge cases, negative cases, and clarity of expected outputs.\n\n"
        f"Agent: {agent_name}\n"
        f"Expectations: {expectations}\n"
        f"Latest score: {state.score:.3f} (attempt {state.attempts + 1})\n"
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
    logger.info(f"finalize_node: {payload}")
    return {}


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

    graph = workflow.compile()
    logger.info("Eval Agent V2 graph compiled successfully")
    return graph


graph = build_graph()


