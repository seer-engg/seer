"""Judge Agent - Runs a single test and writes result to a file"""

import os
import json
import uuid
from typing import Annotated, TypedDict, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Import shared schemas
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.schemas import TestCase, TestResult
from shared.prompts import EVAL_AGENT_JUDGE_PROMPT


class JudgeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    target_url: Optional[str]
    target_agent_id: Optional[str]
    test_case: Optional[TestCase]
    actual_output: Optional[str]
    verdict_passed: Optional[bool]
    verdict_score: Optional[float]
    verdict_reasoning: Optional[str]
    results_file: Optional[str]
    test_result: Optional[TestResult]


class JudgeInput(BaseModel):
    target_url: str
    target_agent_id: str
    test_case: TestCase
    results_file: str = Field(description="Absolute path for results file to append to")


class JudgeVerdict(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


def parse_input_node(state: JudgeState):
    """Parse JSON payload from the latest human message into state variables."""
    last_human = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m
            break
    if not last_human:
        return {}

    payload = last_human.content
    data = {}
    try:
        data = json.loads(payload)
    except Exception:
        # Not JSON; nothing to do
        return {}

    try:
        judge_input = JudgeInput(**data)
    except Exception:
        return {}

    return {
        "target_url": judge_input.target_url,
        "target_agent_id": judge_input.target_agent_id,
        "test_case": judge_input.test_case,
        "results_file": judge_input.results_file,
    }


def run_test_node(state: JudgeState):
    """Call the target agent; prefer /invoke JSON, fallback to /runs/stream SSE; capture final AI output."""
    target_url = state.get("target_url") or ""
    target_agent_id = state.get("target_agent_id") or ""
    tc: TestCase = state.get("test_case")  # type: ignore[assignment]
    if not target_url or not target_agent_id or tc is None:
        return {}

    actual_output = ""
    try:
        import httpx
        with httpx.Client(timeout=60.0) as client:
            # 1) Try /invoke first (JSON response)
            invoke_payload = {
                "assistant_id": target_agent_id,
                "input": {
                    "messages": [{"role": "user", "content": tc.input_message}]
                },
                "config": {"configurable": {"thread_id": str(uuid.uuid4())}}
            }
            try:
                resp = client.post(f"{target_url}/invoke", json=invoke_payload)
                if resp.status_code < 400:
                    try:
                        data = resp.json()
                    except Exception:
                        data = None
                    if isinstance(data, dict):
                        # Common shapes: {'output': {'messages': [...]}} or {'messages': [...]}
                        output = data.get("output") if isinstance(data.get("output"), dict) else data
                        msgs = None
                        if isinstance(output, dict):
                            msgs = output.get("messages") or output.get("outputs")
                        if msgs is None:
                            msgs = data.get("messages")
                        if isinstance(msgs, list):
                            for msg in msgs:
                                if isinstance(msg, dict) and (msg.get("type") == "ai" or msg.get("role") == "assistant"):
                                    content = msg.get("content") or msg.get("text") or ""
                                    if content:
                                        actual_output = content
                        # Fallback to simple content fields
                        if not actual_output:
                            actual_output = (
                                data.get("content")
                                or data.get("response")
                                or data.get("text")
                                or ""
                            )
                    if actual_output:
                        return {"actual_output": actual_output}
            except Exception:
                # Fall through to SSE
                pass

            # 2) Fallback to /runs/stream (SSE)
            sse_payload = {
                "assistant_id": target_agent_id,
                "input": {
                    "messages": [{"role": "user", "content": tc.input_message}]
                },
                "stream_mode": ["values"],
                "config": {"configurable": {"thread_id": str(uuid.uuid4())}}
            }
            # If server returns an error, surface minimal detail in actual_output
            resp = client.post(f"{target_url}/runs/stream", json=sse_payload)
            if resp.status_code >= 400:
                return {"actual_output": f"ERROR: HTTP {resp.status_code} {resp.text[:200]}"}
            resp.raise_for_status()
            for line in resp.text.strip().split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    obj = json.loads(line[6:])
                    # Try multiple shapes for messages
                    vals = obj.get("values", {}) if isinstance(obj, dict) else {}
                    if not isinstance(vals, dict):
                        vals = obj.get("value", {}) if isinstance(obj, dict) else {}
                    data_field = obj.get("data", {}) if isinstance(obj.get("data"), dict) else {}
                    msgs = None
                    if isinstance(vals, dict):
                        msgs = vals.get("messages")
                    if msgs is None and isinstance(data_field, dict):
                        msgs = data_field.get("messages")
                    if msgs is None:
                        msgs = obj.get("messages")
                    if isinstance(msgs, list):
                        for msg in msgs:
                            if isinstance(msg, dict) and (msg.get("type") == "ai" or msg.get("role") == "assistant"):
                                content = msg.get("content") or msg.get("text") or ""
                                if content:
                                    actual_output = content
                except Exception:
                    continue
    except Exception as e:
        actual_output = f"ERROR: {str(e)}"

    if not actual_output:
        actual_output = "No response received"

    return {"actual_output": actual_output}


def judge_node(state: JudgeState):
    """Use LLM structured output to judge the test result."""
    tc: TestCase = state.get("test_case")  # type: ignore[assignment]
    actual_output = state.get("actual_output") or ""
    if tc is None:
        return {}

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(JudgeVerdict)

    prompt = EVAL_AGENT_JUDGE_PROMPT.format(
        input_message=tc.input_message,
        expected_behavior=tc.expected_behavior,
        success_criteria=tc.success_criteria,
        actual_output=actual_output,
    )
    verdict: JudgeVerdict = llm.invoke(prompt)

    result = TestResult(
        test_case_id=tc.id,
        input_sent=tc.input_message,
        actual_output=actual_output,
        expected_behavior=tc.expected_behavior,
        passed=verdict.passed,
        score=verdict.score,
        judge_reasoning=verdict.reasoning,
    )

    return {
        "verdict_passed": verdict.passed,
        "verdict_score": verdict.score,
        "verdict_reasoning": verdict.reasoning,
        "test_result": result,
    }


def write_result_node(state: JudgeState):
    """Append the test_result to the results JSON file path provided."""
    results_file = state.get("results_file") or ""
    result: TestResult = state.get("test_result")  # type: ignore[assignment]
    if not results_file or result is None:
        return {}

    path = Path(results_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"results": []}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"results": []}
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        data = {"results": []}

    data["results"].append(result.model_dump())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Also reflect a minimal message for traceability
    note = (
        "JUDGE_RESULT\n"
        f"file: {str(path)}\n"
        f"appended: {result.test_case_id} passed={result.passed} score={result.score:.2f}"
    )
    return {"messages": [SystemMessage(content=note)]}


def build_graph():
    """Build the judge agent graph."""
    workflow = StateGraph(JudgeState)
    workflow.add_node("parse_input", parse_input_node)
    workflow.add_node("run_test", run_test_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("write", write_result_node)

    workflow.set_entry_point("parse_input")
    workflow.add_edge("parse_input", "run_test")
    workflow.add_edge("run_test", "judge")
    workflow.add_edge("judge", "write")
    workflow.add_edge("write", END)
    return workflow.compile()


graph = build_graph()


