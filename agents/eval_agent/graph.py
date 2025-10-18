"""Eval Agent LangGraph - Deployable Graph"""

import os
import json
import hashlib
import uuid
from typing import Annotated, TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# Simple state
class AgentState(TypedDict):
    """State for Eval agent"""
    messages: Annotated[list[BaseMessage], add_messages]
    agent_name: Optional[str]
    agent_url: Optional[str]
    expectations: Optional[str]
    spec: Optional["AgentSpec"]
    eval_suite: Optional["EvalSuite"]
    current_test_index: int
    passed: int
    failed: int
    total: int
    test_results: List["TestResult"]


# Import shared schemas (these don't depend on event bus)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.schemas import AgentSpec, Expectation, TestCase, EvalSuite, TestResult
from shared.prompts import (
    EVAL_AGENT_SPEC_PROMPT,
    EVAL_AGENT_TEST_GEN_PROMPT,
    EVAL_AGENT_JUDGE_PROMPT
)


# Tools
from langchain_core.tools import tool
import httpx
import asyncio


# Structured output schemas for node LLM calls
class EvalRequest(BaseModel):
    agent_name: str = Field(description="Name/ID of the agent to evaluate")
    agent_url: str = Field(description="Base URL where the agent is hosted")
    expectations: str = Field(description="User's natural language expectations")


class GeneratedTestCase(BaseModel):
    expectation_ref: str
    input_message: str
    expected_behavior: str
    success_criteria: str


class GeneratedTests(BaseModel):
    test_cases: List[GeneratedTestCase]


class JudgeVerdict(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


def parse_request_node(state: AgentState):
    """Extract agent_name, agent_url, expectations from latest user message using structured output."""
    last_human = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m
            break
    user_text = last_human.content if last_human else ""

    extractor = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(EvalRequest)

    instruction = (
        "Extract agent_name, agent_url, and expectations from the user's latest message. "
        "If agent_name or agent_url are not explicitly provided, infer reasonable placeholders "
        "(use any concise name for agent_name, and a URL if present; otherwise reuse any provided URL-like text). "
        "Always include the original message as expectations."
    )
    request: EvalRequest = extractor.invoke(f"{instruction}\n\nUSER:\n{user_text}")

    return {
        "agent_name": request.agent_name,
        "agent_url": request.agent_url,
        "expectations": request.expectations,
    }


def generate_spec_node(state: AgentState):
    """Generate AgentSpec from state expectations using structured output."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(AgentSpec)

    prompt = EVAL_AGENT_SPEC_PROMPT.format(
        expectations=state.get("expectations", ""),
        agent_name=state.get("agent_name", ""),
        agent_url=state.get("agent_url", ""),
    )

    spec: AgentSpec = llm.invoke(prompt)
    return {"spec": spec}


def generate_tests_node(state: AgentState):
    """Generate EvalSuite test cases from AgentSpec using structured output."""
    spec: AgentSpec = state.get("spec")  # type: ignore[assignment]
    if spec is None:
        return {}

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(GeneratedTests)

    spec_json = spec.model_dump_json(indent=2)
    prompt = EVAL_AGENT_TEST_GEN_PROMPT.format(spec_json=spec_json)
    generated: GeneratedTests = llm.invoke(prompt)

    # Create TestCase objects with stable IDs
    test_cases: List[TestCase] = []
    for idx, tc in enumerate(generated.test_cases):
        content_hash = hashlib.md5(
            f"{tc.expectation_ref}{tc.input_message}".encode()
        ).hexdigest()[:8]

        test_case = TestCase(
            id=f"{spec.name}_{idx+1}_{content_hash}",
            expectation_ref=tc.expectation_ref,
            input_message=tc.input_message,
            expected_behavior=tc.expected_behavior,
            success_criteria=tc.success_criteria,
        )
        test_cases.append(test_case)

    eval_suite = EvalSuite(
        id=f"eval_{spec.name}_{uuid.uuid4().hex[:8]}",
        spec_name=spec.name,
        spec_version=spec.version,
        test_cases=test_cases,
    )

    # Provide a concise context message for the agent node
    test_inputs_preview = "\n".join(
        [f"[{i}] {tc.input_message}" for i, tc in enumerate(test_cases)]
    )
    context_msg = (
        "EVAL_CONTEXT\n"
        f"agent_url: {state.get('agent_url','')}\n"
        f"agent_id: {state.get('agent_name','')}\n"
        f"test_count: {len(test_cases)}\n"
        "test_inputs (indexed):\n"
        f"{test_inputs_preview}"
    )

    return {
        "eval_suite": eval_suite,
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "current_test_index": 0,
        "test_results": [],
        "messages": [SystemMessage(content=context_msg)],
    }


@tool
def request_confirmation(question: str, test_count: int) -> str:
    """
    Request user confirmation before running tests.
    
    Args:
        question: Question to ask user
        test_count: Number of tests that will be run
    
    Returns:
        JSON metadata about the confirmation request
    """
    return json.dumps({
        "action": "CONFIRMATION_REQUEST",
        "test_count": test_count,
        "question": question
    })


@tool
async def run_test(target_url: str, target_agent_id: str, test_input: str, thread_id: str = None) -> str:
    """
    Run a single test against the target LangGraph agent.
    
    Args:
        target_url: URL of the target agent to test (e.g. http://localhost:2024)
        target_agent_id: Graph/assistant ID for the target (e.g. 'my_agent')
        test_input: Input message to send to the agent
        thread_id: Optional thread ID for context
    
    Returns:
        JSON string with the agent's response
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    # Use LangGraph runs/stream API
    url = f"{target_url}/runs/stream"
    payload = {
        "assistant_id": target_agent_id,
        "input": {
            "messages": [{"role": "user", "content": test_input}]
        },
        "stream_mode": ["values"],
        "config": {"configurable": {"thread_id": thread_id}}
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            # Parse streaming response to get final message
            final_response = ""
            for line in response.text.strip().split('\n'):
                if line.startswith('data: '):
                    data = json.loads(line[6:])  # Skip 'data: ' prefix
                    if 'messages' in data:
                        messages = data['messages']
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict) and last_msg.get('type') == 'ai':
                                final_response = last_msg.get('content', '')
            
            if final_response:
                return json.dumps({"success": True, "response": final_response})
            else:
                return json.dumps({"success": True, "response": "No response received"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def judge_node(state: AgentState):
    """Judge the last run_test result using structured output and update counters/results."""
    eval_suite: EvalSuite = state.get("eval_suite")  # type: ignore[assignment]
    idx = state.get("current_test_index", 0)
    if eval_suite is None or idx >= len(eval_suite.test_cases):
        return {}

    # Find the most recent ToolMessage (run_test result)
    last_tool_msg: Optional[ToolMessage] = None
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage):
            last_tool_msg = m
            break

    if last_tool_msg is None:
        return {}

    # Parse the tool content
    actual_output = ""
    try:
        tool_payload = json.loads(last_tool_msg.content)
        if tool_payload.get("success"):
            actual_output = tool_payload.get("response", "")
        else:
            actual_output = tool_payload.get("error", "") or ""
    except Exception:
        actual_output = last_tool_msg.content

    tc = eval_suite.test_cases[idx]

    judge_llm = ChatOpenAI(
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
    verdict: JudgeVerdict = judge_llm.invoke(prompt)

    result = TestResult(
        test_case_id=tc.id,
        input_sent=tc.input_message,
        actual_output=actual_output,
        expected_behavior=tc.expected_behavior,
        passed=verdict.passed,
        score=verdict.score,
        judge_reasoning=verdict.reasoning,
    )

    passed = state.get("passed", 0) + (1 if verdict.passed else 0)
    failed = state.get("failed", 0) + (0 if verdict.passed else 1)
    next_idx = idx + 1

    progress_msg = (
        f"EVAL_PROGRESS\n"
        f"completed: {next_idx}/{state.get('total', len(eval_suite.test_cases))}\n"
        f"passed: {passed}\n"
        f"failed: {failed}"
    )

    return {
        "current_test_index": next_idx,
        "passed": passed,
        "failed": failed,
        "test_results": state.get("test_results", []) + [result],
        "messages": [SystemMessage(content=progress_msg)],
    }


@tool
def summarize_results(passed: int, failed: int, total: int) -> str:
    """
    Summarize test results.
    
    Args:
        passed: Number of tests passed
        failed: Number of tests failed
        total: Total tests
    
    Returns:
        JSON summary of test results
    """
    score = (passed / total * 100) if total > 0 else 0
    return json.dumps({
        "action": "TEST_RESULTS",
        "passed": passed,
        "failed": failed,
        "total": total,
        "score": score,
        "summary": f"{passed}/{total} passed ({score:.0f}%)"
    })


# No longer needed: structured outputs replace JSON extraction


TOOLS = [request_confirmation, run_test, summarize_results]


# System prompt
SYSTEM_PROMPT = """You are an Evaluation Agent for Seer.

The system will automatically generate the AgentSpec and test cases from the user's request and provide an EVAL_CONTEXT system message containing:
- agent_url, agent_id
- test_count and indexed test_inputs

Your role now:
1. Request user confirmation exactly once using request_confirmation(question, test_count)
2. When the user confirms, sequentially call run_test(target_url, target_agent_id, test_input) for each test_input shown in EVAL_CONTEXT
   - The system will judge each result automatically; do NOT judge yourself
3. After all tests are run and judged, call summarize_results(passed, failed, total)

IMPORTANT:
- Do NOT attempt to call generate_spec, generate_tests, or any judging tool
- Use only the tools provided: request_confirmation, run_test, summarize_results
- Be concise and efficient"""


def build_graph():
    """Build the eval agent graph with LLM nodes for spec/tests/judging and tool nodes for confirmation/run/summarize."""

    llm_for_agent = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    ).bind_tools(TOOLS)

    def route_entry(state: AgentState):
        """No-op node used to route based on presence of an existing eval_suite in state."""
        return {}

    def agent_node(state: AgentState):
        """Main conversational agent node (confirmation, running tests, summarizing)."""
        # Prepend system prompt and any EVAL_CONTEXT/PROGRESS messages already in state
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_for_agent.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        """If the last agent message requested a tool, go to tools; otherwise end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def route_from_entry(state: AgentState):
        """If eval_suite already exists in state, skip generation and go directly to agent."""
        if state.get("eval_suite") is not None:
            return "agent"
        return "parse_request"

    def route_after_tool(state: AgentState):
        """After a tool runs, if it was run_test then judge; otherwise return to agent."""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, ToolMessage) and getattr(last_msg, "name", "") == "run_test":
            return "judge"
        return "agent"

    workflow = StateGraph(AgentState)
    workflow.add_node("route", route_entry)
    workflow.add_node("parse_request", parse_request_node)
    workflow.add_node("generate_spec", generate_spec_node)
    workflow.add_node("generate_tests", generate_tests_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("judge", judge_node)

    workflow.set_entry_point("route")
    workflow.add_conditional_edges("route", route_from_entry)
    workflow.add_edge("parse_request", "generate_spec")
    workflow.add_edge("generate_spec", "generate_tests")
    workflow.add_edge("generate_tests", "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_conditional_edges("tools", route_after_tool)
    workflow.add_edge("judge", "agent")

    return workflow.compile()


# Create graph instance
graph = build_graph()

