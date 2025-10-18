"""Eval Agent LangGraph - Deployable Graph"""

import os
import json
import hashlib
import uuid
from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages


# Simple state
class AgentState(TypedDict):
    """State for Eval agent"""
    messages: Annotated[list[BaseMessage], add_messages]


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


@tool
async def generate_spec(agent_name: str, agent_url: str, expectations: str) -> str:
    """
    Generate an agent specification from user expectations.
    
    Args:
        agent_name: Name of the agent
        agent_url: URL where agent is hosted
        expectations: User's natural language expectations
    
    Returns:
        JSON string of the generated spec
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    prompt = EVAL_AGENT_SPEC_PROMPT.format(
        expectations=expectations,
        agent_name=agent_name,
        agent_url=agent_url
    )
    
    response = await llm.ainvoke(prompt)
    spec_json = extract_json(response.content)
    
    # Validate
    spec = AgentSpec(**spec_json)
    return spec.model_dump_json(indent=2)


@tool
async def generate_tests(spec_json: str) -> str:
    """
    Generate test cases from an agent specification.
    
    Args:
        spec_json: JSON string of the AgentSpec
    
    Returns:
        JSON string of the generated EvalSuite
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    spec = AgentSpec(**json.loads(spec_json))
    
    prompt = EVAL_AGENT_TEST_GEN_PROMPT.format(spec_json=spec_json)
    response = await llm.ainvoke(prompt)
    tests_json = extract_json(response.content)
    
    # Create TestCase objects with IDs
    test_cases = []
    for idx, tc_dict in enumerate(tests_json["test_cases"]):
        content_hash = hashlib.md5(
            f"{tc_dict['expectation_ref']}{tc_dict['input_message']}".encode()
        ).hexdigest()[:8]
        
        test_case = TestCase(
            id=f"{spec.name}_{idx+1}_{content_hash}",
            **tc_dict
        )
        test_cases.append(test_case)
    
    eval_suite = EvalSuite(
        id=f"eval_{spec.name}_{uuid.uuid4().hex[:8]}",
        spec_name=spec.name,
        spec_version=spec.version,
        test_cases=test_cases
    )
    
    return eval_suite.model_dump_json(indent=2)


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


@tool
async def judge_test_result(test_input: str, expected_behavior: str, success_criteria: str, actual_output: str) -> str:
    """
    Judge whether a test result passes based on criteria.
    
    Args:
        test_input: The input that was sent
        expected_behavior: What the agent should do
        success_criteria: Specific criteria for success
        actual_output: What the agent actually returned
    
    Returns:
        JSON string with pass/fail and reasoning
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    prompt = EVAL_AGENT_JUDGE_PROMPT.format(
        input_message=test_input,
        expected_behavior=expected_behavior,
        success_criteria=success_criteria,
        actual_output=actual_output
    )
    
    response = await llm.ainvoke(prompt)
    result_json = extract_json(response.content)
    
    return json.dumps(result_json)


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


# Helper function
def extract_json(text: str) -> dict:
    """Extract JSON from LLM response"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()
    
    return json.loads(text)


TOOLS = [generate_spec, generate_tests, request_confirmation, run_test, judge_test_result, summarize_results]


# System prompt
SYSTEM_PROMPT = """You are an Evaluation Agent for Seer.

Your role:
1. Generate specifications from user expectations (use generate_spec)
2. Create test cases from specs (use generate_tests)
3. Request user confirmation before running tests (use request_confirmation)
4. Run tests against target agent via A2A (use run_test)
5. Judge test results (use judge_test_result)
6. Summarize results (use summarize_results)

Workflow:
- User provides agent URL and expectations
- Generate spec → generate tests → request confirmation ONCE → run each test → judge results → summarize

IMPORTANT: 
- Use request_confirmation only ONCE after generating tests
- After calling tools and seeing results, don't call the same tools again
- Be thorough but efficient"""


def build_graph():
    """Build the eval agent graph"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    ).bind_tools(TOOLS)
    
    def agent_node(state: AgentState):
        """Main agent node"""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Check if we should continue to tools or end"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# Create graph instance
graph = build_graph()

