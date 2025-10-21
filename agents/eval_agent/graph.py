"""Simplified Eval Agent - Using BaseAgent with specialized nodes"""

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

from seer.shared.base_agent import BaseAgent, BaseAgentState
from seer.shared.agent_tools import run_test
from seer.shared.schemas import AgentSpec, TestCase, EvalSuite, TestResult
from seer.agents.eval_agent.prompts import (
    EVAL_AGENT_PROMPT,
    EVAL_AGENT_SPEC_PROMPT,
    EVAL_AGENT_TEST_GEN_PROMPT,
    EVAL_AGENT_JUDGE_PROMPT
)
from seer.shared.llm import get_llm


class EvalAgentState(BaseAgentState):
    """Extended state for eval agent with eval-specific fields"""
    agent_name: Optional[str] = None
    agent_url: Optional[str] = None
    expectations: Optional[str] = None
    spec: Optional[AgentSpec] = None
    eval_suite: Optional[EvalSuite] = None
    current_test_index: int = 0
    passed: int = 0
    failed: int = 0
    total: int = 0
    test_results: List[TestResult] = []


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


class EvalAgent:
    """Evaluation agent with specialized workflow nodes"""
    
    def __init__(self):
        self.agent_name = "eval_agent"
        self.system_prompt = EVAL_AGENT_PROMPT
        self.tools = [run_test]  # Only run_test - no callback tools!
    
    # Capabilities removed (unused)
    
    def build_graph(self):
        """Build eval agent with specialized nodes"""
        llm_for_agent = get_llm(temperature=0.3).bind_tools(self.tools)

        def route_entry(state: EvalAgentState):
            """Route based on presence of eval_suite in state"""
            return {}

        def parse_request_node(state: EvalAgentState):
            """Extract agent info from user message"""
            last_human = None
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    last_human = m
                    break
            user_text = last_human.content if last_human else ""

            extractor = get_llm(temperature=0.0).with_structured_output(EvalRequest)

            instruction = (
                "Extract agent_name, agent_url, and expectations from the user's latest message.\n\n"
                "IMPORTANT - agent_name is the assistant_id used by LangGraph:\n"
                "- If the message contains 'ID: <uuid>', use the graph name that corresponds to it\n"
                "- If the message contains 'Name: X', use X as the agent_name\n"
                "- If both are present like '(Name: Deep Researcher, ID: uuid)', prefer the Name\n"
                "- LangGraph accepts either the UUID OR the registered graph name\n"
                "- Common formats: 'Deep Researcher', 'my_agent', 'chat_agent', etc.\n\n"
                "Extract the agent_url from patterns like 'localhost:2024' or 'http://localhost:2024'.\n"
                "Always include the full original message as expectations."
            )
            request: EvalRequest = extractor.invoke(f"{instruction}\n\nUSER:\n{user_text}")

            return {
                "agent_name": request.agent_name,
                "agent_url": request.agent_url,
                "expectations": request.expectations,
            }

        def generate_spec_node(state: EvalAgentState):
            """Generate AgentSpec from expectations"""
            llm = get_llm().with_structured_output(AgentSpec)

            prompt = EVAL_AGENT_SPEC_PROMPT.format(
                expectations=state.get("expectations", ""),
                agent_name=state.get("agent_name", ""),
                agent_url=state.get("agent_url", ""),
            )

            spec: AgentSpec = llm.invoke(prompt)
            return {"spec": spec}

        def generate_tests_node(state: EvalAgentState):
            """Generate test cases from AgentSpec"""
            spec: AgentSpec = state.get("spec")
            if spec is None:
                return {}

            llm = get_llm().with_structured_output(GeneratedTests)

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

            # Provide context message for agent node
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

        def agent_node(state: EvalAgentState):
            """Main conversational agent node"""
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            response = llm_for_agent.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: EvalAgentState):
            """Check if we should continue to tools or end"""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        def route_from_entry(state: EvalAgentState):
            """If eval_suite already exists, skip generation and go to agent"""
            if state.get("eval_suite") is not None:
                return "agent"
            return "parse_request"

        def route_after_tool(state: EvalAgentState):
            """After run_test, go to judge; otherwise return to agent"""
            last_msg = state["messages"][-1]
            if isinstance(last_msg, ToolMessage) and getattr(last_msg, "name", "") == "run_test":
                return "judge"
            return "agent"

        def judge_node(state: EvalAgentState):
            """Judge the last run_test result"""
            eval_suite: EvalSuite = state.get("eval_suite")
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

            judge_llm = get_llm().with_structured_output(JudgeVerdict)

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

        # Create graph
        workflow = StateGraph(EvalAgentState)
        workflow.add_node("route", route_entry)
        workflow.add_node("parse_request", parse_request_node)
        workflow.add_node("generate_spec", generate_spec_node)
        workflow.add_node("generate_tests", generate_tests_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
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


# Create agent instance
agent = EvalAgent()

# Create the graph instance for langgraph dev
graph = agent.build_graph()
