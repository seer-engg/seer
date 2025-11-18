"""Builds the per-example test execution subgraph: provision → invoke → assert."""
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from agents.eval_agent.nodes.execute.provision_env import provision_environment_node
from agents.eval_agent.nodes.execute.invoke_target import invoke_target_node
from agents.eval_agent.nodes.execute.assert_state import assert_final_state_node
from agents.eval_agent.nodes.execute.prepare_result import prepare_result_node

logger = get_logger("eval_agent.execute.graph")


def build_test_execution_subgraph():
    """Build the per-example test execution subgraph."""
    builder = StateGraph(TestExecutionState)
    builder.add_node("provision", provision_environment_node)
    builder.add_node("invoke", invoke_target_node)
    builder.add_node("assert", assert_final_state_node)
    builder.add_node("prepare_result", prepare_result_node)

    builder.add_edge(START, "provision")
    builder.add_edge("provision", "invoke")
    builder.add_edge("invoke", "assert")
    builder.add_edge("assert", "prepare_result")
    builder.add_edge("prepare_result", END)

    return builder.compile()


