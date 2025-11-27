"""Builds the test execution subgraph for a batch of dataset examples.
This subgraph:
- dispatches each example one-by-one to the per-example pipeline (provision → invoke → assert → prepare_result)
- accumulates results
- returns latest_results compatible with EvalAgentState
"""
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from agents.eval_agent.nodes.execute.provision_env import provision_environment_node
from agents.eval_agent.nodes.execute.invoke_target import invoke_target_node
from agents.eval_agent.nodes.execute.assert_state import assert_final_state_node
from agents.eval_agent.nodes.execute.prepare_result import prepare_result_node
from agents.eval_agent.nodes.execute.initialize import initialize_node

logger = get_logger("eval_agent.execute.graph")


async def dispatch_examples_node(state: TestExecutionState) -> dict:
    """
    Initialize and dispatch the next dataset example into the per-example pipeline.
    Also enrich mcp_resources with github_owner and github_repo on first initialization.
    """
    updates: dict = {}    

    # Pick next example if available
    pending_examples =  state.pending_examples
    if pending_examples:
        next_example = pending_examples.pop(0)
        updates["dataset_example"] = next_example
        updates["pending_examples"] = pending_examples
        return updates

    # No examples to run - finalize will handle producing latest_results
    updates["dataset_example"] = None
    return updates


def _route_from_dispatch(state: TestExecutionState):
    """Decide whether to run one example or finalize the batch."""
    return "provision" if state.dataset_example is not None else "finalize_batch"

async def finalize_batch_node(state: TestExecutionState) -> dict:
    """Emit latest_results aligned to EvalAgentState from the accumulated results."""
    return {
        "latest_results": list([state.result]),
    }

async def agent_invocation_route_node(state: TestExecutionState) -> dict:
    """Route the agent invocation based on the assertion output."""
    if state.assertion_output:
        return "prepare_result"
    else:
        return "assert"


def build_test_execution_subgraph():
    """Build the batch-aware test execution subgraph."""
    builder = StateGraph(TestExecutionState)
    # Batch routing
    builder.add_node("initialize", initialize_node)
    builder.add_node("dispatch", dispatch_examples_node)
    builder.add_node("finalize_batch", finalize_batch_node)
    # Per-example pipeline
    builder.add_node("provision", provision_environment_node)
    builder.add_node("invoke", invoke_target_node)
    builder.add_node("assert", assert_final_state_node)
    builder.add_node("prepare_result", prepare_result_node)

    # Start by dispatching the first/next example
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "dispatch")
    builder.add_conditional_edges("dispatch", _route_from_dispatch, {
        "provision": "provision",
        "finalize_batch": "finalize_batch",
    })

    builder.add_edge("provision", "invoke")
    builder.add_conditional_edges("invoke", agent_invocation_route_node, {
        "prepare_result": "prepare_result",
        "assert": "assert",
    })

    builder.add_edge("assert", "prepare_result")
    builder.add_edge("prepare_result", "dispatch")

    # Finish when dispatch decides no more examples remain
    builder.add_edge("finalize_batch", END)

    return builder.compile()


