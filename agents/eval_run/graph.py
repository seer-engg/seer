"""Builds the test execution subgraph for a batch of dataset examples.
This subgraph:
- dispatches each example one-by-one to the per-example pipeline (provision → invoke → assert → prepare_result)
- accumulates results
- returns latest_results compatible with EvalAgentState
"""
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from agents.eval_run.provision_env import provision_environment_node
from agents.eval_run.verify_provisioning import verify_provisioning_node
from agents.eval_run.invoke_target import invoke_target_node
from agents.eval_run.assert_state import assert_final_state_node
from agents.eval_run.prepare_result import prepare_result_node
from agents.eval_run.initialize import initialize_node
from agents.eval_run.seed_mcp_resources import seed_mcp_resources
from agents.eval_run.clean_mcp_resources import clean_mcp_resources
from langchain_core.messages import AIMessage
import uuid

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

async def generate_response(state: TestExecutionState) -> dict:
    """Generate a response for the test execution."""
    output_messages = [AIMessage(content=f"test execution completed with {len(state.latest_results)} results")]
    return {
        "messages": output_messages,
    }

def build_test_execution_subgraph():
    """Build the batch-aware test execution subgraph."""
    builder = StateGraph(TestExecutionState)
    # Batch routing
    builder.add_node("initialize", initialize_node)
    builder.add_node("dispatch", dispatch_examples_node)
    builder.add_node("finalize_batch", finalize_batch_node)
    # Per-example pipeline
    builder.add_node("seed_mcp_resources", seed_mcp_resources)
    builder.add_node("clean_mcp_resources", clean_mcp_resources)
    builder.add_node("provision", provision_environment_node)
    builder.add_node("verify_provisioning", verify_provisioning_node)
    builder.add_node("invoke", invoke_target_node)
    builder.add_node("assert", assert_final_state_node)
    builder.add_node("prepare_result", prepare_result_node)
    builder.add_node("generate_response", generate_response)

    # Start by dispatching the first/next example
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "seed_mcp_resources")
    builder.add_edge("seed_mcp_resources", "dispatch")
    builder.add_conditional_edges("dispatch", _route_from_dispatch, {
        "provision": "provision",
        "finalize_batch": "finalize_batch",
    })

    builder.add_edge("provision", "verify_provisioning")
    
    def route_after_verification(state: TestExecutionState):
        """Route based on provisioning verification result."""
        verification = state.provisioning_verification
        if verification and verification.get("provisioning_succeeded", False):
            return "invoke"  # Proceed to target agent
        else:
            # Provisioning failed - skip target agent, go straight to prepare_result
            logger.warning(f"Provisioning verification failed - skipping target agent invocation")
            return "prepare_result"
    
    builder.add_conditional_edges("verify_provisioning", route_after_verification, {
        "invoke": "invoke",
        "prepare_result": "prepare_result",
    })
    
    builder.add_conditional_edges("invoke", agent_invocation_route_node, {
        "prepare_result": "prepare_result",
        "assert": "assert",
    })

    builder.add_edge("assert", "prepare_result")
    builder.add_edge("prepare_result", "dispatch")

    # Finish when dispatch decides no more examples remain
    builder.add_edge("finalize_batch", "clean_mcp_resources")
    builder.add_edge("clean_mcp_resources", "generate_response")
    builder.add_edge("generate_response", END)
    return builder.compile()


