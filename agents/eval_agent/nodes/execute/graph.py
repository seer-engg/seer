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
from shared.parameter_population import extract_all_context_variables

logger = get_logger("eval_agent.execute.graph")


async def dispatch_examples_node(state: TestExecutionState) -> dict:
    """
    Initialize and dispatch the next dataset example into the per-example pipeline.
    Also enrich mcp_resources with github_owner and github_repo on first initialization.
    """
    updates: dict = {}

    # Initialize pending list and accumulator on first entry
    if not state.pending_examples:
        pending = list(state.dataset_examples or [])
        updates["pending_examples"] = pending
        updates["accumulated_results"] = []

        # Enrich mcp_resources once using context variables (if github context is present)
        enriched_resources = dict(state.mcp_resources or {})
        if state.context.github_context and state.context.github_context.repo_url:
            context_vars = extract_all_context_variables(
                user_context=state.context.user_context,
                github_context=state.context.github_context,
                mcp_resources=enriched_resources,
            )
            if "github_owner" in context_vars:
                enriched_resources["github_owner"] = {"id": context_vars["github_owner"]}
                logger.info(f"Added github_owner to mcp_resources: {context_vars['github_owner']}")
            if "github_repo" in context_vars:
                enriched_resources["github_repo"] = {"id": context_vars["github_repo"]}
                logger.info(f"Added github_repo to mcp_resources: {context_vars['github_repo']}")
        updates["mcp_resources"] = enriched_resources

    # Pick next example if available
    pending_examples = updates.get("pending_examples", state.pending_examples)
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


async def collect_result_node(state: TestExecutionState) -> dict:
    """Append the per-example result to the accumulator and clear current example fields."""
    if not state.result:
        raise ValueError("collect_result_node expects a 'result' produced by prepare_result_node")
    accumulated = list(state.accumulated_results or [])
    accumulated.append(state.result)
    # Clear per-example keys to prepare for next dispatch
    return {
        "accumulated_results": accumulated,
        "dataset_example": None,
        "thread_id": None,
        "agent_output": "",
        "analysis": None,
        "assertion_output": None,
        "provisioning_output": None,
        "started_at": None,
        "completed_at": None,
        "result": None,
    }


async def finalize_batch_node(state: TestExecutionState) -> dict:
    """Emit latest_results aligned to EvalAgentState from the accumulated results."""
    return {
        "latest_results": list(state.accumulated_results or []),
    }


def build_test_execution_subgraph():
    """Build the batch-aware test execution subgraph."""
    builder = StateGraph(TestExecutionState)
    # Batch routing
    builder.add_node("dispatch", dispatch_examples_node)
    builder.add_node("collect_result", collect_result_node)
    builder.add_node("finalize_batch", finalize_batch_node)
    # Per-example pipeline
    builder.add_node("provision", provision_environment_node)
    builder.add_node("invoke", invoke_target_node)
    builder.add_node("assert", assert_final_state_node)
    builder.add_node("prepare_result", prepare_result_node)

    # Start by dispatching the first/next example
    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", _route_from_dispatch, {
        "provision": "provision",
        "finalize_batch": "finalize_batch",
    })

    builder.add_edge("provision", "invoke")
    builder.add_edge("invoke", "assert")
    builder.add_edge("assert", "prepare_result")
    builder.add_edge("prepare_result", "collect_result")
    builder.add_edge("collect_result", "dispatch")

    # Finish when dispatch decides no more examples remain
    builder.add_edge("finalize_batch", END)

    return builder.compile()


