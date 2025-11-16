"""overall graph for the eval agent"""
import json
from typing import Literal, Dict
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import BaseTool

from agents.eval_agent.constants import N_ROUNDS, N_VERSIONS
from agents.eval_agent.nodes.finalize import build_finalize_subgraph
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.plan import build_plan_subgraph
from agents.eval_agent.nodes.reflect.graph import reflect_node
from agents.eval_agent.nodes.run import build_run_subgraph
from shared.logger import get_logger
from shared.tool_service import get_tool_service
from shared.tools import canonicalize_tool_name
from shared.test_runner.variable_injection import inject_variables


logger = get_logger("eval_agent.graph")


# _resolve_tool_key removed - no longer needed with dynamic cleanup


async def cleanup_environment(state: EvalAgentState) -> dict:
    """
    Execute cleanup actions in reverse order (LIFO).
    
    This is fully dynamic - no hardcoded service logic. Cleanup actions
    are generated automatically during provisioning as inverse operations.
    """
    if not state.cleanup_stack:
        logger.info("cleanup_environment: No cleanup actions to execute.")
        return {}

    logger.info(f"cleanup_environment: Executing {len(state.cleanup_stack)} cleanup actions in LIFO order...")
    
    # Use ToolService for tool access
    tool_service = get_tool_service()
    await tool_service.initialize(state.context.mcp_services)
    tools_dict = tool_service.get_tools()
    
    # Execute in REVERSE order (LIFO - last created = first deleted)
    for idx, action in enumerate(reversed(state.cleanup_stack), 1):
        tool_name = canonicalize_tool_name(action.tool)
        
        if tool_name not in tools_dict:
            logger.warning(f"Cleanup tool not found: {tool_name}")
            continue
        
        try:
            params = json.loads(action.params or "{}")
            
            # Inject variables from mcp_resources if needed (e.g., [var:asana_project])
            params = inject_variables(params, {}, state.context.mcp_resources)
            
            logger.info(f"Cleanup {idx}/{len(state.cleanup_stack)}: {action.tool} with {params}")
            await tools_dict[tool_name].ainvoke(params)
            
        except Exception as e:
            # Log but continue - don't let one failure stop other cleanups
            logger.error(f"Cleanup failed for {action.tool}: {e}", exc_info=True)
    
    # Clear both cleanup stack and resources
    return {
        "cleanup_stack": [],
        "context": state.context.model_copy(update={"mcp_resources": {}})
    }


def update_state_from_handoff(state: EvalAgentState) -> dict:
    """
    Unpacks the codex_output handoff object into the main state for the next round.
    This is the single source of truth for updating state after a codex run.
    """
    codex_handoff = state.codex_output
    if not codex_handoff:
        raise ValueError("No codex handoff object found in state to update from.")

    logger.info("Updating state from codex handoff for the next evaluation round.")
    return {
        "target_agent_version": codex_handoff.target_agent_version,
        "sandbox_context": codex_handoff.updated_sandbox_context,
        "codex_output": None,  # Clear the handoff object after processing
    }


def should_continue(state: EvalAgentState) -> Literal["plan", "finalize"]:
    """Determine if the eval loop should continue plan or finalize."""
    return "plan" if state.attempts < N_ROUNDS else "finalize"


def should_start_new_round(state: EvalAgentState) -> Literal["update_state_from_handoff", "__end__"]:
    """
    Decision node based on the codex handoff object.
    If a valid handoff exists and we haven't hit the version limit, route to update state.
    """
    codex_handoff = state.codex_output
    if codex_handoff and codex_handoff.agent_updated and codex_handoff.target_agent_version < N_VERSIONS:
        logger.info(f"Codex provided an update to v{codex_handoff.target_agent_version}. Starting new evaluation round.")
        return "update_state_from_handoff"
    else:
        if not codex_handoff or not codex_handoff.agent_updated:
            logger.info("Codex did not provide an update. Ending workflow.")
        else:
            logger.info(f"Reached max versions ({N_VERSIONS}). Ending workflow.")
        return "__end__"


def build_graph():
    """Build the evaluation agent graph."""
    workflow = StateGraph(EvalAgentState)
    plan_subgraph = build_plan_subgraph()
    run_subgraph = build_run_subgraph()
    finalize_subgraph = build_finalize_subgraph()

    workflow.add_node("plan", plan_subgraph)
    workflow.add_node("run", run_subgraph)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("finalize", finalize_subgraph)
    workflow.add_node("update_state_from_handoff", update_state_from_handoff)
    workflow.add_node("cleanup", cleanup_environment) # ADDED

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "run")
    workflow.add_edge("run", "reflect")
    workflow.add_conditional_edges("reflect", should_continue, {
        "plan": "plan",
        "finalize": "finalize"
    })
    
    # MODIFIED: Finalize now goes to cleanup
    workflow.add_edge("finalize", "cleanup")
    
    # MODIFIED: Conditional logic moves to after cleanup
    workflow.add_conditional_edges("cleanup", should_start_new_round, {
        "update_state_from_handoff": "update_state_from_handoff",
        "__end__": END
    })
    workflow.add_edge("update_state_from_handoff", "plan")

    return workflow.compile(debug=True)


graph = build_graph()
