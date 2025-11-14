"""overall graph for the eval agent"""
import asyncio
from typing import Literal, Optional, Dict
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import BaseTool

from agents.eval_agent.constants import N_ROUNDS, N_VERSIONS
from agents.eval_agent.nodes.finalize import build_finalize_subgraph
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.plan import build_plan_subgraph
from agents.eval_agent.nodes.reflect.graph import reflect_node
from agents.eval_agent.nodes.run import build_run_subgraph
from shared.logger import get_logger
from shared.mcp_client import get_mcp_client_and_configs
from shared.tool_catalog import canonicalize_tool_name


logger = get_logger("eval_agent.graph")


def _resolve_tool_key(tools_dict: Dict[str, BaseTool], *candidates: str) -> str | None:
    for candidate in candidates:
        key = canonicalize_tool_name(candidate)
        if key in tools_dict:
            return key
    return None


async def cleanup_environment(state: EvalAgentState) -> dict:
    """
    Deletes the MCP resources (Asana project, GitHub repo)
    created for this evaluation experiment.
    """
    if not state.mcp_resources:
        logger.info("cleanup_environment: No MCP resources to clean up.")
        return {}

    logger.info(f"cleanup_environment: Cleaning up resources: {state.mcp_resources.keys()}")
    mcp_client, _ = await get_mcp_client_and_configs(state.mcp_services)
    mcp_tools = await mcp_client.get_tools()
    tools_dict: Dict[str, BaseTool] = {
        canonicalize_tool_name(t.name): t for t in mcp_tools
    }
    
    try:
        # 1. Delete Asana Project
        if "asana_project" in state.mcp_resources:
            delete_project = _resolve_tool_key(
                tools_dict,
                "ASANA_DELETE_PROJECT",
                "asana.delete_project",
            )
            if delete_project:
                project_id = state.mcp_resources["asana_project"].get("id")
                if project_id:
                    logger.info(f"Deleting Asana project: {project_id}")
                    await tools_dict[delete_project].ainvoke({"id": project_id})
        
        # 2. Delete GitHub Repo
        if "github_repo" in state.mcp_resources:
            delete_repo = _resolve_tool_key(
                tools_dict,
                "GITHUB_DELETE_REPOSITORY",
                "github.delete_repo",
                "GITHUB_DELETE_A_REPOSITORY",
            )
            if delete_repo:
                repo_full_name = state.mcp_resources["github_repo"].get("full_name")
                if repo_full_name:
                    logger.info(f"Deleting GitHub repo: {repo_full_name}")
                    await tools_dict[delete_repo].ainvoke({"full_name": repo_full_name})
                
    except Exception as e:
        # Log errors but don't stop the graph
        logger.error(f"cleanup_environment: Error during cleanup: {e}", exc_info=True)
    
    # Return an empty dict to clear the resources from the state
    return {"mcp_resources": {}}


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
