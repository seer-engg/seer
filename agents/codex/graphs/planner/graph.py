from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import PlannerState

from shared.logger import get_logger

from agents.codex.graphs.planner.nodes import initialize_project
from agents.codex.graphs.planner.nodes import context_and_plan_agent
from agents.codex.graphs.planner.nodes.raise_pr import raise_pr
from agents.codex.graphs.planner.nodes.deploy import deploy_service
from agents.codex.graphs.planner.nodes.test_server_ready import test_server_ready
from agents.codex.graphs.programmer.graph import graph as programmer_graph
from shared.schema import CodexInput, CodexOutput
from agents.codex.graphs.planner.nodes import finalize

logger = get_logger("codex.planner")


def _prepare_graph_state(state: PlannerState) -> PlannerState:
    state = dict(state)
    state.setdefault("messages", [])
    state.setdefault("autoAcceptPlan", True)
    return state

async def _run_programmer(state: PlannerState) -> PlannerState:
    programmer_input = {
        "updated_sandbox_context": state.updated_sandbox_context,
        "user_context": state.user_context,
        "github_context": state.github_context,
        "sandbox_context": state.sandbox_context,
        "dataset_context": state.dataset_context,
        "experiment_context": state.experiment_context,
        "dataset_examples": state.dataset_examples,
        "taskPlan": state.taskPlan,
        "target_agent_version": state.target_agent_version,
    }
    programmer_output = await programmer_graph.ainvoke(programmer_input)

    return {
        "pr_summary": programmer_output.get("pr_summary"),
        "updated_sandbox_context": programmer_output.get("updated_sandbox_context"),
        "success": programmer_output.get("success"),
        "messages": programmer_output.get("messages"),
    }


def is_server_ready(state: PlannerState) -> PlannerState:
    if state.server_running:
        return "context-plan-agent"
    else:
        return "end"

async def is_success(state: PlannerState) -> PlannerState:
    if state.success:
        return "raise-pr"
    else:
        return "end"


def compile_planner_graph():
    workflow = StateGraph(state_schema=PlannerState, input=CodexInput, output=CodexOutput)
    workflow.add_node("prepare-graph-state", _prepare_graph_state)
    workflow.add_node("initialize-project", initialize_project)
    # Combined context + planning agent
    workflow.add_node("context-plan-agent", context_and_plan_agent)
    # Removed separate context agent; using combined agent instead
    workflow.add_node("programmer", _run_programmer)
    workflow.add_node("raise-pr", raise_pr)
    workflow.add_node("deploy-service", deploy_service)
    workflow.add_node("test-server-ready", test_server_ready)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "prepare-graph-state")
    workflow.add_edge("prepare-graph-state", "initialize-project")
    workflow.add_edge("initialize-project", "test-server-ready")

    workflow.add_conditional_edges("test-server-ready", is_server_ready, {
        "context-plan-agent": "context-plan-agent",
        "end": END
    })

    workflow.add_edge("context-plan-agent", "programmer")
    workflow.add_conditional_edges("programmer", is_success, {
        "raise-pr": "raise-pr",
        "end": END
    })
    workflow.add_edge("raise-pr", "deploy-service")
    workflow.add_edge("deploy-service", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(debug=True)

graph = compile_planner_graph()