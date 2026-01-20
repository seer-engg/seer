"""LangGraph construction for supervisor multi-agent architecture."""
from typing import Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from seer.agents.nexus.supervisor.state import SupervisorState
from seer.agents.nexus.supervisor.router import supervisor_router
from seer.agents.nexus.supervisor.specialists.tool_discovery import tool_discovery_specialist
from seer.agents.nexus.supervisor.specialists.trigger_discovery import trigger_discovery_specialist
from seer.agents.nexus.supervisor.specialists.workflow_architect import workflow_architect_specialist
from seer.agents.nexus.supervisor.specialists.validation import validation_specialist


def create_supervisor_graph(
    model: str = "gpt-4o-mini",  # pylint: disable=unused-argument  # Reserved for future specialist model configuration
    checkpointer: Optional[BaseCheckpointSaver] = None,
    workflow_state: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Create supervisor-style multi-agent graph for Nexus.

    Args:
        model: LLM model to use for specialists
        checkpointer: Optional checkpoint saver for persistence
        workflow_state: Optional existing workflow state for editing

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize state graph
    workflow = StateGraph(SupervisorState)

    # Add specialist nodes
    workflow.add_node("tool_discovery", tool_discovery_specialist)
    workflow.add_node("trigger_discovery", trigger_discovery_specialist)
    workflow.add_node("workflow_architect", workflow_architect_specialist)
    workflow.add_node("validation", validation_specialist)
    workflow.add_node("supervisor", lambda state: {"current_specialist": "supervisor"})

    # Conditional routing from START
    async def route_from_start(state: SupervisorState) -> str:
        """Route from start based on initial state."""
        # Set initial context
        if workflow_state:
            state["workflow_state"] = workflow_state

        # Extract user intent from first message
        if state.get("messages") and len(state["messages"]) > 0:
            first_msg = state["messages"][0]
            if hasattr(first_msg, "content"):
                state["user_intent"] = first_msg.content

        # Use supervisor to decide first step
        return await supervisor_router(state)

    # Conditional routing from supervisor
    async def route_from_supervisor(state: SupervisorState) -> str:
        """Route from supervisor to next specialist or finish."""
        return await supervisor_router(state)

    # Add conditional edges
    workflow.add_conditional_edges(
        START,
        route_from_start,
        {
            "tool_discovery": "tool_discovery",
            "trigger_discovery": "trigger_discovery",
            "workflow_architect": "workflow_architect",
            "validation": "validation",
            "FINISH": END
        }
    )

    # Each specialist routes back to supervisor for next decision
    for specialist in ["tool_discovery", "trigger_discovery", "workflow_architect", "validation"]:
        workflow.add_conditional_edges(
            specialist,
            route_from_supervisor,
            {
                "tool_discovery": "tool_discovery",
                "trigger_discovery": "trigger_discovery",
                "workflow_architect": "workflow_architect",
                "validation": "validation",
                "FINISH": END
            }
        )

    # Compile and return
    return workflow.compile(checkpointer=checkpointer)
