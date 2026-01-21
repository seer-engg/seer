"""Unit tests for supervisor multi-agent architecture."""
import pytest
from langchain_core.messages import HumanMessage
from seer.agents.nexus.supervisor.state import SupervisorState
from seer.agents.nexus.supervisor.router import supervisor_router


class TestSupervisorArchitecture:
    """Test supervisor routing and specialist coordination."""

    @pytest.mark.asyncio
    async def test_supervisor_routing_tool_discovery(self):
        """Test supervisor routes tool discovery queries correctly."""
        state: SupervisorState = {
            "messages": [HumanMessage(content="What tools can send email?")],
            "discovered_tools": None,
            "discovered_triggers": None,
            "workflow_draft": None,
            "validation_result": None,
            "current_specialist": None,
            "workflow_complete": False,
            "user_intent": "What tools can send email?",
            "workflow_state": None
        }

        next_agent = await supervisor_router(state)
        assert next_agent in ["tool_discovery", "FINISH"]

    @pytest.mark.asyncio
    async def test_supervisor_routing_workflow_creation(self):
        """Test supervisor handles full workflow creation flow."""
        # Initial state - should route to tool discovery or trigger discovery
        state: SupervisorState = {
            "messages": [HumanMessage(content="Create a draft when someone signs up")],
            "discovered_tools": None,
            "discovered_triggers": None,
            "workflow_draft": None,
            "validation_result": None,
            "current_specialist": None,
            "workflow_complete": False,
            "user_intent": "Create a draft when someone signs up",
            "workflow_state": None
        }

        next_agent = await supervisor_router(state)
        assert next_agent in ["tool_discovery", "trigger_discovery"]

        # After tools discovered - should route to trigger discovery, architect, or continue tool discovery
        state["discovered_tools"] = [{"tool": "gmail_create_draft", "confidence": 95}]
        next_agent = await supervisor_router(state)
        assert next_agent in ["tool_discovery", "trigger_discovery", "workflow_architect"]

        # After triggers discovered - should route to architect (or may continue discovery)
        state["discovered_triggers"] = [{"key": "webhook.supabase.db_changes", "provider": "supabase"}]
        next_agent = await supervisor_router(state)
        # LLM routing may vary; accept any valid specialist
        assert next_agent in ["tool_discovery", "trigger_discovery", "workflow_architect"]

        # After draft created - should route to validation
        state["workflow_draft"] = {"nodes": [], "edges": []}
        next_agent = await supervisor_router(state)
        assert next_agent == "validation"

        # After validation succeeds - should finish
        state["validation_result"] = {"status": "ok"}
        state["workflow_complete"] = True
        next_agent = await supervisor_router(state)
        assert next_agent == "FINISH"

    @pytest.mark.asyncio
    async def test_supervisor_routing_validation_error(self):
        """Test supervisor handles validation errors correctly."""
        state: SupervisorState = {
            "messages": [HumanMessage(content="Create workflow")],
            "discovered_tools": [{"tool": "test_tool"}],
            "discovered_triggers": None,
            "workflow_draft": {"nodes": []},
            "validation_result": {"status": "error", "message": "Tool not found"},
            "current_specialist": None,
            "workflow_complete": False,
            "user_intent": None,
            "workflow_state": None
        }

        # Should route back to architect to fix the error
        next_agent = await supervisor_router(state)
        assert next_agent in ["workflow_architect", "tool_discovery"]

    @pytest.mark.asyncio
    async def test_supervisor_finish_when_complete(self):
        """Test supervisor finishes when workflow is complete."""
        state: SupervisorState = {
            "messages": [],
            "discovered_tools": None,
            "discovered_triggers": None,
            "workflow_draft": None,
            "validation_result": None,
            "current_specialist": None,
            "workflow_complete": True,
            "user_intent": None,
            "workflow_state": None
        }

        next_agent = await supervisor_router(state)
        assert next_agent == "FINISH"


class TestSupervisorState:
    """Test supervisor state schema."""

    def test_state_schema(self):
        """Test state schema is properly typed."""
        state: SupervisorState = {
            "messages": [],
            "discovered_tools": None,
            "discovered_triggers": None,
            "workflow_draft": None,
            "validation_result": None,
            "current_specialist": None,
            "workflow_complete": False,
            "user_intent": None,
            "workflow_state": None
        }

        assert isinstance(state["messages"], list)
        assert state["workflow_complete"] is False
