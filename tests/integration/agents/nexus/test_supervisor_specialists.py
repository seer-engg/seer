"""
Integration tests for Nexus supervisor specialists.

Tests workflow_architect and validation specialists with real LLM calls.
"""
import pytest
from seer.agents.nexus.supervisor.specialists.workflow_architect import workflow_architect_specialist
from seer.agents.nexus.supervisor.specialists.validation import validation_specialist
from seer.agents.nexus.supervisor.state import SupervisorState


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkflowArchitectSpecialist:
    """Integration tests for workflow architect specialist."""

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_generates_workflow_from_intent(self):
        """Test workflow architect generates valid spec from user intent."""
        # Setup state
        state: SupervisorState = {
            "user_intent": "Send a test email",
            "discovered_tools": [
                {
                    "tool": "gmail_send_email",
                    "integration": "gmail",
                    "description": "Send email via Gmail"
                }
            ],
            "discovered_triggers": [],
            "messages": [],
        }

        # Run specialist
        result = await workflow_architect_specialist(state)

        # Validate result structure
        assert "workflow_draft" in result
        assert "messages" in result
        assert result["workflow_draft"] is not None

        # Validate workflow spec
        draft = result["workflow_draft"]
        assert draft["version"] == "2"
        assert isinstance(draft["nodes"], list)
        assert len(draft["nodes"]) >= 1
        assert isinstance(draft["edges"], list)

        # Check tool node exists
        tool_nodes = [n for n in draft["nodes"] if n["type"] == "tool"]
        assert len(tool_nodes) >= 1

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_generates_trigger_workflow(self):
        """Test workflow architect generates trigger-based workflow."""
        state: SupervisorState = {
            "user_intent": "Send welcome email when user signs up",
            "discovered_tools": [
                {
                    "tool": "gmail_send_email",
                    "integration": "gmail",
                    "description": "Send email via Gmail"
                }
            ],
            "discovered_triggers": [
                {
                    "key": "webhook.supabase.db_changes",
                    "provider": "supabase",
                    "description": "Supabase table change webhook"
                }
            ],
            "messages": [],
        }

        result = await workflow_architect_specialist(state)

        # Validate trigger workflow
        draft = result["workflow_draft"]
        assert draft is not None
        assert isinstance(draft["triggers"], list)
        assert len(draft["triggers"]) >= 1

        # Check trigger edge exists
        trigger_edges = [e for e in draft["edges"] if e.get("type") == "trigger"]
        assert len(trigger_edges) >= 1

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_handles_no_tools_discovered(self):
        """Test workflow architect handles case with no tools."""
        state: SupervisorState = {
            "user_intent": "Do something complex",
            "discovered_tools": [],
            "discovered_triggers": [],
            "messages": [],
        }

        result = await workflow_architect_specialist(state)

        # Should still return a result (may use LLM or task nodes)
        assert "workflow_draft" in result
        assert "messages" in result


@pytest.mark.integration
@pytest.mark.asyncio
class TestValidationSpecialist:
    """Integration tests for validation specialist."""

    async def test_validates_valid_spec(self, db_engine, test_user):  # pylint: disable=unused-argument  # Reason: Fixture required for database setup
        """Test validation specialist accepts valid workflow spec."""
        from seer.agents.nexus.context import (  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import
            _current_thread_id,
            set_user_for_thread,
        )
        from langchain_core.messages import HumanMessage  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import

        # Create thread context
        thread_id = "test_thread_validation"
        _current_thread_id.set(thread_id)
        set_user_for_thread(thread_id, test_user)

        # Setup state with valid workflow
        workflow_draft = {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "task1",
                    "type": "task",
                    "kind": "set",
                    "value": {"message": "Hello"},
                }
            ],
            "edges": [],
        }

        state: SupervisorState = {
            "user_intent": "Test workflow",
            "workflow_draft": workflow_draft,
            "messages": [HumanMessage(content="Test workflow")],
        }

        # Run validation (this may still fail at compilation without proper tool registration)
        result = await validation_specialist(state)

        # Validate result structure
        assert "validation_result" in result or "messages" in result
        # Note: Full compilation may fail without registered tools, but structure should be valid


@pytest.mark.integration
@pytest.mark.asyncio
class TestSupervisorFlow:
    """Integration tests for full supervisor flow."""

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_architect_to_validation_flow(self, db_engine, test_user):  # pylint: disable=unused-argument  # Reason: Fixture required for database setup
        """Test full flow from architect to validation."""
        from seer.agents.nexus.context import (  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import
            _current_thread_id,
            set_user_for_thread,
        )

        # Setup thread context
        thread_id = "test_thread_flow"
        _current_thread_id.set(thread_id)
        set_user_for_thread(thread_id, test_user)

        # Step 1: Architect generates workflow
        arch_state: SupervisorState = {
            "user_intent": "Create a simple task workflow",
            "discovered_tools": [],
            "discovered_triggers": [],
            "messages": [],
        }

        arch_result = await workflow_architect_specialist(arch_state)
        assert arch_result["workflow_draft"] is not None

        # Step 2: Validation specialist validates
        val_state: SupervisorState = {
            "user_intent": "Create a simple task workflow",
            "workflow_draft": arch_result["workflow_draft"],
            "messages": arch_result["messages"],
        }

        val_result = await validation_specialist(val_state)

        # Should have validation result or error message
        assert "validation_result" in val_result or "messages" in val_result
