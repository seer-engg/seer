"""
Unit tests for Nexus workflow generation with structured outputs.

Tests the create_workflow_spec_structured function and WorkflowProposal model.
"""
# pylint: disable=import-outside-toplevel  # Reason: Test-specific imports are acceptable
import pytest
from pydantic import ValidationError
from seer.agents.nexus.tools.workflow_tools import (
    WorkflowProposal,
    create_workflow_spec_structured,
)
from seer.core.schema.models import WorkflowSpec


class TestWorkflowProposal:
    """Test WorkflowProposal model validation."""

    def test_valid_proposal(self):
        """Test valid WorkflowProposal creation."""
        spec = WorkflowSpec(
            version="2",
            nodes=[
                {
                    "id": "test_node",
                    "type": "task",
                    "kind": "set",
                    "value": {"result": "test"},
                }
            ],
            edges=[],
            triggers=[],
        )

        proposal = WorkflowProposal(
            spec=spec,
            summary="Test workflow",
            reasoning="Testing purposes"
        )

        assert proposal.spec.version == "2"  # pylint: disable=no-member  # Reason: Pydantic FieldInfo internals
        assert len(proposal.spec.nodes) == 1  # pylint: disable=no-member  # Reason: Pydantic FieldInfo internals
        assert proposal.summary == "Test workflow"
        assert proposal.reasoning == "Testing purposes"

    def test_proposal_requires_all_fields(self):
        """Test that WorkflowProposal requires all fields."""
        spec = WorkflowSpec(version="2", nodes=[], edges=[], triggers=[])

        # Missing summary
        with pytest.raises(ValidationError) as exc_info:
            WorkflowProposal(spec=spec, reasoning="Test")  # type: ignore

        assert "summary" in str(exc_info.value)

        # Missing reasoning
        with pytest.raises(ValidationError) as exc_info:
            WorkflowProposal(spec=spec, summary="Test")  # type: ignore

        assert "reasoning" in str(exc_info.value)

    def test_proposal_validates_spec(self):
        """Test that WorkflowProposal validates the workflow spec."""
        # Invalid spec (wrong version format)
        with pytest.raises(ValidationError):
            WorkflowProposal(
                spec={"version": 2, "nodes": [], "edges": [], "triggers": []},  # type: ignore  # version should be string
                summary="Test",
                reasoning="Test"
            )

        # Invalid spec (invalid node type)
        with pytest.raises(ValidationError):
            WorkflowProposal(
                spec={
                    "version": "2",
                    "nodes": [{"id": "bad", "type": "invalid_type"}],
                    "edges": [],
                    "triggers": []
                },  # type: ignore
                summary="Test",
                reasoning="Test"
            )


@pytest.mark.asyncio
class TestCreateWorkflowSpecStructured:
    """Test create_workflow_spec_structured function with real LLM."""

    async def test_generates_valid_spec_simple_workflow(self):
        """Test generating a simple workflow with structured output."""
        # This test uses mock LLM to avoid API calls in unit tests
        # The mock should return a valid WorkflowProposal structure
        from unittest.mock import MagicMock  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import

        # Create mock LLM that returns valid proposal
        mock_llm_instance = MagicMock()
        mock_structured = MagicMock()

        # Mock a valid proposal response
        mock_proposal = WorkflowProposal(
            spec=WorkflowSpec(
                version="2",
                nodes=[
                    {
                        "id": "fetch_data",
                        "type": "tool",
                        "tool": "demo.fetch",
                        "inputs": {"query": "test"},
                    }
                ],
                edges=[],
                triggers=[],
            ),
            summary="Fetch data workflow",
            reasoning="Uses demo fetch tool to get data"
        )

        mock_structured.invoke = MagicMock(return_value=mock_proposal)
        mock_llm_instance.with_structured_output = MagicMock(return_value=mock_structured)

        # Test the function
        proposal = create_workflow_spec_structured(
            llm=mock_llm_instance,
            user_intent="Fetch some data",
            discovered_tools=[{"tool": "demo.fetch", "description": "Fetch data"}],
            discovered_triggers=[],
        )

        # Validate result
        assert isinstance(proposal, WorkflowProposal)
        assert isinstance(proposal.spec, WorkflowSpec)
        assert proposal.spec.version == "2"
        assert len(proposal.spec.nodes) > 0
        assert proposal.summary
        assert proposal.reasoning

        # Verify with_structured_output was called with correct model
        mock_llm_instance.with_structured_output.assert_called_once()
        call_args = mock_llm_instance.with_structured_output.call_args
        assert call_args[0][0] == WorkflowProposal
        assert call_args[1]["method"] == "function_calling"


@pytest.mark.integration
@pytest.mark.asyncio
class TestStructuredOutputIntegration:
    """Integration tests with real LLM calls (requires API key)."""

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_real_llm_simple_workflow(self):
        """Test with real LLM for simple workflow generation."""
        from seer.llm import get_llm_without_responses_api

        llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)

        proposal = create_workflow_spec_structured(
            llm=llm,
            user_intent="Send a test email",
            discovered_tools=[
                {"tool": "gmail_send_email", "description": "Send email via Gmail"}
            ],
            discovered_triggers=[],
        )

        # Validate structure
        assert isinstance(proposal, WorkflowProposal)
        assert proposal.spec.version == "2"
        assert len(proposal.spec.nodes) >= 1
        assert any(node.type == "tool" for node in proposal.spec.nodes)
        assert proposal.summary
        assert proposal.reasoning

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_real_llm_trigger_workflow(self):
        """Test with real LLM for trigger-based workflow."""
        from seer.llm import get_llm_without_responses_api

        llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)

        proposal = create_workflow_spec_structured(
            llm=llm,
            user_intent="Send welcome email when user signs up",
            discovered_tools=[
                {"tool": "gmail_send_email", "description": "Send email via Gmail"}
            ],
            discovered_triggers=[
                {"key": "webhook.supabase.db_changes", "description": "Supabase table change"}
            ],
        )

        # Validate structure
        assert isinstance(proposal, WorkflowProposal)
        assert proposal.spec.version == "2"
        assert len(proposal.spec.triggers) >= 1
        assert len(proposal.spec.nodes) >= 1
        assert len(proposal.spec.edges) >= 1

        # Check trigger edge exists
        trigger_edges = [e for e in proposal.spec.edges if e.type == "trigger"]
        assert len(trigger_edges) >= 1
