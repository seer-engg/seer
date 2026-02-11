# pylint: disable=import-outside-toplevel
# Reason: Test-specific imports are acceptable
"""
Integration tests for Nexus workflow generation with structured outputs.

These tests require real LLM API access (OpenAI API key) and test the
create_workflow_spec_structured function with actual LLM calls.

Run with: pytest -m integration tests/integration/agents/nexus/
"""
import os

import pytest

from seer.agents.nexus.tools.workflow_tools import (
    WorkflowProposal,
    create_workflow_spec_structured,
)
from seer.core.schema.models import WorkflowSpec


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
class TestStructuredOutputIntegration:
    """Integration tests with real LLM calls (requires API key)."""

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ,
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
        "OPENAI_API_KEY" not in os.environ,
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
