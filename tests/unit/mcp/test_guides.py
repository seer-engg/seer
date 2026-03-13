"""
Unit tests for MCP guide tools.

Tests the get_workflow_guide and get_started MCP tools that provide on-demand
access to workflow building documentation.
"""

import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestGetStarted:
    """Tests for get_started MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.get_nexus_system_prompt")
    async def test_returns_base_prompt(self, mock_prompt):
        """Test that get_started returns the base system prompt."""
        mock_prompt.return_value = "# Nexus Base\nCore principles..."

        from seer.mcp.tools.guides import get_started
        result = await get_started.fn()

        assert "Nexus Base" in result
        assert "Core principles" in result
        mock_prompt.assert_called_once()

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.get_nexus_system_prompt")
    async def test_impl_is_synchronous(self, mock_prompt):
        """Verify get_started_impl can be called synchronously from agent.py."""
        mock_prompt.return_value = "# System Prompt\nWorkflow assistant instructions"

        from seer.mcp.tools.guides import get_started_impl
        result = get_started_impl()

        assert isinstance(result, str)
        assert "System Prompt" in result
        mock_prompt.assert_called_once()

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.get_nexus_system_prompt")
    async def test_mcp_tool_uses_impl(self, mock_prompt):
        """Verify MCP tool delegates to sync implementation."""
        mock_prompt.return_value = "# Base Instructions"

        from seer.mcp.tools.guides import get_started, get_started_impl

        mcp_result = await get_started.fn()
        impl_result = get_started_impl()

        assert mcp_result == impl_result


@pytest.mark.unit
class TestGetWorkflowGuide:
    """Tests for get_workflow_guide MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.get_primitive_blocks_guide")
    @patch("seer.mcp.tools.guides.get_graph_structure_guide")
    @patch("seer.mcp.tools.guides.generate_trigger_reference")
    async def test_returns_combined_guide_by_default(self, mock_triggers, mock_graph, mock_blocks):
        """Test that combined guide is returned when no section specified."""
        mock_blocks.return_value = "# Blocks Guide\nTool nodes are..."
        mock_graph.return_value = "# Graph Guide\nEdge types include..."
        mock_triggers.return_value = "# Trigger Guide\nTriggers define when..."

        from seer.mcp.tools.guides import get_workflow_guide
        result = await get_workflow_guide.fn()

        assert "Blocks Guide" in result
        assert "Graph Guide" in result
        assert "Trigger Guide" in result
        mock_blocks.assert_called_once()
        mock_graph.assert_called_once()
        mock_triggers.assert_called_once()

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.get_primitive_blocks_guide")
    async def test_blocks_section(self, mock_blocks):
        """Test blocks section filter."""
        mock_blocks.return_value = "# Primitive Blocks\nTool nodes execute..."

        from seer.mcp.tools.guides import get_workflow_guide
        result = await get_workflow_guide.fn(section="blocks")

        assert "Primitive Blocks" in result
        mock_blocks.assert_called_once()

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.get_graph_structure_guide")
    async def test_graph_section(self, mock_graph):
        """Test graph section filter."""
        mock_graph.return_value = "# Graph Structure\nEdges connect nodes..."

        from seer.mcp.tools.guides import get_workflow_guide
        result = await get_workflow_guide.fn(section="graph")

        assert "Graph Structure" in result
        mock_graph.assert_called_once()

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.generate_trigger_reference")
    async def test_triggers_section(self, mock_triggers):
        """Test triggers section filter returns trigger specification docs."""
        mock_triggers.return_value = "### Trigger Specification\nRequired fields: id, key, mode"

        from seer.mcp.tools.guides import get_workflow_guide
        result = await get_workflow_guide.fn(section="triggers")

        assert "Trigger Specification" in result
        assert "id, key, mode" in result
        mock_triggers.assert_called_once()

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.list_available_skills")
    async def test_integration_list(self, mock_list_skills):
        """Test listing available integrations."""
        mock_list_skills.return_value = ["gmail", "slack", "supabase"]

        from seer.mcp.tools.guides import get_workflow_guide
        result = await get_workflow_guide.fn(integration="list")

        assert "Available Integration Guides" in result
        assert "gmail" in result
        assert "slack" in result

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.list_available_skills")
    async def test_integration_list_empty(self, mock_list_skills):
        """Test listing when no integrations available."""
        mock_list_skills.return_value = []

        from seer.mcp.tools.guides import get_workflow_guide
        result = await get_workflow_guide.fn(integration="list")

        assert "No integration guides available" in result

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.get_skill_guide")
    async def test_gmail_integration(self, mock_skill_guide):
        """Test gmail integration guide is returned."""
        mock_skill_guide.return_value = "# Gmail Integration\nSend emails using..."

        from seer.mcp.tools.guides import get_workflow_guide
        result = await get_workflow_guide.fn(integration="gmail")

        assert "Gmail Integration" in result
        mock_skill_guide.assert_called_once_with("gmail")

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.guides.list_available_skills")
    @patch("seer.mcp.tools.guides.get_skill_guide")
    async def test_unknown_integration(self, mock_skill_guide, mock_list_skills):
        """Test graceful handling of unknown integration."""
        mock_skill_guide.return_value = None
        mock_list_skills.return_value = ["gmail", "slack"]

        from seer.mcp.tools.guides import get_workflow_guide
        result = await get_workflow_guide.fn(integration="nonexistent_xyz")

        assert "not found" in result.lower()
        assert "gmail" in result
        assert "slack" in result
