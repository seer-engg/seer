"""
Unit tests for MCP workflow CRUD tools.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestValidateWorkflow:
    """Tests for validate_workflow MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows.parse_workflow_spec")
    async def test_validate_catches_schema_errors(self, mock_parse):
        """Test that validation catches schema errors."""
        from seer.core.errors import ValidationPhaseError
        mock_parse.side_effect = ValidationPhaseError("Invalid node type")

        from seer.mcp.tools.workflows import validate_workflow
        result = await validate_workflow.fn({"version": "2", "nodes": [], "edges": []})
        data = json.loads(result)

        assert data["ok"] is False
        assert "schema_validation" in data["error_type"]

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    @patch("seer.mcp.tools.workflows.parse_workflow_spec")
    @patch("seer.tools.base.get_tool")
    async def test_validate_checks_tool_references(self, mock_get_tool, mock_parse, mock_get_user):
        """Test that validation checks tool references exist."""
        # Mock a valid spec with a nonexistent tool
        mock_spec = MagicMock()
        mock_node = MagicMock()
        mock_node.type = "tool"
        mock_node.tool = "nonexistent_tool"
        mock_spec.nodes = [mock_node]
        mock_spec.triggers = []
        mock_parse.return_value = mock_spec

        mock_user = MagicMock()
        mock_get_user.return_value = mock_user
        mock_get_tool.return_value = None  # Tool not found

        from seer.mcp.tools.workflows import validate_workflow
        result = await validate_workflow.fn({
            "version": "2",
            "nodes": [{"id": "n1", "type": "tool", "tool": "nonexistent_tool", "inputs": {}}],
            "edges": []
        })
        data = json.loads(result)

        assert data["ok"] is False
        assert "reference_validation" in data.get("error_type", "")


class TestCreateWorkflow:
    """Tests for create_workflow MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    @patch("seer.mcp.tools.workflows.parse_workflow_spec")
    async def test_create_validates_spec_first(self, mock_parse, mock_get_user):
        """Test that create_workflow validates spec before creating."""
        from seer.core.errors import ValidationPhaseError
        mock_parse.side_effect = ValidationPhaseError("Invalid spec")

        from seer.mcp.tools.workflows import create_workflow
        result = await create_workflow.fn("Test Workflow", {"invalid": "spec"})
        data = json.loads(result)

        assert "error" in data
        assert data["error"] == "validation_failed"


class TestListWorkflows:
    """Tests for list_workflows MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    async def test_list_workflows_returns_empty_on_error(self, mock_get_user):
        """Test that list_workflows returns empty list on error."""
        mock_get_user.side_effect = Exception("Database error")

        from seer.mcp.tools.workflows import list_workflows
        result = await list_workflows.fn()
        data = json.loads(result)

        assert "workflows" in data
        assert data["workflows"] == []
        assert "error" in data


class TestDeleteWorkflow:
    """Tests for delete_workflow MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    async def test_delete_returns_error_on_failure(self, mock_get_user):
        """Test that delete_workflow returns error on failure."""
        mock_get_user.side_effect = Exception("Not found")

        from seer.mcp.tools.workflows import delete_workflow
        result = await delete_workflow.fn("wf_nonexistent")
        data = json.loads(result)

        assert data["deleted"] is False
        assert "error" in data
