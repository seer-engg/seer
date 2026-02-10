"""
Unit tests for MCP template tools.

Tests the MCP tool wrappers that use shared template search logic from
seer.tools.template_shared.
"""

import json
import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestGetWorkflowTemplate:
    """Tests for get_workflow_template MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.tools.template_shared.get_workflow_templates")
    async def test_get_template_finds_matching(self, mock_get_templates):
        """Test that get_workflow_template finds matching templates."""
        mock_get_templates.return_value = [
            {
                "name": "Supabase to Gmail Welcome",
                "description": "Send welcome email on Supabase signup",
                "tags": ["supabase", "gmail", "welcome"],
                "customization_guide": "Update email content",
                "spec": {"version": "2", "nodes": [], "edges": []}
            },
            {
                "name": "Slack Notification",
                "description": "Send Slack notifications",
                "tags": ["slack", "notification"],
                "customization_guide": "Update channel",
                "spec": {"version": "2", "nodes": [], "edges": []}
            },
        ]

        from seer.mcp.tools.templates import get_workflow_template
        result = await get_workflow_template.fn("gmail")
        data = json.loads(result)

        assert data["count"] == 1
        assert len(data["matches"]) == 1
        assert "gmail" in data["matches"][0]["name"].lower()

    @pytest.mark.asyncio
    @patch("seer.tools.template_shared.get_workflow_templates")
    async def test_get_template_no_matches(self, mock_get_templates):
        """Test that get_workflow_template handles no matches."""
        mock_get_templates.return_value = [
            {
                "name": "Test Template",
                "description": "A test template",
                "tags": ["test"],
                "spec": {}
            }
        ]

        from seer.mcp.tools.templates import get_workflow_template
        result = await get_workflow_template.fn("nonexistent")
        data = json.loads(result)

        assert data["matches"] == []
        assert "message" in data
        assert "available_templates" in data


@pytest.mark.unit
class TestListWorkflowTemplates:
    """Tests for list_workflow_templates MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.tools.template_shared.get_workflow_templates")
    async def test_list_templates_returns_all(self, mock_get_templates):
        """Test that list_workflow_templates returns all templates."""
        mock_get_templates.return_value = [
            {"name": "Template 1", "description": "Desc 1", "tags": ["t1"]},
            {"name": "Template 2", "description": "Desc 2", "tags": ["t2"]},
        ]

        from seer.mcp.tools.templates import list_workflow_templates
        result = await list_workflow_templates.fn()
        data = json.loads(result)

        assert data["total"] == 2
        assert len(data["templates"]) == 2

    @pytest.mark.asyncio
    @patch("seer.tools.template_shared.get_workflow_templates")
    async def test_list_templates_empty(self, mock_get_templates):
        """Test that list_workflow_templates handles empty list."""
        mock_get_templates.return_value = []

        from seer.mcp.tools.templates import list_workflow_templates
        result = await list_workflow_templates.fn()
        data = json.loads(result)

        assert data["total"] == 0
        assert data["templates"] == []
