"""
Unit tests for template tools (unified implementations).

Tests the canonical tool implementations from seer.tools.unified_tools,
using shared template search logic from seer.tools.template_shared.
"""

import json
import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestGetWorkflowTemplate:
    """Tests for get_workflow_template_impl — unified canonical implementation."""

    @pytest.mark.asyncio
    @patch("seer.tools.template_shared.search_templates")
    async def test_get_template_finds_matching(self, mock_search):
        """Test that get_workflow_template_impl finds matching templates."""
        # search_templates returns a dict with matches, count, message
        mock_search.return_value = {
            "query": "gmail",
            "matches": [
                {
                    "name": "Supabase to Gmail Welcome",
                    "description": "Send welcome email on Supabase signup",
                    "tags": ["supabase", "gmail", "welcome"],
                    "customization_guide": "Update email content",
                    "spec": {"version": "2", "nodes": [], "edges": []}
                }
            ],
            "count": 1,
            "message": "Found 1 template(s) matching 'gmail'"
        }

        from seer.tools.unified_tools import get_workflow_template_impl
        result = await get_workflow_template_impl("gmail")
        data = json.loads(result)

        assert data["count"] == 1
        assert len(data["matches"]) == 1
        assert "gmail" in data["matches"][0]["name"].lower()

    @pytest.mark.asyncio
    @patch("seer.tools.template_shared.search_templates")
    async def test_get_template_no_matches(self, mock_search):
        """Test that get_workflow_template_impl handles no matches."""
        # search_templates returns a dict with empty matches when no results
        mock_search.return_value = {
            "query": "nonexistent",
            "matches": [],
            "message": "No templates found matching 'nonexistent'",
            "available_templates": [{"name": "Test Template", "tags": ["test"]}],
            "suggestion": "Try searching with integration names"
        }

        from seer.tools.unified_tools import get_workflow_template_impl
        result = await get_workflow_template_impl("nonexistent")
        data = json.loads(result)

        assert data["matches"] == []
        assert "message" in data
        assert "available_templates" in data


@pytest.mark.unit
class TestListWorkflowTemplates:
    """Tests for list_workflow_templates_impl — unified canonical implementation."""

    @pytest.mark.asyncio
    @patch("seer.tools.template_shared.list_all_templates")
    async def test_list_templates_returns_all(self, mock_list):
        """Test that list_workflow_templates_impl returns all templates."""
        # list_all_templates returns a dict with templates list and total
        mock_list.return_value = {
            "templates": [
                {"name": "Template 1", "description": "Desc 1", "tags": ["t1"]},
                {"name": "Template 2", "description": "Desc 2", "tags": ["t2"]},
            ],
            "total": 2
        }

        from seer.tools.unified_tools import list_workflow_templates_impl
        result = await list_workflow_templates_impl()
        data = json.loads(result)

        assert data["total"] == 2
        assert len(data["templates"]) == 2

    @pytest.mark.asyncio
    @patch("seer.tools.template_shared.list_all_templates")
    async def test_list_templates_empty(self, mock_list):
        """Test that list_workflow_templates_impl handles empty list."""
        # list_all_templates returns a dict with empty templates list
        mock_list.return_value = {
            "templates": [],
            "total": 0
        }

        from seer.tools.unified_tools import list_workflow_templates_impl
        result = await list_workflow_templates_impl()
        data = json.loads(result)

        assert data["total"] == 0
        assert data["templates"] == []
