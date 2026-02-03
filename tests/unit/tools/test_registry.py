"""
Unit tests for tools.registry module.

Tests tool filtering and metadata retrieval.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# get_tools_by_integration Tests
# =============================================================================


@pytest.mark.unit
class TestGetToolsByIntegration:
    """Tests for get_tools_by_integration function."""

    @pytest.fixture
    def mock_tools(self):
        """Create a list of mock tools."""
        gmail_tool = MagicMock()
        gmail_tool.name = "gmail_send"
        gmail_tool.integration_type = "gmail"
        gmail_tool.required_scopes = ["gmail.send"]
        gmail_tool.get_metadata.return_value = {
            "name": "gmail_send",
            "integration_type": "gmail",
            "required_scopes": ["gmail.send"],
        }

        github_tool = MagicMock()
        github_tool.name = "github_list_repos"
        github_tool.integration_type = "github"
        github_tool.required_scopes = ["repo"]
        github_tool.get_metadata.return_value = {
            "name": "github_list_repos",
            "integration_type": "github",
            "required_scopes": ["repo"],
        }

        generic_tool = MagicMock()
        generic_tool.name = "http_request"
        generic_tool.integration_type = None
        generic_tool.required_scopes = []
        generic_tool.get_metadata.return_value = {
            "name": "http_request",
            "integration_type": None,
            "required_scopes": [],
        }

        return [gmail_tool, github_tool, generic_tool]

    def test_get_tools_by_integration_no_filter(self, mock_tools):
        """Test get_tools_by_integration returns all tools when no filter."""
        from seer.tools.registry import get_tools_by_integration

        with patch("seer.tools.registry.list_tools", return_value=mock_tools):
            result = get_tools_by_integration(None)

        assert len(result) == 3
        names = [t["name"] for t in result]
        assert "gmail_send" in names
        assert "github_list_repos" in names
        assert "http_request" in names

    def test_get_tools_by_integration_filter_by_type(self, mock_tools):
        """Test get_tools_by_integration filters by integration_type."""
        from seer.tools.registry import get_tools_by_integration

        with patch("seer.tools.registry.list_tools", return_value=mock_tools):
            result = get_tools_by_integration("gmail")

        assert len(result) == 1
        assert result[0]["name"] == "gmail_send"

    def test_get_tools_by_integration_case_insensitive(self, mock_tools):
        """Test get_tools_by_integration is case-insensitive."""
        from seer.tools.registry import get_tools_by_integration

        with patch("seer.tools.registry.list_tools", return_value=mock_tools):
            result = get_tools_by_integration("GMAIL")

        assert len(result) == 1
        assert result[0]["name"] == "gmail_send"

    def test_get_tools_by_integration_fallback_to_name(self):
        """Test get_tools_by_integration falls back to name matching."""
        from seer.tools.registry import get_tools_by_integration

        tool = MagicMock()
        tool.name = "slack_send_message"
        tool.integration_type = None  # No integration_type set
        tool.required_scopes = []
        tool.get_metadata.return_value = {
            "name": "slack_send_message",
            "integration_type": None,
        }

        with patch("seer.tools.registry.list_tools", return_value=[tool]):
            result = get_tools_by_integration("slack")

        assert len(result) == 1
        assert result[0]["name"] == "slack_send_message"

    def test_get_tools_by_integration_fallback_to_scopes(self):
        """Test get_tools_by_integration falls back to scope matching."""
        from seer.tools.registry import get_tools_by_integration

        tool = MagicMock()
        tool.name = "email_tool"  # Name doesn't match
        tool.integration_type = None  # Type doesn't match
        tool.required_scopes = ["gmail.readonly", "gmail.send"]
        tool.get_metadata.return_value = {
            "name": "email_tool",
            "integration_type": None,
            "required_scopes": ["gmail.readonly", "gmail.send"],
        }

        with patch("seer.tools.registry.list_tools", return_value=[tool]):
            result = get_tools_by_integration("gmail")

        assert len(result) == 1
        assert result[0]["name"] == "email_tool"

    def test_get_tools_by_integration_no_matches(self, mock_tools):
        """Test get_tools_by_integration returns empty list when no matches."""
        from seer.tools.registry import get_tools_by_integration

        with patch("seer.tools.registry.list_tools", return_value=mock_tools):
            result = get_tools_by_integration("nonexistent")

        assert result == []

    def test_get_tools_by_integration_empty_registry(self):
        """Test get_tools_by_integration handles empty registry."""
        from seer.tools.registry import get_tools_by_integration

        with patch("seer.tools.registry.list_tools", return_value=[]):
            result = get_tools_by_integration("any")

        assert result == []

    def test_get_tools_by_integration_multiple_matches(self):
        """Test get_tools_by_integration returns multiple matching tools."""
        from seer.tools.registry import get_tools_by_integration

        tool1 = MagicMock()
        tool1.name = "github_list_repos"
        tool1.integration_type = "github"
        tool1.required_scopes = []
        tool1.get_metadata.return_value = {"name": "github_list_repos", "integration_type": "github"}

        tool2 = MagicMock()
        tool2.name = "github_create_issue"
        tool2.integration_type = "github"
        tool2.required_scopes = []
        tool2.get_metadata.return_value = {"name": "github_create_issue", "integration_type": "github"}

        tool3 = MagicMock()
        tool3.name = "github_pr_review"
        tool3.integration_type = "github"
        tool3.required_scopes = []
        tool3.get_metadata.return_value = {"name": "github_pr_review", "integration_type": "github"}

        with patch("seer.tools.registry.list_tools", return_value=[tool1, tool2, tool3]):
            result = get_tools_by_integration("github")

        assert len(result) == 3
        names = [t["name"] for t in result]
        assert "github_list_repos" in names
        assert "github_create_issue" in names
        assert "github_pr_review" in names


# =============================================================================
# ToolEntry Tests
# =============================================================================


@pytest.mark.unit
class TestToolEntry:
    """Tests for ToolEntry dataclass."""

    def test_tool_entry_creation(self):
        """Test creating a ToolEntry."""
        from seer.tools.registry import ToolEntry

        entry = ToolEntry(
            name="my_tool",
            description="My tool description",
            service="my_service"
        )

        assert entry.name == "my_tool"
        assert entry.description == "My tool description"
        assert entry.service == "my_service"

    def test_tool_entry_equality(self):
        """Test ToolEntry equality."""
        from seer.tools.registry import ToolEntry

        entry1 = ToolEntry(name="tool", description="desc", service="svc")
        entry2 = ToolEntry(name="tool", description="desc", service="svc")

        assert entry1 == entry2

    def test_tool_entry_different(self):
        """Test ToolEntry inequality."""
        from seer.tools.registry import ToolEntry

        entry1 = ToolEntry(name="tool1", description="desc", service="svc")
        entry2 = ToolEntry(name="tool2", description="desc", service="svc")

        assert entry1 != entry2
