"""
Integration tests for tool registry.

Tests:
- Tool registration and discovery
- Filtering tools by integration type
- Tool metadata retrieval
"""
from unittest.mock import MagicMock, patch

import pytest

from seer.tools.base import BaseTool
from seer.tools.registry import get_tools_by_integration


# =============================================================================
# Mock Tools for Testing
# =============================================================================


class MockGmailTool(BaseTool):
    name = "gmail_send_email"
    description = "Send an email via Gmail"
    integration_type = "gmail"
    required_scopes = ["gmail.send"]

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
        }

    async def execute(self, access_token: str, arguments: dict, credentials=None):
        return {"success": True}


class MockGithubTool(BaseTool):
    name = "github_create_issue"
    description = "Create a GitHub issue"
    integration_type = "github"
    required_scopes = ["repo"]

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "title": {"type": "string"},
            },
        }

    async def execute(self, access_token: str, arguments: dict, credentials=None):
        return {"success": True}


class MockSlackTool(BaseTool):
    name = "slack_post_message"
    description = "Post a message to Slack"
    integration_type = "slack"
    required_scopes = ["chat:write"]

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "text": {"type": "string"},
            },
        }

    async def execute(self, access_token: str, arguments: dict, credentials=None):
        return {"success": True}


class MockGenericTool(BaseTool):
    name = "http_request"
    description = "Make an HTTP request"
    required_scopes = []

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
        }

    async def execute(self, access_token: str, arguments: dict, credentials=None):
        return {"success": True}


# =============================================================================
# Tool Registry Tests
# =============================================================================


@pytest.mark.integration
def test_get_all_tools():
    """Test getting all tools without filtering."""
    mock_tools = [
        MockGmailTool(),
        MockGithubTool(),
        MockSlackTool(),
        MockGenericTool(),
    ]

    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = mock_tools

        tools = get_tools_by_integration()

        assert len(tools) == 4
        tool_names = {t["name"] for t in tools}
        assert tool_names == {
            "gmail_send_email",
            "github_create_issue",
            "slack_post_message",
            "http_request",
        }


@pytest.mark.integration
def test_get_tools_by_integration_type():
    """Test filtering tools by integration type."""
    mock_tools = [
        MockGmailTool(),
        MockGithubTool(),
        MockSlackTool(),
        MockGenericTool(),
    ]

    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = mock_tools

        # Test Gmail tools
        gmail_tools = get_tools_by_integration("gmail")
        assert len(gmail_tools) == 1
        assert gmail_tools[0]["name"] == "gmail_send_email"

        # Test GitHub tools
        github_tools = get_tools_by_integration("github")
        assert len(github_tools) == 1
        assert github_tools[0]["name"] == "github_create_issue"

        # Test Slack tools
        slack_tools = get_tools_by_integration("slack")
        assert len(slack_tools) == 1
        assert slack_tools[0]["name"] == "slack_post_message"


@pytest.mark.integration
def test_get_tools_by_integration_case_insensitive():
    """Test that integration type filtering is case-insensitive."""
    mock_tools = [MockGmailTool(), MockGithubTool()]

    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = mock_tools

        # Test different cases
        assert len(get_tools_by_integration("gmail")) == 1
        assert len(get_tools_by_integration("GMAIL")) == 1
        assert len(get_tools_by_integration("Gmail")) == 1


@pytest.mark.integration
def test_get_tools_by_name_fallback():
    """Test filtering by integration type in tool name (fallback)."""

    class ToolWithoutIntegrationType(BaseTool):
        name = "github_list_repos"
        description = "List GitHub repos"
        # No integration_type property
        required_scopes = []

        def get_parameters_schema(self):
            return {"type": "object"}

        async def execute(self, access_token: str, arguments: dict, credentials=None):
            return {}

    mock_tools = [ToolWithoutIntegrationType()]

    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = mock_tools

        # Should match because "github" is in tool name
        github_tools = get_tools_by_integration("github")
        assert len(github_tools) == 1
        assert github_tools[0]["name"] == "github_list_repos"


@pytest.mark.integration
def test_get_tools_by_scope_fallback():
    """Test filtering by integration type in scopes (fallback)."""

    class ToolWithScopeOnly(BaseTool):
        name = "custom_tool"
        description = "Custom tool"
        # No integration_type, no "gmail" in name
        required_scopes = ["gmail.readonly"]

        def get_parameters_schema(self):
            return {"type": "object"}

        async def execute(self, access_token: str, arguments: dict, credentials=None):
            return {}

    mock_tools = [ToolWithScopeOnly()]

    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = mock_tools

        # Should match because "gmail" is in required scopes
        gmail_tools = get_tools_by_integration("gmail")
        assert len(gmail_tools) == 1
        assert gmail_tools[0]["name"] == "custom_tool"


@pytest.mark.integration
def test_get_tools_no_matches():
    """Test filtering returns empty list when no tools match."""
    mock_tools = [MockGmailTool(), MockGithubTool()]

    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = mock_tools

        # No tools match "nonexistent"
        tools = get_tools_by_integration("nonexistent")
        assert len(tools) == 0


@pytest.mark.integration
def test_tool_metadata_format():
    """Test that tool metadata is returned in correct format."""
    mock_tools = [MockGmailTool()]

    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = mock_tools

        tools = get_tools_by_integration()

        assert len(tools) == 1
        tool_meta = tools[0]

        # Verify metadata structure
        assert "name" in tool_meta
        assert "description" in tool_meta
        assert tool_meta["name"] == "gmail_send_email"
        assert tool_meta["description"] == "Send an email via Gmail"


@pytest.mark.integration
def test_multiple_tools_same_integration():
    """Test retrieving multiple tools from same integration."""

    class GmailReadTool(BaseTool):
        name = "gmail_read_email"
        description = "Read Gmail emails"
        integration_type = "gmail"
        required_scopes = ["gmail.readonly"]

        def get_parameters_schema(self):
            return {"type": "object"}

        async def execute(self, access_token: str, arguments: dict, credentials=None):
            return {}

    mock_tools = [
        MockGmailTool(),  # gmail_send_email
        GmailReadTool(),  # gmail_read_email
        MockGithubTool(),  # github_create_issue
    ]

    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = mock_tools

        gmail_tools = get_tools_by_integration("gmail")
        assert len(gmail_tools) == 2

        tool_names = {t["name"] for t in gmail_tools}
        assert tool_names == {"gmail_send_email", "gmail_read_email"}


@pytest.mark.integration
def test_empty_tool_list():
    """Test behavior with empty tool list."""
    with patch("seer.tools.registry.list_tools") as mock_list_tools:
        mock_list_tools.return_value = []

        tools = get_tools_by_integration()
        assert len(tools) == 0

        filtered_tools = get_tools_by_integration("gmail")
        assert len(filtered_tools) == 0
