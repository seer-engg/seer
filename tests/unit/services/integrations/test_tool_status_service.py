"""
Tests for tool status service with provider-specific refresh token handling.
"""
import pytest

from seer.services.integrations.tool_status_service import (
    PROVIDERS_WITHOUT_REFRESH_TOKENS,
    build_tool_status,
    determine_tool_auth_requirements,
)


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name, required_scopes=None, required_secrets=None, provider=None, integration_type=None):
        self.name = name
        self.required_scopes = required_scopes or []
        self.required_secrets = required_secrets or []
        self.provider = provider
        self.integration_type = integration_type


@pytest.mark.unit
class TestProviderWithoutRefreshTokens:
    """Test LinkedIn, Slack, and other providers without refresh tokens."""

    def test_linkedin_in_providers_without_refresh_tokens(self):
        """Verify LinkedIn is in the PROVIDERS_WITHOUT_REFRESH_TOKENS set."""
        assert "linkedin" in PROVIDERS_WITHOUT_REFRESH_TOKENS

    def test_slack_in_providers_without_refresh_tokens(self):
        """Verify Slack is in the PROVIDERS_WITHOUT_REFRESH_TOKENS set.

        Slack bot tokens (xoxb-*) are permanent and don't use refresh tokens.
        """
        assert "slack" in PROVIDERS_WITHOUT_REFRESH_TOKENS

    def test_slack_connected_without_refresh_token(self):
        """Slack tools should show connected with scopes but no refresh token.

        Slack's OAuth returns permanent bot tokens without refresh_token field.
        """
        tool = MockTool(
            name="slack_send_channel_message",
            required_scopes=["chat:write", "channels:read"],
            provider="slack",
            integration_type="slack"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        # Connection with scopes but NO refresh token (Slack's actual behavior)
        conn_info = {
            "scopes": "chat:write,channels:read",  # Slack uses comma-separated scopes
            "has_refresh_token": False,  # Slack doesn't provide refresh tokens
            "connection_id": "slack:123",
            "provider_account_id": "T08ABKNJPGT"
        }

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="slack",
            provider_aliases=[],
            conn_info=conn_info,
            provider_secrets={}
        )

        # Should be connected despite no refresh token
        assert status["connected"] is True
        assert status["missing_scopes"] == []
        assert status["connection_id"] == "slack:123"

    def test_slack_not_connected_missing_scopes(self):
        """Slack tools should show not connected when scopes are missing."""
        tool = MockTool(
            name="slack_list_channels",
            required_scopes=["channels:read", "groups:read"],
            provider="slack",
            integration_type="slack"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        # Connection without groups:read scope
        conn_info = {
            "scopes": "chat:write,channels:read",  # Missing groups:read
            "has_refresh_token": False,
            "connection_id": "slack:123",
            "provider_account_id": "T08ABKNJPGT"
        }

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="slack",
            provider_aliases=[],
            conn_info=conn_info,
            provider_secrets={}
        )

        # Should NOT be connected due to missing scope
        assert status["connected"] is False
        assert "groups:read" in status["missing_scopes"]

    def test_linkedin_connected_without_refresh_token(self):
        """LinkedIn tools should show connected with scopes but no refresh token."""
        tool = MockTool(
            name="linkedin_get_profile",
            required_scopes=["openid", "profile", "email"],
            provider="linkedin",
            integration_type="linkedin"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        # Connection with scopes but NO refresh token (LinkedIn's actual behavior)
        conn_info = {
            "scopes": "openid profile email",
            "has_refresh_token": False,  # LinkedIn doesn't provide refresh tokens
            "connection_id": "linkedin:123",
            "provider_account_id": "abc123"
        }

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="linkedin",
            provider_aliases=[],
            conn_info=conn_info,
            provider_secrets={}
        )

        # Should be connected despite no refresh token
        assert status["connected"] is True
        assert status["missing_scopes"] == []
        assert status["connection_id"] == "linkedin:123"

    def test_linkedin_not_connected_missing_scopes(self):
        """LinkedIn tools should show not connected when scopes are missing."""
        tool = MockTool(
            name="linkedin_create_post",
            required_scopes=["openid", "profile", "email", "w_member_social"],
            provider="linkedin",
            integration_type="linkedin"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        # Connection without posting scope
        conn_info = {
            "scopes": "openid profile email",  # Missing w_member_social
            "has_refresh_token": False,
            "connection_id": "linkedin:123",
            "provider_account_id": "abc123"
        }

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="linkedin",
            provider_aliases=[],
            conn_info=conn_info,
            provider_secrets={}
        )

        # Should NOT be connected due to missing scope
        assert status["connected"] is False
        assert "w_member_social" in status["missing_scopes"]

    def test_linkedin_not_connected_no_connection(self):
        """LinkedIn tools should show not connected when no connection exists."""
        tool = MockTool(
            name="linkedin_get_profile",
            required_scopes=["openid", "profile", "email"],
            provider="linkedin",
            integration_type="linkedin"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="linkedin",
            provider_aliases=[],
            conn_info=None,  # No connection
            provider_secrets={}
        )

        # Should NOT be connected
        assert status["connected"] is False
        assert status["missing_scopes"] == ["openid", "profile", "email"]


@pytest.mark.unit
class TestStandardProvidersRequireRefreshTokens:
    """Test that standard providers (Google, GitHub) still require refresh tokens."""

    def test_google_requires_refresh_token(self):
        """Google tools should require both scopes AND refresh token."""
        tool = MockTool(
            name="gmail_read_messages",
            required_scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            provider="google",
            integration_type="gmail"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        # Connection with scopes but NO refresh token
        conn_info = {
            "scopes": "https://www.googleapis.com/auth/gmail.readonly",
            "has_refresh_token": False,  # Missing refresh token
            "connection_id": "google:123",
            "provider_account_id": "user@gmail.com"
        }

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="google",
            provider_aliases=["gmail"],
            conn_info=conn_info,
            provider_secrets={}
        )

        # Should NOT be connected without refresh token
        assert status["connected"] is False

    def test_google_connected_with_refresh_token(self):
        """Google tools should be connected with both scopes and refresh token."""
        tool = MockTool(
            name="gmail_read_messages",
            required_scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            provider="google",
            integration_type="gmail"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        # Connection with scopes AND refresh token
        conn_info = {
            "scopes": "https://www.googleapis.com/auth/gmail.readonly",
            "has_refresh_token": True,  # Has refresh token
            "connection_id": "google:123",
            "provider_account_id": "user@gmail.com"
        }

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="google",
            provider_aliases=["gmail"],
            conn_info=conn_info,
            provider_secrets={}
        )

        # Should be connected with both scopes and refresh token
        assert status["connected"] is True
        assert status["missing_scopes"] == []

    def test_github_requires_refresh_token(self):
        """GitHub tools should require both scopes AND refresh token."""
        tool = MockTool(
            name="github_create_issue",
            required_scopes=["repo"],
            provider="github",
            integration_type="github"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        # Connection with scopes but NO refresh token
        conn_info = {
            "scopes": "repo",
            "has_refresh_token": False,  # Missing refresh token
            "connection_id": "github:123",
            "provider_account_id": "octocat"
        }

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="github",
            provider_aliases=[],
            conn_info=conn_info,
            provider_secrets={}
        )

        # Should NOT be connected without refresh token
        assert status["connected"] is False


@pytest.mark.unit
class TestMixedAuthModes:
    """Test tools that support both OAuth and manual secrets."""

    def test_linkedin_with_manual_secrets_fallback(self):
        """LinkedIn tools should connect via secrets even without OAuth."""
        tool = MockTool(
            name="linkedin_api_call",
            required_scopes=["openid", "profile"],
            required_secrets=["api_key"],
            provider="linkedin",
            integration_type="linkedin"
        )

        auth_requirements = determine_tool_auth_requirements(tool)

        # No OAuth connection but has manual secret
        provider_secrets = {
            "linkedin": {"api_key"}
        }

        status = build_tool_status(
            tool=tool,
            auth_requirements=auth_requirements,
            provider="linkedin",
            provider_aliases=[],
            conn_info=None,  # No OAuth connection
            provider_secrets=provider_secrets
        )

        # Should be connected via manual secrets
        assert status["connected"] is True
