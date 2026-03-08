"""
Unit tests for NotionProvider.

Tests cover the OAuth lifecycle methods specific to Notion's unusual
OAuth flow (no URL scopes, owner=user param, token-embedded profile).
"""
from unittest.mock import MagicMock

import pytest

from seer.services.integrations.providers.notion import NotionProvider


def make_context(requested_scopes=None):
    ctx = MagicMock()
    ctx.requested_scopes = requested_scopes or []
    ctx.existing_connection = None
    ctx.integration_type = "notion"
    return ctx


@pytest.mark.unit
class TestNotionProvider:
    """Test NotionProvider OAuth lifecycle methods."""

    def test_provider_name(self):
        provider = NotionProvider()
        assert provider.provider == "notion"

    def test_get_oauth_scope_returns_empty_string(self):
        """Notion doesn't use URL scopes - capabilities set in dashboard."""
        provider = NotionProvider()
        ctx = make_context(["some_scope"])
        scope = provider.get_oauth_scope(ctx)
        assert scope == ""

    def test_build_authorize_kwargs_includes_owner_user(self):
        """Notion requires owner=user in the authorize URL."""
        provider = NotionProvider()
        ctx = make_context()
        kwargs = provider.build_authorize_kwargs(ctx, state="enc-state", scope="")
        assert kwargs.get("owner") == "user"
        assert kwargs.get("state") == "enc-state"

    @pytest.mark.asyncio
    async def test_resolve_granted_scopes_returns_empty_string(self):
        """Notion capabilities are dashboard-configured, not URL scopes."""
        provider = NotionProvider()
        result = await provider.resolve_granted_scopes(
            token={"access_token": "tok", "scope": "should-be-ignored"},
            state_data={"requested_scope": "also-ignored"},
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_user_profile_extracts_from_token(self):
        """User profile is embedded in token['owner']['user']."""
        provider = NotionProvider()
        token = {
            "access_token": "ntn_token",
            "bot_id": "bot-123",
            "workspace_id": "ws-456",
            "workspace_name": "Acme Corp",
            "owner": {
                "user": {
                    "id": "user-789",
                    "name": "Alice Smith",
                    "avatar_url": "https://notion.so/avatar.png",
                    "person": {"email": "alice@acme.com"},
                }
            },
        }
        profile = await provider.fetch_user_profile(client=None, token=token, state_data={})

        assert profile["id"] == "user-789"
        assert profile["name"] == "Alice Smith"
        assert profile["email"] == "alice@acme.com"
        assert profile["avatar_url"] == "https://notion.so/avatar.png"
        assert profile["workspace_id"] == "ws-456"
        assert profile["workspace_name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_fetch_user_profile_falls_back_to_bot_id(self):
        """Falls back to bot_id when owner.user.id is missing."""
        provider = NotionProvider()
        token = {
            "access_token": "ntn_token",
            "bot_id": "bot-fallback",
            "workspace_id": "ws-1",
            "workspace_name": "My Workspace",
            "owner": {"user": {}},
        }
        profile = await provider.fetch_user_profile(client=None, token=token, state_data={})
        assert profile["id"] == "bot-fallback"

    @pytest.mark.asyncio
    async def test_fetch_user_profile_handles_missing_owner(self):
        """Handles token response with missing owner field gracefully."""
        provider = NotionProvider()
        token = {
            "access_token": "ntn_token",
            "bot_id": "bot-123",
            "workspace_id": "ws-1",
            "workspace_name": "Workspace",
        }
        profile = await provider.fetch_user_profile(client=None, token=token, state_data={})
        assert profile["id"] == "bot-123"
        assert profile["name"] == ""
        assert profile["email"] == ""
