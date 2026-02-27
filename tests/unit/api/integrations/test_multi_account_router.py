"""
Unit tests for multi-account OAuth API endpoints.

Tests the new API endpoints that support multi-account OAuth scenarios.
"""
from unittest.mock import MagicMock

import pytest

from seer.api.integrations.models import (
    AccountInfo,
    ConnectionsByProvider,
    ToolAccountInfo,
    ToolAccountsResponse,
)
from seer.services.integrations.auth.helpers import get_connection_display_name

pytestmark = pytest.mark.unit


# =============================================================================
# Helper Function Tests: get_connection_display_name
# =============================================================================


def _make_mock_connection(provider: str, provider_account_id: str, metadata: dict) -> MagicMock:
    """Factory for creating mock OAuthConnection objects."""
    conn = MagicMock()
    conn.id = 123
    conn.provider = provider
    conn.provider_account_id = provider_account_id
    conn.provider_metadata = metadata
    return conn


def test_get_display_name_google_with_email():
    """Test display name extraction for Google with email in metadata."""
    conn = _make_mock_connection(
        provider="google",
        provider_account_id="113421198217490684849",
        metadata={"email": "alice@gmail.com", "sub": "113421198217490684849"},
    )
    assert get_connection_display_name(conn) == "alice@gmail.com"


def test_get_display_name_google_fallback_to_account_id():
    """Test Google fallback when email is missing from metadata."""
    conn = _make_mock_connection(
        provider="google",
        provider_account_id="113421198217490684849",
        metadata={"sub": "113421198217490684849"},  # No email
    )
    assert get_connection_display_name(conn) == "113421198217490684849"


def test_get_display_name_github_with_login():
    """Test display name extraction for GitHub with login in metadata."""
    conn = _make_mock_connection(
        provider="github",
        provider_account_id="12345678",
        metadata={"login": "alice", "name": "Alice Smith", "id": 12345678},
    )
    assert get_connection_display_name(conn) == "alice"


def test_get_display_name_github_fallback_to_name():
    """Test GitHub fallback to name when login is missing."""
    conn = _make_mock_connection(
        provider="github",
        provider_account_id="12345678",
        metadata={"name": "Alice Smith", "id": 12345678},  # No login
    )
    assert get_connection_display_name(conn) == "Alice Smith"


def test_get_display_name_slack_with_team():
    """Test display name extraction for Slack with team info."""
    conn = _make_mock_connection(
        provider="slack",
        provider_account_id="T12345",
        metadata={"team": {"id": "T12345", "name": "My Workspace"}},
    )
    assert get_connection_display_name(conn) == "My Workspace"


def test_get_display_name_discord_with_username():
    """Test display name extraction for Discord with username."""
    conn = _make_mock_connection(
        provider="discord",
        provider_account_id="123456789012345678",
        metadata={"username": "alice#1234", "id": "123456789012345678"},
    )
    assert get_connection_display_name(conn) == "alice#1234"


def test_get_display_name_linkedin_with_email():
    """Test display name extraction for LinkedIn with email."""
    conn = _make_mock_connection(
        provider="linkedin",
        provider_account_id="abc123xyz",
        metadata={"email": "alice@company.com", "name": "Alice Smith"},
    )
    assert get_connection_display_name(conn) == "alice@company.com"


def test_get_display_name_unknown_provider_generic_fallback():
    """Test generic fallback for unknown providers."""
    conn = _make_mock_connection(
        provider="unknown_provider",
        provider_account_id="some_id",
        metadata={"email": "test@example.com"},
    )
    assert get_connection_display_name(conn) == "test@example.com"


def test_get_display_name_empty_metadata():
    """Test fallback when metadata is empty."""
    conn = _make_mock_connection(
        provider="google",
        provider_account_id="113421198217490684849",
        metadata={},
    )
    assert get_connection_display_name(conn) == "113421198217490684849"


def test_get_display_name_none_metadata():
    """Test fallback when metadata is None."""
    conn = MagicMock()
    conn.id = 123
    conn.provider = "google"
    conn.provider_account_id = "113421198217490684849"
    conn.provider_metadata = None
    assert get_connection_display_name(conn) == "113421198217490684849"


def test_get_display_name_all_none():
    """Test fallback when all fields are None/empty."""
    conn = MagicMock()
    conn.id = 456
    conn.provider = "google"
    conn.provider_account_id = None
    conn.provider_metadata = None
    assert get_connection_display_name(conn) == "ID:456"


# =============================================================================
# Model Tests
# =============================================================================


def test_account_info_model():
    """Test AccountInfo model serialization."""
    account = AccountInfo(
        id=123,
        provider="google",
        provider_account_id="113421198217490684849",
        display_name="alice@gmail.com",
        status="active",
        scopes="https://www.googleapis.com/auth/gmail.send",
    )

    assert account.id == 123
    assert account.provider == "google"
    assert account.provider_account_id == "113421198217490684849"
    assert account.display_name == "alice@gmail.com"


def test_connections_by_provider_model():
    """Test ConnectionsByProvider model with grouped connections."""
    connections = ConnectionsByProvider(
        connections={
            "google": [
                AccountInfo(
                    id=123,
                    provider="google",
                    provider_account_id="113421198217490684849",
                    display_name="alice@gmail.com",
                    status="active",
                    scopes=None,
                )
            ],
            "github": [
                AccountInfo(
                    id=124,
                    provider="github",
                    provider_account_id="12345678",
                    display_name="alice",
                    status="active",
                    scopes=None,
                )
            ],
        }
    )

    assert "google" in connections.connections
    assert "github" in connections.connections
    assert len(connections.connections["google"]) == 1
    assert connections.connections["google"][0].display_name == "alice@gmail.com"
    assert connections.connections["github"][0].display_name == "alice"


def test_tool_account_info_model():
    """Test ToolAccountInfo model for tool-specific account info."""
    account = ToolAccountInfo(
        id=123,
        provider_account_id="113421198217490684849",
        display_name="alice@gmail.com",
        has_required_scopes=True,
        missing_scopes=[],
    )

    assert account.id == 123
    assert account.provider_account_id == "113421198217490684849"
    assert account.display_name == "alice@gmail.com"
    assert account.has_required_scopes is True


def test_tool_account_info_with_missing_scopes():
    """Test ToolAccountInfo with missing scopes."""
    account = ToolAccountInfo(
        id=123,
        provider_account_id="113421198217490684849",
        display_name="alice@gmail.com",
        has_required_scopes=False,
        missing_scopes=["https://www.googleapis.com/auth/gmail.send"],
    )

    assert account.has_required_scopes is False
    assert len(account.missing_scopes) == 1
    assert account.display_name == "alice@gmail.com"


def test_tool_accounts_response_model():
    """Test ToolAccountsResponse model."""
    response = ToolAccountsResponse(
        tool_name="google.gmail.send_email",
        provider="google",
        accounts=[
            ToolAccountInfo(
                id=123,
                provider_account_id="113421198217490684849",
                display_name="alice@gmail.com",
                has_required_scopes=True,
                missing_scopes=[],
            ),
            ToolAccountInfo(
                id=124,
                provider_account_id="116269834399389497153",
                display_name="bob@gmail.com",
                has_required_scopes=True,
                missing_scopes=[],
            ),
        ],
        requires_selection=True,
    )

    assert response.tool_name == "google.gmail.send_email"
    assert response.requires_selection is True
    assert len(response.accounts) == 2
    assert response.accounts[0].display_name == "alice@gmail.com"
    assert response.accounts[1].display_name == "bob@gmail.com"


def test_tool_accounts_response_no_selection_needed():
    """Test ToolAccountsResponse when only one account exists."""
    response = ToolAccountsResponse(
        tool_name="google.gmail.send_email",
        provider="google",
        accounts=[
            ToolAccountInfo(
                id=123,
                provider_account_id="113421198217490684849",
                display_name="alice@gmail.com",
                has_required_scopes=True,
                missing_scopes=[],
            ),
        ],
        requires_selection=False,
    )

    assert response.requires_selection is False
    assert len(response.accounts) == 1
    assert response.accounts[0].display_name == "alice@gmail.com"
