"""Tests for account_helpers module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCheckScopeCoverage:
    """Tests for check_scope_coverage function."""

    def test_no_required_scopes_returns_true(self):
        """When no scopes are required, always returns True."""
        from seer.tools.account_helpers import check_scope_coverage

        has_scopes, missing = check_scope_coverage("some_scope", [])
        assert has_scopes is True
        assert missing == []

    def test_no_connection_scopes_returns_all_missing(self):
        """When connection has no scopes, all required scopes are missing."""
        from seer.tools.account_helpers import check_scope_coverage

        required = ["scope_a", "scope_b"]
        has_scopes, missing = check_scope_coverage(None, required)
        assert has_scopes is False
        assert missing == ["scope_a", "scope_b"]

    def test_empty_connection_scopes_returns_all_missing(self):
        """When connection has empty scopes string, all required scopes are missing."""
        from seer.tools.account_helpers import check_scope_coverage

        required = ["scope_a", "scope_b"]
        has_scopes, missing = check_scope_coverage("", required)
        assert has_scopes is False
        assert missing == ["scope_a", "scope_b"]

    def test_all_scopes_present(self):
        """When all required scopes are present, returns True with empty missing list."""
        from seer.tools.account_helpers import check_scope_coverage

        with patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has:
            mock_has.return_value = True

            has_scopes, missing = check_scope_coverage(
                "scope_a scope_b",
                ["scope_a", "scope_b"]
            )
            assert has_scopes is True
            assert missing == []

    def test_partial_scopes_returns_missing(self):
        """When some scopes are missing, returns False with list of missing scopes."""
        from seer.tools.account_helpers import check_scope_coverage

        with patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has, \
             patch("seer.services.integrations.auth.helpers.parse_scopes") as mock_parse:
            mock_has.return_value = False
            mock_parse.return_value = {"scope_a"}  # Only scope_a granted

            has_scopes, missing = check_scope_coverage(
                "scope_a",
                ["scope_a", "scope_b", "scope_c"]
            )
            assert has_scopes is False
            assert missing == ["scope_b", "scope_c"]


class TestBuildAccountEntry:
    """Tests for build_account_entry function."""

    def test_builds_complete_entry(self):
        """Test building a complete account entry with all fields."""
        from seer.tools.account_helpers import build_account_entry

        conn = MagicMock()
        conn.id = 42
        conn.provider_account_id = "user@example.com"
        conn.scopes = "read write"

        with patch("seer.services.integrations.auth.helpers.get_connection_display_name") as mock_display, \
             patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has:
            mock_display.return_value = "User (user@example.com)"
            mock_has.return_value = True

            entry = build_account_entry(conn, ["read", "write"])

            assert entry["id"] == 42
            assert entry["provider_account_id"] == "user@example.com"
            assert entry["display_name"] == "User (user@example.com)"
            assert entry["has_required_scopes"] is True
            assert entry["missing_scopes"] == []

    def test_fallback_provider_account_id(self):
        """Test fallback when provider_account_id is None."""
        from seer.tools.account_helpers import build_account_entry

        conn = MagicMock()
        conn.id = 99
        conn.provider_account_id = None
        conn.scopes = "read"

        with patch("seer.services.integrations.auth.helpers.get_connection_display_name") as mock_display, \
             patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has:
            mock_display.return_value = "Unknown Account"
            mock_has.return_value = True

            entry = build_account_entry(conn, [])

            assert entry["provider_account_id"] == "ID:99"

    def test_includes_missing_scopes(self):
        """Test that missing scopes are correctly included."""
        from seer.tools.account_helpers import build_account_entry

        conn = MagicMock()
        conn.id = 1
        conn.provider_account_id = "test"
        conn.scopes = "scope_a"

        with patch("seer.services.integrations.auth.helpers.get_connection_display_name") as mock_display, \
             patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has, \
             patch("seer.services.integrations.auth.helpers.parse_scopes") as mock_parse:
            mock_display.return_value = "Test"
            mock_has.return_value = False
            mock_parse.return_value = {"scope_a"}

            entry = build_account_entry(conn, ["scope_a", "scope_b"])

            assert entry["has_required_scopes"] is False
            assert entry["missing_scopes"] == ["scope_b"]


class TestMakeErrorResponse:
    """Tests for make_error_response function."""

    def test_tool_error_response(self):
        """Test creating error response for tools."""
        from seer.tools.account_helpers import make_error_response

        response = make_error_response("tool_name", "gmail_send_email", "Tool not found")

        assert response["tool_name"] == "gmail_send_email"
        assert response["provider"] is None
        assert response["accounts"] == []
        assert response["requires_selection"] is False
        assert response["error"] == "Tool not found"

    def test_trigger_error_response(self):
        """Test creating error response for triggers."""
        from seer.tools.account_helpers import make_error_response

        response = make_error_response("trigger_key", "poll.gmail.email_received", "Trigger not found")

        assert response["trigger_key"] == "poll.gmail.email_received"
        assert response["error"] == "Trigger not found"


class TestMakeNoOauthResponse:
    """Tests for make_no_oauth_response function."""

    def test_tool_no_oauth_response(self):
        """Test creating no-OAuth response for tools."""
        from seer.tools.account_helpers import make_no_oauth_response

        response = make_no_oauth_response(
            "tool_name",
            "local_tool",
            None,
            "This tool does not require OAuth"
        )

        assert response["tool_name"] == "local_tool"
        assert response["provider"] is None
        assert response["accounts"] == []
        assert response["requires_selection"] is False
        assert response["message"] == "This tool does not require OAuth"

    def test_trigger_no_oauth_response_with_provider(self):
        """Test creating no-OAuth response with provider specified."""
        from seer.tools.account_helpers import make_no_oauth_response

        response = make_no_oauth_response(
            "trigger_key",
            "webhook.generic",
            "webhook",
            "This trigger does not require OAuth authentication"
        )

        assert response["trigger_key"] == "webhook.generic"
        assert response["provider"] == "webhook"
        assert response["message"] == "This trigger does not require OAuth authentication"
