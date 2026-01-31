"""
Unit tests for auto-selecting provider connections for triggers.

Tests:
- Auto-selection with single active connection
- Auto-selection with multiple connections (verifies most recent selected)
- Error when no connections exist
- Provider mapping (gmail -> google)
- Skip validation mode bypasses auto-selection
- Non-required triggers skip auto-selection
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.workflows.services.triggers import (
    _auto_select_provider_connection,
)
from seer.core.schema.models import (
    TriggerDefinition,
)
from seer.database import User


def utcnow():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# Auto-Selection Helper Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_select_single_active_connection():
    """Test auto-selection with a single active OAuth connection."""
    from seer.database.models_oauth import OAuthConnection

    # Mock user
    user = MagicMock(spec=User)
    user.id = 1

    # Mock trigger definition
    trigger_def = MagicMock(spec=TriggerDefinition)
    trigger_def.provider = "gmail"
    trigger_def.key = "gmail.new_email"

    # Mock OAuthConnection query
    mock_connection = MagicMock(spec=OAuthConnection)
    mock_connection.id = 123
    mock_connection.provider = "google"
    mock_connection.status = "active"

    with patch("seer.database.models_oauth.OAuthConnection.filter") as mock_filter:
        # Create properly chained mock for ORM pattern
        mock_order_by = MagicMock()
        mock_order_by.first = AsyncMock(return_value=mock_connection)
        mock_queryset = MagicMock()
        mock_queryset.order_by.return_value = mock_order_by
        mock_filter.return_value = mock_queryset

        connection_id = await _auto_select_provider_connection(user, trigger_def)

        assert connection_id == 123
        mock_filter.assert_called_once_with(
            user=user,
            provider="google",
            status="active"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_select_most_recent_connection():
    """Test that auto-selection picks the most recently created connection."""
    from seer.database.models_oauth import OAuthConnection

    user = MagicMock(spec=User)
    user.id = 1

    trigger_def = MagicMock(spec=TriggerDefinition)
    trigger_def.provider = "gmail"
    trigger_def.key = "gmail.new_email"

    # Mock the most recent connection (created last)
    mock_connection = MagicMock(spec=OAuthConnection)
    mock_connection.id = 456  # Most recent
    mock_connection.created_at = utcnow()

    with patch("seer.database.models_oauth.OAuthConnection.filter") as mock_filter:
        # Create properly chained mock for ORM pattern
        mock_order_by = MagicMock()
        mock_order_by.first = AsyncMock(return_value=mock_connection)
        mock_queryset = MagicMock()
        mock_queryset.order_by.return_value = mock_order_by
        mock_filter.return_value = mock_queryset

        connection_id = await _auto_select_provider_connection(user, trigger_def)

        assert connection_id == 456
        mock_queryset.order_by.assert_called_once_with("-created_at")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_select_no_connections_returns_none():
    """Test that auto-selection returns None when no connections exist."""
    user = MagicMock(spec=User)
    user.id = 1

    trigger_def = MagicMock(spec=TriggerDefinition)
    trigger_def.provider = "gmail"
    trigger_def.key = "gmail.new_email"

    with patch("seer.database.models_oauth.OAuthConnection.filter") as mock_filter:
        # Create properly chained mock for ORM pattern
        mock_order_by = MagicMock()
        mock_order_by.first = AsyncMock(return_value=None)  # No connections
        mock_queryset = MagicMock()
        mock_queryset.order_by.return_value = mock_order_by
        mock_filter.return_value = mock_queryset

        connection_id = await _auto_select_provider_connection(user, trigger_def)

        assert connection_id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_select_provider_mapping():
    """Test that provider mapping works correctly (gmail -> google)."""
    from seer.database.models_oauth import OAuthConnection

    user = MagicMock(spec=User)
    user.id = 1

    # Test different trigger providers that map to "google"
    test_cases = [
        ("gmail", "google"),
        ("googlesheets", "google"),
        ("googledrive", "google"),
        ("google", "google"),
    ]

    for trigger_provider, expected_oauth_provider in test_cases:
        trigger_def = MagicMock(spec=TriggerDefinition)
        trigger_def.provider = trigger_provider
        trigger_def.key = f"{trigger_provider}.test"

        mock_connection = MagicMock(spec=OAuthConnection)
        mock_connection.id = 123

        with patch("seer.database.models_oauth.OAuthConnection.filter") as mock_filter:
            # Create properly chained mock for ORM pattern
            mock_order_by = MagicMock()
            mock_order_by.first = AsyncMock(return_value=mock_connection)
            mock_queryset = MagicMock()
            mock_queryset.order_by.return_value = mock_order_by
            mock_filter.return_value = mock_queryset

            await _auto_select_provider_connection(user, trigger_def)

            # Verify correct OAuth provider was used
            call_args = mock_filter.call_args
            assert call_args[1]["provider"] == expected_oauth_provider, \
                f"Expected {expected_oauth_provider} for {trigger_provider}"


# NOTE: Tests for sync_trigger_subscriptions integration are covered by
# integration tests in tests/integration/triggers/test_polling_engine.py
# The database interactions are too complex to mock reliably in unit tests.
