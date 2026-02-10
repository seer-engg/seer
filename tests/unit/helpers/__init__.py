"""
Test helper utilities for common mock patterns.

This module provides reusable utilities to reduce code duplication across unit tests:
- utcnow(): Standard UTC timestamp helper
- ORMQueryMockBuilder: Builder for ORM query chain mocks
- create_oauth_connection_mock(): Factory for OAuth connection mocks
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock


def utcnow():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class ORMQueryMockBuilder:
    """
    Builder for creating properly chained ORM query mocks.

    Tortoise ORM uses method chaining (filter().order_by().limit()) which
    requires careful mock setup. This builder simplifies that process.

    Usage:
        mock_qs = ORMQueryMockBuilder() \\
            .with_filter() \\
            .with_order_by("-created_at") \\
            .with_first(mock_result) \\
            .build()

        Model.filter.return_value = mock_qs
    """

    def __init__(self):
        self._queryset = MagicMock()
        self._current = self._queryset

    def with_filter(self, return_self=True):
        """Add filter() method to chain."""
        if return_self:
            self._queryset.filter.return_value = self._queryset
        return self

    def with_order_by(self):
        """Add order_by() method to chain."""
        order_by_mock = MagicMock()
        self._queryset.order_by.return_value = order_by_mock
        self._current = order_by_mock
        return self

    def with_limit(self, return_value):
        """Add limit() as an async method."""
        self._current.limit = AsyncMock(return_value=return_value)
        return self

    def with_first(self, return_value):
        """Add first() as an async method."""
        self._current.first = AsyncMock(return_value=return_value)
        return self

    def with_all(self, return_value):
        """Add all() as an async method."""
        self._current.all = AsyncMock(return_value=return_value)
        return self

    def with_count(self, return_value):
        """Add count() as an async method."""
        self._current.count = AsyncMock(return_value=return_value)
        return self

    def with_update(self):
        """Add update() as an async method."""
        self._current.update = AsyncMock()
        return self

    def build(self):
        """Return the configured queryset mock."""
        return self._queryset


class AsyncQuerySetMock:
    """
    A proper async-awaitable mock for Tortoise ORM QuerySets.

    Supports both awaiting and async iteration, which is required
    for testing code that iterates over querysets.

    Usage:
        mock_qs = AsyncQuerySetMock([item1, item2])
        async for item in mock_qs:
            ...  # Works!
    """

    def __init__(self, items, with_update=False):
        self._items = items if items is not None else []
        self._update_mock = AsyncMock() if with_update else None

    def __await__(self):
        async def _coro():
            return self._items
        return _coro().__await__()

    def __aiter__(self):
        return self._async_iter()

    async def _async_iter(self):
        for item in self._items:
            yield item

    @property
    def update(self):
        return self._update_mock


def create_oauth_connection_mock(
    connection_id: int = 123,
    provider: str = "google",
    status: str = "active",
):
    """
    Create a mock OAuth connection with common attributes.

    Args:
        connection_id: The connection ID
        provider: OAuth provider name (google, github, etc.)
        status: Connection status (active, revoked, etc.)

    Returns:
        MagicMock configured as an OAuthConnection
    """
    connection = MagicMock()
    connection.id = connection_id
    connection.provider = provider
    connection.status = status
    connection.access_token_enc = "encrypted_token"
    connection.refresh_token_enc = "encrypted_refresh"
    connection.scopes = "email profile"
    connection.provider_account_id = f"account_{connection_id}"
    connection.provider_metadata = {"email": "test@example.com"}
    connection.expires_at = datetime(2025, 12, 31, tzinfo=timezone.utc)
    connection.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    connection.updated_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
    return connection
