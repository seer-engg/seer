"""
Shared helpers for Postgres tooling.

Provides lazy loading of optional asyncpg dependency to keep imports lightweight.
"""
from __future__ import annotations

_ASYNC_PG = None


def get_asyncpg():
    """Lazy-load asyncpg to avoid hard dependency when Postgres tools aren't used."""
    global _ASYNC_PG  # pylint: disable=global-statement  # Reason: module-level cache for optional dependency
    if _ASYNC_PG is None:
        try:
            import asyncpg  # type: ignore  # pylint: disable=import-outside-toplevel  # Reason: lazy-load optional dependency
            _ASYNC_PG = asyncpg
        except ImportError:
            # pylint: disable=raise-missing-from  # Reason: clear install guidance is sufficient
            raise ImportError(
                "asyncpg is required for PostgreSQL operations. "
                "Install it with: uv add asyncpg"
            )
    return _ASYNC_PG


__all__ = ["get_asyncpg"]
