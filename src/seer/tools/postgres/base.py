"""
PostgreSQL tools base class and connection helpers.

This module provides:
- PostgresTool base class with integration metadata
- Connection management using asyncpg
- Credential resolution helpers
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import asyncpg
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool, ResourcePickerConfig
from seer.tools.postgres.common import (
    POSTGRES_DATABASE_PICKER,
    POSTGRES_DEFAULT_RESOURCE,
    AccessMode,
)

if TYPE_CHECKING:
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.postgres")


def _get_access_mode(credentials: "ResolvedCredentials") -> AccessMode:
    """
    Extract access mode from credential resource metadata.

    Defaults to RESTRICTED if not specified.
    """
    if not credentials or not credentials.resource:
        return AccessMode.RESTRICTED

    metadata = getattr(credentials.resource, "resource_metadata", None) or {}
    mode_str = metadata.get("access_mode", "restricted")

    try:
        return AccessMode(mode_str.lower())
    except ValueError:
        logger.warning("Invalid access_mode '%s', defaulting to restricted", mode_str)
        return AccessMode.RESTRICTED


def _require_postgres_connection_string(credentials: Optional["ResolvedCredentials"]) -> str:
    """
    Extract and validate PostgreSQL connection string from credentials.

    Args:
        credentials: Resolved credentials containing secrets

    Returns:
        PostgreSQL connection string

    Raises:
        HTTPException: If credentials or connection string is missing
    """
    if not credentials:
        raise HTTPException(status_code=400, detail="PostgreSQL database binding is required.")

    if not credentials.resource:
        raise HTTPException(status_code=400, detail="PostgreSQL database resource is required.")

    connection_string = (credentials.secrets or {}).get("postgres_connection_string")
    if not connection_string:
        raise HTTPException(
            status_code=400,
            detail="PostgreSQL database is missing connection string. Please re-bind the database.",
        )

    return connection_string


async def get_connection(
    credentials: "ResolvedCredentials",
    *,
    timeout: float = 30.0,
    statement_timeout: Optional[int] = None,
) -> asyncpg.Connection:
    """
    Create an asyncpg connection from resolved credentials.

    Args:
        credentials: Resolved credentials containing postgres_connection_string
        timeout: Connection timeout in seconds
        statement_timeout: Optional statement timeout in milliseconds

    Returns:
        asyncpg.Connection instance (caller must close)

    Raises:
        HTTPException: If connection fails
    """
    connection_string = _require_postgres_connection_string(credentials)

    try:
        # Build connection options
        server_settings = {}
        if statement_timeout:
            server_settings["statement_timeout"] = str(statement_timeout)

        conn = await asyncpg.connect(
            connection_string,
            timeout=timeout,
            server_settings=server_settings if server_settings else None,
        )
        return conn

    except asyncpg.InvalidCatalogNameError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Database does not exist: {str(e)}",
        ) from e
    except asyncpg.InvalidPasswordError as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid database credentials. Please check username and password.",
        ) from e
    except asyncpg.PostgresConnectionError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to PostgreSQL server: {str(e)}",
        ) from e
    except OSError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Network error connecting to PostgreSQL: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception("Unexpected PostgreSQL connection error")
        raise HTTPException(
            status_code=500,
            detail=f"PostgreSQL connection failed: {str(e)}",
        ) from e


class PostgresTool(BaseTool):
    """
    Base class for PostgreSQL tools.

    Provides:
    - Integration metadata for credential resolution
    - Resource picker configuration for database selection
    - Common connection and access mode handling
    """

    integration_type = "postgres"
    provider = "postgres"
    required_scopes: list[str] = []  # No OAuth scopes, uses connection string
    required_secrets = ["postgres_connection_string"]
    default_resource = POSTGRES_DEFAULT_RESOURCE

    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        """Return resource picker for database selection."""
        return {"integration_resource_id": dict(POSTGRES_DATABASE_PICKER)}

    def _get_base_parameters(self) -> Dict[str, Any]:
        """Common parameter schema for all Postgres tools."""
        return {
            "integration_resource_id": {
                "type": "integer",
                "description": "Persisted PostgreSQL database resource ID.",
            },
        }
