from __future__ import annotations

import logging
import os
from typing import Any, Dict
from urllib.parse import urlparse

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Ensure environment variables from .env are available when running locally.
try:
    load_dotenv()
except PermissionError:
    logger.warning(
        "Could not read .env file due to insufficient permissions. "
        "Continuing with existing environment variables.",
    )


def _get_bool(name: str, default: str = "false") -> bool:
    """Read boolean-ish environment variables safely."""
    value = os.getenv(name, default)
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_postgres_credentials(url: str) -> Dict[str, Any]:
    """Convert a postgres-style DSN into asyncpg credential kwargs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("DATABASE_URL must use postgres:// or postgresql:// scheme")

    database = (parsed.path or "").lstrip("/") or "postgres"
    # Get schema from env var, default to 'public' (PostgreSQL default)
    schema = os.getenv("DB_SCHEMA", "public")
    
    credentials = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username,
        "password": parsed.password,
        "database": database,
        "minsize": DB_MIN_CONNECTIONS,
        "maxsize": DB_MAX_CONNECTIONS,
    }
    
    # Set search_path via server_settings for non-public schemas
    # asyncpg doesn't support "schema" parameter directly
    if schema != "public":
        credentials["server_settings"] = {"search_path": schema}
    
    return credentials


DATABASE_URL = os.getenv("DATABASE_URL")
DB_MAX_CONNECTIONS = int(os.getenv("DB_MAX_CONNECTIONS", "10"))
DB_MIN_CONNECTIONS = int(os.getenv("DB_MIN_CONNECTIONS", "1"))
DB_GENERATE_SCHEMAS = _get_bool("DB_GENERATE_SCHEMAS", "false")

try:
    DB_CREDENTIALS = _parse_postgres_credentials(DATABASE_URL)
except ValueError as exc:
    raise RuntimeError(f"Invalid DATABASE_URL: {exc}") from exc

TORTOISE_ORM: Dict[str, Any] = {
    "connections": {
        "default": {
            "engine": "tortoise.backends.asyncpg",
            "credentials": DB_CREDENTIALS,
        },
    },
    "apps": {
        "models": {
            "models": [
                "api.projects.models",
                "shared.database.models",
                "shared.database.models_oauth",
                "api.workflows.models",
            ],
            "default_connection": "default",
        },
    },
    "use_tz": True,
    "timezone": "UTC",
}


__all__ = [
    "DATABASE_URL",
    "DB_GENERATE_SCHEMAS",
    "DB_MAX_CONNECTIONS",
    "DB_MIN_CONNECTIONS",
    "DB_CREDENTIALS",
    "TORTOISE_ORM",
]


