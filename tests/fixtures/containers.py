# pylint: disable=import-outside-toplevel,redefined-outer-name
# Reason: Test fixtures use lazy imports and pytest fixture pattern requires name reuse
"""
Shared Testcontainers fixtures for PostgreSQL and Redis.

Provides session-scoped containers used by both integration and E2E tests.
Containers start once per pytest session, significantly reducing overhead.
"""
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def postgres_container() -> Generator:
    """
    Session-scoped PostgreSQL container with pgvector extension.

    Uses the pgvector/pgvector:pg17 image to match production environment.
    The container is started once for all tests and cleaned up at session end.

    Yields:
        PostgresContainer: Running PostgreSQL container instance
    """
    from testcontainers.postgres import PostgresContainer

    # Use pgvector image to match production with vector support
    container = PostgresContainer(
        image="pgvector/pgvector:pg17",
        username="test",
        password="test",
        dbname="seer_test",
    )

    with container as pg:
        # Enable pgvector extension after container starts
        import psycopg2
        conn = psycopg2.connect(
            host=pg.get_container_host_ip(),
            port=pg.get_exposed_port(5432),
            user="test",  # psycopg2 uses 'user', not 'username'
            password="test",
            database="seer_test",
        )
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
        finally:
            conn.close()

        yield pg


@pytest.fixture(scope="session")
def redis_container() -> Generator:
    """
    Session-scoped Redis/Valkey container.

    Uses valkey/valkey:7-alpine to match production Valkey deployment.
    Valkey is a Redis-compatible fork maintained by Linux Foundation.

    Yields:
        RedisContainer: Running Redis-compatible container instance
    """
    from testcontainers.redis import RedisContainer

    # Use Valkey image to match production
    container = RedisContainer(image="valkey/valkey:7-alpine")

    with container as redis:
        yield redis


@pytest.fixture(scope="session")
def database_url(postgres_container) -> str:
    """
    PostgreSQL connection URL from the running container.

    Args:
        postgres_container: Running PostgreSQL container

    Returns:
        str: Database URL in postgresql:// format
    """
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    return f"postgresql://test:test@{host}:{port}/seer_test"


@pytest.fixture(scope="session")
def redis_url(redis_container) -> str:
    """
    Redis connection URL from the running container.

    Args:
        redis_container: Running Redis container

    Returns:
        str: Redis URL in redis:// format
    """
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}/0"


__all__ = [
    "postgres_container",
    "redis_container",
    "database_url",
    "redis_url",
]
