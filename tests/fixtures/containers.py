# pylint: disable=import-outside-toplevel,redefined-outer-name
# Reason: Test fixtures use lazy imports and pytest fixture pattern requires name reuse
"""
Shared Testcontainers fixtures for PostgreSQL and Redis.

Provides session-scoped containers used by both integration and E2E tests.
Containers start once per pytest session, significantly reducing overhead.

In CI (GitHub Actions), PostgreSQL/Redis are provided as service containers
and DATABASE_URL/REDIS_URL are already set — Testcontainers is skipped.
"""
import os
from typing import Generator, Optional

import pytest  # noqa: F401


def _ci_database_url() -> Optional[str]:
    """
    Return DATABASE_URL if running in CI with a pre-provisioned PostgreSQL.

    CI environments (GitHub Actions) provide PostgreSQL as a service container.
    We detect this via the CI/GITHUB_ACTIONS env vars, NOT by probing
    localhost — to avoid accidentally using a developer's local PostgreSQL.
    """
    is_ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
    if not is_ci:
        return None
    return os.environ.get("DATABASE_URL")


@pytest.fixture(scope="session")
def postgres_container() -> Generator:
    """
    Session-scoped PostgreSQL container with pgvector extension.

    In CI, where PostgreSQL is a service container, this yields None
    and database_url uses DATABASE_URL directly.

    Locally, starts a Testcontainers PostgreSQL instance.
    """
    ci_url = _ci_database_url()
    if ci_url is not None:
        # CI: PostgreSQL is already running as a service container
        yield None
        return

    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        image="pgvector/pgvector:pg17",
        username="test",
        password="test",
        dbname="seer_test",
    )

    with container as pg:
        import psycopg2
        conn = psycopg2.connect(
            host=pg.get_container_host_ip(),
            port=pg.get_exposed_port(5432),
            user="test",
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
    """
    from testcontainers.redis import RedisContainer

    container = RedisContainer(image="valkey/valkey:7-alpine")

    with container as redis:
        yield redis


@pytest.fixture(scope="session")
def database_url(postgres_container) -> str:
    """
    PostgreSQL connection URL.

    Uses CI's DATABASE_URL when available, otherwise builds URL from
    the Testcontainers instance.
    """
    ci_url = _ci_database_url()
    if ci_url is not None:
        return ci_url

    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    return f"postgresql://test:test@{host}:{port}/seer_test"


@pytest.fixture(scope="session")
def redis_url(redis_container) -> str:
    """
    Redis connection URL from the running container.
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
