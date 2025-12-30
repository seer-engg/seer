"""
Reusable async context managers for LangGraph PostgreSQL checkpointers.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def open_checkpointer(
    dsn: str,
    *,
    max_size: int = 10,
    pipeline: bool = False,
    prepare_threshold: int = 0,
):
    """
    Open a pooled AsyncPostgresSaver checkpointer for the caller's lifespan.

    The returned saver keeps its underlying psycopg pool open until the context
    exits, ensuring LangGraph graphs can call methods like ainvoke/aget_state_history
    safely across the entire runtime of the workflow.
    """
    pool = AsyncConnectionPool(
        conninfo=dsn,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": prepare_threshold,
        },
    )

    await pool.open()
    saver = AsyncPostgresSaver(pool)

    # Safe to call more than once; ensures checkpoint tables exist.
    await saver.setup()

    try:
        yield saver
    finally:
        await pool.close()


