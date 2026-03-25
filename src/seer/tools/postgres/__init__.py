"""
PostgreSQL tools module.

Provides 9 tools for PostgreSQL database operations:
- postgres_list_schemas: List database schemas
- postgres_list_objects: List tables/views/functions in a schema
- postgres_get_object_details: Get columns/constraints/indexes for a table
- postgres_execute_sql: Execute SQL (validates in restricted mode)
- postgres_explain_query: Get query execution plan
- postgres_get_top_queries: Top queries from pg_stat_statements
- postgres_analyze_workload_indexes: Suggest indexes for workload
- postgres_analyze_query_indexes: Suggest indexes for specific queries
- postgres_analyze_db_health: Database health metrics
"""
from __future__ import annotations

from seer.logger import get_logger
from seer.tools.base import register_tool
from seer.tools.postgres.tools import (
    PostgresAnalyzeDbHealthTool,
    PostgresAnalyzeQueryIndexesTool,
    PostgresAnalyzeWorkloadIndexesTool,
    PostgresExecuteSqlTool,
    PostgresExplainQueryTool,
    PostgresGetObjectDetailsTool,
    PostgresGetTopQueriesTool,
    PostgresListObjectsTool,
    PostgresListSchemasTool,
)

logger = get_logger("shared.tools.postgres")


def register_postgres_tools() -> None:
    """Register all PostgreSQL tools."""
    register_tool(PostgresListSchemasTool())
    register_tool(PostgresListObjectsTool())
    register_tool(PostgresGetObjectDetailsTool())
    register_tool(PostgresExecuteSqlTool())
    register_tool(PostgresExplainQueryTool())
    register_tool(PostgresGetTopQueriesTool())
    register_tool(PostgresAnalyzeWorkloadIndexesTool())
    register_tool(PostgresAnalyzeQueryIndexesTool())
    register_tool(PostgresAnalyzeDbHealthTool())


__all__ = [
    "register_postgres_tools",
    "PostgresListSchemasTool",
    "PostgresListObjectsTool",
    "PostgresGetObjectDetailsTool",
    "PostgresExecuteSqlTool",
    "PostgresExplainQueryTool",
    "PostgresGetTopQueriesTool",
    "PostgresAnalyzeWorkloadIndexesTool",
    "PostgresAnalyzeQueryIndexesTool",
    "PostgresAnalyzeDbHealthTool",
]
