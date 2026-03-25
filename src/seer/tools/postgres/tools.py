"""
PostgreSQL tool implementations.

This module provides 9 PostgreSQL tools for database operations:
1. postgres_list_schemas - List database schemas
2. postgres_list_objects - List tables/views/functions in a schema
3. postgres_get_object_details - Get columns/constraints/indexes for a table
4. postgres_execute_sql - Execute SQL (validates in restricted mode)
5. postgres_explain_query - Get query execution plan
6. postgres_get_top_queries - Top queries from pg_stat_statements
7. postgres_analyze_workload_indexes - Suggest indexes for workload
8. postgres_analyze_query_indexes - Suggest indexes for specific queries
9. postgres_analyze_db_health - Database health metrics

All tools respect the access_mode configured on the database binding:
- restricted: Only SELECT, EXPLAIN, SHOW allowed
- unrestricted: All SQL operations allowed
"""
# pylint: disable=too-many-lines
# Reason: This module implements all 9 PostgreSQL tools with their full SQL logic in one place for cohesion.
# TODO: Consider splitting into schema_tools.py, query_tools.py, and analysis_tools.py in a future refactor.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.postgres.base import PostgresTool, _get_access_mode, get_connection
from seer.tools.postgres.common import AccessMode, validate_sql_for_restricted_mode

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.postgres.tools")


class PostgresListSchemasTool(PostgresTool):
    """List all schemas in the database."""

    name = "postgres_list_schemas"
    description = "List all schemas in the PostgreSQL database, excluding system schemas."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self._get_base_parameters(),
            "required": ["integration_resource_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "schema_name": {"type": "string"},
                    "owner": {"type": "string"},
                },
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> List[Dict[str, Any]]:
        _ = access_token, context  # Unused for Postgres tools

        conn = await get_connection(credentials)
        try:
            rows = await conn.fetch("""
                SELECT
                    n.nspname AS schema_name,
                    pg_catalog.pg_get_userbyid(n.nspowner) AS owner
                FROM pg_catalog.pg_namespace n
                WHERE n.nspname NOT LIKE 'pg_%'
                  AND n.nspname <> 'information_schema'
                ORDER BY n.nspname
            """)
            return [dict(row) for row in rows]
        finally:
            await conn.close()


class PostgresListObjectsTool(PostgresTool):
    """List tables, views, and functions in a schema."""

    name = "postgres_list_objects"
    description = "List tables, views, and functions in a PostgreSQL schema."

    def get_parameters_schema(self) -> Dict[str, Any]:
        props = self._get_base_parameters()
        props.update({
            "schema": {
                "type": "string",
                "description": "Schema name (defaults to 'public').",
                "default": "public",
            },
            "object_type": {
                "type": "string",
                "description": "Filter by object type: 'table', 'view', 'function', or 'all'.",
                "enum": ["table", "view", "function", "all"],
                "default": "all",
            },
        })
        return {
            "type": "object",
            "properties": props,
            "required": ["integration_resource_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "owner": {"type": "string"},
                },
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> List[Dict[str, Any]]:
        _ = access_token, context

        schema = arguments.get("schema", "public")
        object_type = arguments.get("object_type", "all")

        conn = await get_connection(credentials)
        try:
            results = []

            # Tables and views
            if object_type in ("table", "view", "all"):
                table_filter = ""
                if object_type == "table":
                    table_filter = "AND c.relkind = 'r'"
                elif object_type == "view":
                    table_filter = "AND c.relkind IN ('v', 'm')"

                rows = await conn.fetch(f"""
                    SELECT
                        c.relname AS name,
                        CASE c.relkind
                            WHEN 'r' THEN 'table'
                            WHEN 'v' THEN 'view'
                            WHEN 'm' THEN 'materialized_view'
                            WHEN 'p' THEN 'partitioned_table'
                        END AS type,
                        pg_catalog.pg_get_userbyid(c.relowner) AS owner
                    FROM pg_catalog.pg_class c
                    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = $1
                      AND c.relkind IN ('r', 'v', 'm', 'p')
                      {table_filter}
                    ORDER BY c.relname
                """, schema)
                results.extend([dict(row) for row in rows])

            # Functions
            if object_type in ("function", "all"):
                rows = await conn.fetch("""
                    SELECT
                        p.proname AS name,
                        'function' AS type,
                        pg_catalog.pg_get_userbyid(p.proowner) AS owner
                    FROM pg_catalog.pg_proc p
                    JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = $1
                      AND p.prokind IN ('f', 'p')
                    ORDER BY p.proname
                """, schema)
                results.extend([dict(row) for row in rows])

            return results
        finally:
            await conn.close()


class PostgresGetObjectDetailsTool(PostgresTool):
    """Get detailed information about a table including columns, constraints, and indexes."""

    name = "postgres_get_object_details"
    description = "Get columns, constraints, and indexes for a PostgreSQL table or view."

    def get_parameters_schema(self) -> Dict[str, Any]:
        props = self._get_base_parameters()
        props.update({
            "schema": {
                "type": "string",
                "description": "Schema name.",
                "default": "public",
            },
            "table_name": {
                "type": "string",
                "description": "Table or view name.",
            },
        })
        return {
            "type": "object",
            "properties": props,
            "required": ["integration_resource_id", "table_name"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array"},
                "constraints": {"type": "array"},
                "indexes": {"type": "array"},
                "row_estimate": {"type": "integer"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        schema = arguments.get("schema", "public")
        table_name = arguments["table_name"]

        conn = await get_connection(credentials)
        try:
            # Get columns
            columns = await conn.fetch("""
                SELECT
                    a.attname AS column_name,
                    pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
                    a.attnotnull AS not_null,
                    pg_catalog.pg_get_expr(d.adbin, d.adrelid) AS default_value,
                    col_description(c.oid, a.attnum) AS description
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
                LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
                WHERE n.nspname = $1
                  AND c.relname = $2
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                ORDER BY a.attnum
            """, schema, table_name)

            # Get constraints
            constraints = await conn.fetch("""
                SELECT
                    con.conname AS constraint_name,
                    CASE con.contype
                        WHEN 'p' THEN 'PRIMARY KEY'
                        WHEN 'u' THEN 'UNIQUE'
                        WHEN 'f' THEN 'FOREIGN KEY'
                        WHEN 'c' THEN 'CHECK'
                        WHEN 'x' THEN 'EXCLUSION'
                    END AS constraint_type,
                    pg_catalog.pg_get_constraintdef(con.oid, true) AS definition
                FROM pg_catalog.pg_constraint con
                JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = $1
                  AND c.relname = $2
                ORDER BY con.contype, con.conname
            """, schema, table_name)

            # Get indexes
            indexes = await conn.fetch("""
                SELECT
                    i.relname AS index_name,
                    am.amname AS index_type,
                    pg_catalog.pg_get_indexdef(i.oid) AS definition,
                    idx.indisunique AS is_unique,
                    idx.indisprimary AS is_primary
                FROM pg_catalog.pg_index idx
                JOIN pg_catalog.pg_class c ON c.oid = idx.indrelid
                JOIN pg_catalog.pg_class i ON i.oid = idx.indexrelid
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_catalog.pg_am am ON am.oid = i.relam
                WHERE n.nspname = $1
                  AND c.relname = $2
                ORDER BY i.relname
            """, schema, table_name)

            # Get row estimate
            row_estimate = await conn.fetchval("""
                SELECT c.reltuples::bigint AS row_estimate
                FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = $1 AND c.relname = $2
            """, schema, table_name)

            return {
                "columns": [dict(row) for row in columns],
                "constraints": [dict(row) for row in constraints],
                "indexes": [dict(row) for row in indexes],
                "row_estimate": row_estimate or 0,
            }
        finally:
            await conn.close()


class PostgresExecuteSqlTool(PostgresTool):
    """Execute SQL queries with access mode enforcement."""

    name = "postgres_execute_sql"
    description = (
        "Execute a SQL query on the PostgreSQL database. "
        "In restricted mode, only SELECT, EXPLAIN, and SHOW statements are allowed."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        props = self._get_base_parameters()
        props.update({
            "sql": {
                "type": "string",
                "description": "SQL query to execute.",
            },
            "params": {
                "type": "array",
                "description": "Optional query parameters for parameterized queries ($1, $2, etc.).",
                "items": {},
            },
        })
        return {
            "type": "object",
            "properties": props,
            "required": ["integration_resource_id", "sql"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rows": {"type": "array"},
                "row_count": {"type": "integer"},
                "columns": {"type": "array", "items": {"type": "string"}},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        sql = arguments["sql"]
        params = arguments.get("params") or []

        # Check access mode and validate SQL
        access_mode = _get_access_mode(credentials)
        if access_mode == AccessMode.RESTRICTED:
            is_valid, error = validate_sql_for_restricted_mode(sql)
            if not is_valid:
                raise HTTPException(
                    status_code=403,
                    detail=f"Query blocked by restricted access mode: {error}",
                )

        conn = await get_connection(credentials, statement_timeout=30000)
        try:
            # Execute query
            if params:
                rows = await conn.fetch(sql, *params)
            else:
                rows = await conn.fetch(sql)

            # Extract column names from first row
            columns = []
            if rows:
                columns = list(rows[0].keys())

            return {
                "rows": [dict(row) for row in rows],
                "row_count": len(rows),
                "columns": columns,
            }

        except Exception as e:
            logger.exception("PostgreSQL query execution error")
            raise HTTPException(
                status_code=500,
                detail=f"Query execution failed: {str(e)}",
            ) from e
        finally:
            await conn.close()


class PostgresExplainQueryTool(PostgresTool):
    """Get the execution plan for a SQL query."""

    name = "postgres_explain_query"
    description = "Get the execution plan (EXPLAIN ANALYZE) for a SQL query."

    def get_parameters_schema(self) -> Dict[str, Any]:
        props = self._get_base_parameters()
        props.update({
            "sql": {
                "type": "string",
                "description": "SQL query to analyze.",
            },
            "analyze": {
                "type": "boolean",
                "description": "Run EXPLAIN ANALYZE (actually executes the query). Default: false.",
                "default": False,
            },
            "format": {
                "type": "string",
                "description": "Output format: 'text', 'json', 'yaml', or 'xml'.",
                "enum": ["text", "json", "yaml", "xml"],
                "default": "json",
            },
        })
        return {
            "type": "object",
            "properties": props,
            "required": ["integration_resource_id", "sql"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan": {"type": "array"},
                "execution_time_ms": {"type": "number"},
                "planning_time_ms": {"type": "number"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        sql = arguments["sql"]
        analyze = arguments.get("analyze", False)
        output_format = arguments.get("format", "json")

        # For ANALYZE, check if restricted mode allows the underlying query
        access_mode = _get_access_mode(credentials)
        if analyze and access_mode == AccessMode.RESTRICTED:
            is_valid, error = validate_sql_for_restricted_mode(sql)
            if not is_valid:
                raise HTTPException(
                    status_code=403,
                    detail=f"Cannot ANALYZE query in restricted mode: {error}",
                )

        conn = await get_connection(credentials, statement_timeout=60000)
        try:
            explain_sql = f"EXPLAIN (FORMAT {output_format.upper()}"
            if analyze:
                explain_sql += ", ANALYZE, BUFFERS"
            explain_sql += f") {sql}"

            rows = await conn.fetch(explain_sql)

            if output_format == "json":
                # JSON format returns plan as JSONB in first column
                plan = rows[0][0] if rows else []
                result = {"plan": plan}

                # Extract timing if available
                if plan and isinstance(plan, list) and plan:
                    plan_obj = plan[0] if isinstance(plan[0], dict) else {}
                    if "Execution Time" in plan_obj:
                        result["execution_time_ms"] = plan_obj["Execution Time"]
                    if "Planning Time" in plan_obj:
                        result["planning_time_ms"] = plan_obj["Planning Time"]

                return result
            # Text/YAML/XML format returns lines
            return {"plan": [row[0] for row in rows]}

        finally:
            await conn.close()


class PostgresGetTopQueriesTool(PostgresTool):
    """Get top queries from pg_stat_statements."""

    name = "postgres_get_top_queries"
    description = (
        "Get the top queries by execution time from pg_stat_statements. "
        "Requires pg_stat_statements extension to be installed."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        props = self._get_base_parameters()
        props.update({
            "limit": {
                "type": "integer",
                "description": "Maximum number of queries to return.",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
            },
            "order_by": {
                "type": "string",
                "description": "Metric to order by.",
                "enum": ["total_time", "mean_time", "calls", "rows"],
                "default": "total_time",
            },
        })
        return {
            "type": "object",
            "properties": props,
            "required": ["integration_resource_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "calls": {"type": "integer"},
                    "total_time_ms": {"type": "number"},
                    "mean_time_ms": {"type": "number"},
                    "rows": {"type": "integer"},
                },
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> List[Dict[str, Any]]:
        _ = access_token, context

        limit = arguments.get("limit", 10)
        order_by = arguments.get("order_by", "total_time")

        order_column = {
            "total_time": "total_exec_time",
            "mean_time": "mean_exec_time",
            "calls": "calls",
            "rows": "rows",
        }.get(order_by, "total_exec_time")

        conn = await get_connection(credentials)
        try:
            # Check if pg_stat_statements is available
            ext_check = await conn.fetchval("""
                SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
            """)

            if not ext_check:
                raise HTTPException(
                    status_code=400,
                    detail="pg_stat_statements extension is not installed. "
                           "Run: CREATE EXTENSION pg_stat_statements;",
                )

            rows = await conn.fetch(f"""
                SELECT
                    query,
                    calls,
                    total_exec_time AS total_time_ms,
                    mean_exec_time AS mean_time_ms,
                    rows
                FROM pg_stat_statements
                WHERE userid = (SELECT usesysid FROM pg_user WHERE usename = current_user)
                ORDER BY {order_column} DESC
                LIMIT $1
            """, limit)

            return [dict(row) for row in rows]

        except HTTPException:
            raise
        except Exception as e:
            # pg_stat_statements might not be accessible
            if "permission denied" in str(e).lower():
                raise HTTPException(
                    status_code=403,
                    detail="Permission denied for pg_stat_statements. "
                           "Grant SELECT on pg_stat_statements to the database user.",
                ) from e
            raise HTTPException(
                status_code=500,
                detail=f"Failed to query pg_stat_statements: {str(e)}",
            ) from e
        finally:
            await conn.close()


class PostgresAnalyzeWorkloadIndexesTool(PostgresTool):
    """Analyze workload and suggest indexes based on pg_stat_statements."""

    name = "postgres_analyze_workload_indexes"
    description = (
        "Analyze database workload from pg_stat_statements and suggest indexes. "
        "Requires pg_stat_statements extension."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        props = self._get_base_parameters()
        props.update({
            "min_calls": {
                "type": "integer",
                "description": "Minimum number of calls for a query to be considered.",
                "default": 100,
            },
            "min_total_time_ms": {
                "type": "number",
                "description": "Minimum total execution time (ms) for a query to be considered.",
                "default": 1000,
            },
        })
        return {
            "type": "object",
            "properties": props,
            "required": ["integration_resource_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "suggestions": {"type": "array"},
                "analyzed_queries": {"type": "integer"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        min_calls = arguments.get("min_calls", 100)
        min_total_time = arguments.get("min_total_time_ms", 1000)

        conn = await get_connection(credentials)
        try:
            # Get slow queries with sequential scans
            rows = await conn.fetch("""
                SELECT
                    query,
                    calls,
                    total_exec_time AS total_time_ms,
                    mean_exec_time AS mean_time_ms
                FROM pg_stat_statements
                WHERE calls >= $1
                  AND total_exec_time >= $2
                  AND query !~* '^(COMMIT|BEGIN|ROLLBACK|SET|SHOW)'
                ORDER BY total_exec_time DESC
                LIMIT 20
            """, min_calls, min_total_time)

            suggestions = []
            for row in rows:
                query = row["query"]
                # Simple heuristic: look for WHERE clauses without index hints
                if "WHERE" in query.upper() and "USING INDEX" not in query.upper():
                    suggestions.append({
                        "query_preview": query[:200] + "..." if len(query) > 200 else query,
                        "calls": row["calls"],
                        "total_time_ms": row["total_time_ms"],
                        "mean_time_ms": row["mean_time_ms"],
                        "suggestion": "Consider adding an index on columns used in WHERE clause",
                    })

            return {
                "suggestions": suggestions,
                "analyzed_queries": len(rows),
            }

        except Exception as e:
            if "pg_stat_statements" in str(e):
                raise HTTPException(
                    status_code=400,
                    detail="pg_stat_statements extension is not available or accessible.",
                ) from e
            raise HTTPException(
                status_code=500,
                detail=f"Workload analysis failed: {str(e)}",
            ) from e
        finally:
            await conn.close()


class PostgresAnalyzeQueryIndexesTool(PostgresTool):
    """Analyze a specific query and suggest indexes."""

    name = "postgres_analyze_query_indexes"
    description = "Analyze a specific SQL query and suggest indexes to improve performance."

    def get_parameters_schema(self) -> Dict[str, Any]:
        props = self._get_base_parameters()
        props.update({
            "sql": {
                "type": "string",
                "description": "SQL query to analyze.",
            },
        })
        return {
            "type": "object",
            "properties": props,
            "required": ["integration_resource_id", "sql"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan": {"type": "array"},
                "suggestions": {"type": "array"},
                "seq_scans": {"type": "array"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        sql = arguments["sql"]

        conn = await get_connection(credentials)
        try:
            # Get execution plan
            rows = await conn.fetch(f"EXPLAIN (FORMAT JSON) {sql}")
            plan = rows[0][0] if rows else []

            # Analyze plan for sequential scans
            seq_scans = []
            suggestions = []

            def find_seq_scans(node: dict, depth: int = 0):
                if not isinstance(node, dict):
                    return

                node_type = node.get("Node Type", "")
                if node_type == "Seq Scan":
                    table = node.get("Relation Name", "unknown")
                    filter_cond = node.get("Filter", "")
                    seq_scans.append({
                        "table": table,
                        "filter": filter_cond,
                        "rows": node.get("Plan Rows", 0),
                    })
                    if filter_cond:
                        suggestions.append({
                            "type": "index",
                            "table": table,
                            "reason": f"Sequential scan with filter: {filter_cond}",
                            "suggestion": f"Consider adding an index on {table} for the filter columns",
                        })

                # Recurse into child plans
                for child in node.get("Plans", []):
                    find_seq_scans(child, depth + 1)

            if plan and isinstance(plan, list) and plan:
                plan_root = plan[0].get("Plan", {})
                find_seq_scans(plan_root)

            return {
                "plan": plan,
                "suggestions": suggestions,
                "seq_scans": seq_scans,
            }

        finally:
            await conn.close()


class PostgresAnalyzeDbHealthTool(PostgresTool):
    """Get database health metrics."""

    name = "postgres_analyze_db_health"
    description = "Get database health metrics including connections, cache hit ratio, and bloat."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self._get_base_parameters(),
            "required": ["integration_resource_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "database_size": {"type": "string"},
                "active_connections": {"type": "integer"},
                "max_connections": {"type": "integer"},
                "cache_hit_ratio": {"type": "number"},
                "deadlocks": {"type": "integer"},
                "conflicts": {"type": "integer"},
                "temp_files": {"type": "integer"},
                "uptime": {"type": "string"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        conn = await get_connection(credentials)
        try:
            # Database size
            db_size = await conn.fetchval("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)

            # Connection stats
            conn_stats = await conn.fetchrow("""
                SELECT
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') AS active_connections,
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') AS max_connections
            """)

            # Cache hit ratio
            cache_stats = await conn.fetchrow("""
                SELECT
                    CASE
                        WHEN (blks_hit + blks_read) > 0
                        THEN round(100.0 * blks_hit / (blks_hit + blks_read), 2)
                        ELSE 0
                    END AS cache_hit_ratio
                FROM pg_stat_database
                WHERE datname = current_database()
            """)

            # Database stats
            db_stats = await conn.fetchrow("""
                SELECT
                    deadlocks,
                    conflicts,
                    temp_files
                FROM pg_stat_database
                WHERE datname = current_database()
            """)

            # Uptime
            uptime = await conn.fetchval("""
                SELECT current_timestamp - pg_postmaster_start_time()
            """)

            return {
                "database_size": db_size,
                "active_connections": conn_stats["active_connections"],
                "max_connections": conn_stats["max_connections"],
                "cache_hit_ratio": float(cache_stats["cache_hit_ratio"] or 0),
                "deadlocks": db_stats["deadlocks"] or 0,
                "conflicts": db_stats["conflicts"] or 0,
                "temp_files": db_stats["temp_files"] or 0,
                "uptime": str(uptime),
            }

        finally:
            await conn.close()
