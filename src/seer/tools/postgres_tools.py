"""
PostgreSQL tool factory functions for LangChain integration.

This module contains functions to create LangChain-compatible tools
for PostgreSQL database operations (query, execute, schema, batch).
"""
from typing import Any, List, Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt

from seer.logger import get_logger

logger = get_logger("shared.tools.postgres_tools")


def _request_write_approval(statement: str, parameters: Optional[List[Any]] = None) -> str:
    """
    Request human approval for a PostgreSQL write operation using LangGraph interrupt.

    Args:
        statement: The SQL statement to be executed
        parameters: Optional list of statement parameters

    Returns:
        The human's response (approval or rejection)
    """
    from langgraph.types import interrupt

    # Format the approval request message
    params_str = f"\nParameters: {parameters}" if parameters else ""
    approval_request = (
        f"🔒 **PostgreSQL Write Approval Required**\n\n"
        f"The agent wants to execute the following database write operation:\n\n"
        f"```sql\n{statement}\n```{params_str}\n\n"
        f"Do you approve this operation? Reply with 'yes' or 'approve' to proceed, "
        f"or 'no' or 'reject' to cancel."
    )

    # Trigger interrupt and wait for human response
    response = interrupt(approval_request)
    return response


def _is_write_approved(response: Any) -> bool:
    """Check if the human response indicates approval."""
    if response is None:
        return False
    response_str = str(response).strip().lower()
    return response_str in ("yes", "y", "approve", "approved", "ok", "proceed", "true", "1")


def _create_query_tool(client):
    """Create postgres_query tool."""
    @tool
    async def postgres_query(
        query: str,
        parameters: Optional[List[Any]] = None,
    ) -> str:
        """
        Execute a read-only SQL query (SELECT) on the PostgreSQL database.

        Use this tool when you need to retrieve data from the database.
        The query should be a valid PostgreSQL SELECT statement.

        Args:
            query: The SQL SELECT query to execute. Use $1, $2, etc. for parameters.
            parameters: Optional list of query parameters in order.

        Returns:
            Query results as a formatted string.

        Example:
            postgres_query("SELECT * FROM users WHERE status = $1", ["active"])
            postgres_query("SELECT id, name FROM products LIMIT 10")
        """
        try:
            query_upper = query.strip().upper()
            if not query_upper.startswith("SELECT") and not query_upper.startswith("WITH"):
                return (
                    "Error: This tool only supports SELECT queries. "
                    "Use postgres_execute for modifications."
                )

            params = tuple(parameters) if parameters else ()
            results = await client.query(query, *params)

            if not results:
                return "Query returned no results."

            import json
            return json.dumps(results, indent=2, default=str)
        except (TypeError, KeyError) as e:
            logger.error("Data processing error in postgres_query: %s", e)
            return f"Data processing error: {e}"
        except Exception as e:
            # Import asyncpg types dynamically to check
            try:
                from seer.tools.postgres_client import _get_asyncpg
                asyncpg = _get_asyncpg()
                if isinstance(e, (asyncpg.PostgresError, asyncpg.InterfaceError)):
                    logger.error("Database error in postgres_query: %s", e)
                    return f"Database error: {e}"
            except Exception:
                pass
            # Unexpected error - log for debugging
            logger.exception("Unexpected error in postgres_query")
            return f"Unexpected error: {str(e)}"
    return postgres_query


def _create_execute_tool(client):
    """Create postgres_execute tool."""
    @tool
    async def postgres_execute(
        statement: str,
        parameters: Optional[List[Any]] = None,
    ) -> str:
        """
        Execute a write SQL statement (INSERT, UPDATE, DELETE, CREATE, ALTER, DROP)
        on the PostgreSQL database.

        Use this tool when you need to modify data or schema in the database.

        Note: This operation may require human approval depending on configuration.
        If approval is required, the tool will pause and wait for confirmation.

        Args:
            statement: The SQL statement to execute. Use $1, $2, etc. for parameters.
            parameters: Optional list of statement parameters in order.

        Returns:
            Status message indicating the result (e.g., "INSERT 0 1", "UPDATE 5").

        Example:
            postgres_execute(
                "INSERT INTO users (name, email) VALUES ($1, $2)",
                ["John", "john@example.com"]
            )
            postgres_execute("UPDATE products SET price = $1 WHERE id = $2", [29.99, 123])
            postgres_execute("DELETE FROM sessions WHERE expires_at < NOW()")
        """
        try:
            from seer.config import config
            if config.postgres_write_requires_approval:
                logger.info("Write approval required for statement: %s...", statement[:100])
                response = _request_write_approval(statement, parameters)

                if not _is_write_approved(response):
                    logger.info("Write operation rejected by user: %s", response)
                    return (
                        f"Operation cancelled: User rejected the write operation. "
                        f"Response: {response}"
                    )

                logger.info("Write operation approved by user")

            params = tuple(parameters) if parameters else ()
            result = await client.execute(statement, *params)
            return f"Statement executed successfully: {result}"

        except GraphInterrupt:
            raise
        except ValueError as e:
            logger.error("Value error in postgres_execute: %s", e)
            return f"Value error: {e}"
        except Exception as e:
            # Import asyncpg types dynamically to check
            try:
                from seer.tools.postgres_client import _get_asyncpg
                asyncpg = _get_asyncpg()
                if isinstance(e, (asyncpg.PostgresError, asyncpg.InterfaceError)):
                    logger.error("Database error in postgres_execute: %s", e)
                    return f"Database error: {e}"
            except Exception:
                pass
            # Unexpected error - log for debugging
            logger.exception("Unexpected error in postgres_execute")
            return f"Unexpected error: {str(e)}"
    return postgres_execute


def _create_schema_tool(client):
    """Create postgres_get_schema tool."""
    @tool
    async def postgres_get_schema(
        schema_name: str = "public",
        table_name: Optional[str] = None,
    ) -> str:
        """
        Get database schema information from PostgreSQL.

        Use this tool to explore the database structure before writing queries.
        You can get an overview of all tables or detailed info about a specific table.

        Args:
            schema_name: The database schema to inspect (default: "public").
            table_name: Optional specific table to get detailed info about.

        Returns:
            Schema information as formatted JSON.

        Example:
            postgres_get_schema()  # Get all tables in public schema
            postgres_get_schema("public", "users")  # Get detailed info about users table
        """
        try:
            import json

            if table_name:
                result = await client.get_table_info(table_name, schema_name)
            else:
                result = await client.get_schema(schema_name, include_columns=True)

            return json.dumps(result, indent=2, default=str)
        except TypeError as e:
            logger.error("Type error in postgres_get_schema: %s", e)
            return f"Type error: {e}"
        except Exception as e:
            # Import asyncpg types dynamically to check
            try:
                from seer.tools.postgres_client import _get_asyncpg
                asyncpg = _get_asyncpg()
                if isinstance(e, (asyncpg.PostgresError, asyncpg.InterfaceError)):
                    logger.error("Database error in postgres_get_schema: %s", e)
                    return f"Database error: {e}"
            except Exception:
                pass
            # Unexpected error - log for debugging
            logger.exception("Unexpected error in postgres_get_schema")
            return f"Unexpected error: {str(e)}"
    return postgres_get_schema


def _create_batch_tool(client):
    """Create postgres_execute_batch tool."""
    @tool
    async def postgres_execute_batch(
        statement: str,
        parameters_list: List[List[Any]],
    ) -> str:
        """
        Execute a SQL statement with multiple parameter sets (bulk insert/update).

        Use this tool for efficient batch operations when you need to insert
        or update many rows at once.

        Note: This operation may require human approval depending on configuration.
        If approval is required, the tool will pause and wait for confirmation.

        Args:
            statement: The SQL statement with $1, $2, etc. placeholders.
            parameters_list: List of parameter lists, one for each execution.

        Returns:
            Status message indicating success or failure.

        Example:
            postgres_execute_batch(
                "INSERT INTO users (name, email) VALUES ($1, $2)",
                [
                    ["Alice", "alice@example.com"],
                    ["Bob", "bob@example.com"],
                    ["Charlie", "charlie@example.com"]
                ]
            )
        """
        try:
            from seer.config import config
            if config.postgres_write_requires_approval:
                batch_size = len(parameters_list)
                sample_params = parameters_list[:3] if batch_size > 3 else parameters_list
                summary_note = f" (showing first 3 of {batch_size})" if batch_size > 3 else ""

                logger.info(
                    "Write approval required for batch statement: %s... ({batch_size} rows)",
                    statement[:100]
                )
                response = _request_write_approval(
                    f"{statement}\n\n-- Batch operation: {batch_size} rows{summary_note}",
                    sample_params
                )

                if not _is_write_approved(response):
                    logger.info("Batch write operation rejected by user: %s", response)
                    return (
                        f"Operation cancelled: User rejected the batch write operation. "
                        f"Response: {response}"
                    )

                logger.info("Batch write operation approved by user")

            args_list = [tuple(params) for params in parameters_list]
            await client.execute_many(statement, args_list)
            return f"Batch executed successfully: {len(args_list)} operations completed."

        except GraphInterrupt:
            raise
        except (ValueError, TypeError) as e:
            logger.error("Data error in postgres_execute_batch: %s", e)
            return f"Data error: {e}"
        except Exception as e:
            # Import asyncpg types dynamically to check
            try:
                from seer.tools.postgres_client import _get_asyncpg
                asyncpg = _get_asyncpg()
                if isinstance(e, (asyncpg.PostgresError, asyncpg.InterfaceError)):
                    logger.error("Database error in postgres_execute_batch: %s", e)
                    return f"Database error: {e}"
            except Exception:
                pass
            # Unexpected error - log for debugging
            logger.exception("Unexpected error in postgres_execute_batch")
            return f"Unexpected error: {str(e)}"
    return postgres_execute_batch


def _create_tools_for_client(client) -> List[BaseTool]:
    """Create LangChain tools bound to a specific PostgresClient instance."""
    return [
        _create_query_tool(client),
        _create_execute_tool(client),
        _create_schema_tool(client),
        _create_batch_tool(client),
    ]
