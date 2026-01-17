"""
PostgreSQL client with connection pooling and async operations.

This module contains the PostgresClient class which provides an async-native
interface to PostgreSQL databases with connection pooling support.
"""
# pylint: disable=import-outside-toplevel # Reason: Lazy loading for asyncpg (optional dependency)
# pylint: disable=global-statement # Reason: Singleton lazy-loading pattern requires global state
# pylint: disable=invalid-name # Reason: _asyncpg follows lazy-loading naming convention
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from seer.logger import get_logger

logger = get_logger("shared.tools.postgres_client")

# Lazy import for asyncpg to avoid import errors if not installed
_asyncpg = None


def _get_asyncpg():
    """Lazy load asyncpg to avoid import errors."""
    global _asyncpg
    if _asyncpg is None:
        try:
            import asyncpg
            _asyncpg = asyncpg
        except ImportError:
            # pylint: disable=raise-missing-from # Reason: Clear install instructions don't need traceback chain
            raise ImportError(
                "asyncpg is required for PostgreSQL operations. "
                "Install it with: pip install asyncpg"
            )
    return _asyncpg


class PostgresClient:
    """
    Async-native PostgreSQL client with connection pooling.

    Provides both a client interface and LangChain-compatible tools
    for database operations.

    Example:
        client = PostgresClient("postgresql://user:pass@localhost/db")
        await client.connect()

        # Direct usage
        results = await client.query("SELECT * FROM users WHERE id = $1", 1)

        # As LangChain tools
        tools = client.get_tools()
    """

    def __init__(
        self,
        connection_string: str,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
        command_timeout: float = 60.0,
    ):
        """
        Initialize the PostgreSQL client.

        Args:
            connection_string: PostgreSQL connection URI
                (e.g., postgresql://user:pass@host:port/database)
            min_pool_size: Minimum number of connections in the pool
            max_pool_size: Maximum number of connections in the pool
            command_timeout: Default timeout for queries in seconds
        """
        self._connection_string = connection_string
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._command_timeout = command_timeout
        self._pool: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialize the connection pool."""
        asyncpg = _get_asyncpg()
        async with self._lock:
            if self._pool is None:
                logger.info("Creating PostgreSQL connection pool")
                self._pool = await asyncpg.create_pool(
                    self._connection_string,
                    min_size=self._min_pool_size,
                    max_size=self._max_pool_size,
                    command_timeout=self._command_timeout,
                )
                logger.info("PostgreSQL connection pool created successfully")

    async def close(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            if self._pool is not None:
                await self._pool.close()
                self._pool = None
                logger.info("PostgreSQL connection pool closed")

    async def _ensure_connected(self) -> None:
        """Ensure the connection pool is initialized."""
        if self._pool is None:
            await self.connect()

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        await self._ensure_connected()
        async with self._pool.acquire() as conn:
            yield conn

    async def query(
        self,
        sql: str,
        *args,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as dictionaries.

        Args:
            sql: SQL query string with $1, $2, ... placeholders
            *args: Query parameters
            timeout: Query timeout in seconds (optional)

        Returns:
            List of dictionaries with column names as keys
        """
        await self._ensure_connected()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args, timeout=timeout)
            return [dict(row) for row in rows]

    async def query_one(
        self,
        sql: str,
        *args,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a SELECT query and return a single result.

        Args:
            sql: SQL query string with $1, $2, ... placeholders
            *args: Query parameters
            timeout: Query timeout in seconds (optional)

        Returns:
            Single dictionary or None if no results
        """
        await self._ensure_connected()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args, timeout=timeout)
            return dict(row) if row else None

    async def execute(
        self,
        sql: str,
        *args,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Execute an INSERT, UPDATE, DELETE, or DDL statement.

        Args:
            sql: SQL statement with $1, $2, ... placeholders
            *args: Statement parameters
            timeout: Query timeout in seconds (optional)

        Returns:
            Status message (e.g., "INSERT 0 1", "UPDATE 5")
        """
        await self._ensure_connected()
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *args, timeout=timeout)
            return result

    async def execute_many(
        self,
        sql: str,
        args_list: List[tuple],
        timeout: Optional[float] = None,
    ) -> None:
        """
        Execute a statement with multiple parameter sets (bulk insert/update).

        Args:
            sql: SQL statement with $1, $2, ... placeholders
            args_list: List of parameter tuples
            timeout: Query timeout in seconds (optional)
        """
        await self._ensure_connected()
        async with self._pool.acquire() as conn:
            await conn.executemany(sql, args_list, timeout=timeout)

    async def get_schema(
        self,
        schema_name: str = "public",
        include_columns: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get database schema information.

        Args:
            schema_name: Schema to inspect (default: "public")
            include_columns: Include column details for each table

        Returns:
            List of table information dictionaries
        """
        await self._ensure_connected()

        # Get tables
        tables_query = """
            SELECT
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema = $1
            ORDER BY table_name
        """

        tables = await self.query(tables_query, schema_name)

        if not include_columns:
            return tables

        # Get columns for each table
        columns_query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """

        for table in tables:
            columns = await self.query(columns_query, schema_name, table["table_name"])
            table["columns"] = columns

        return tables

    async def get_table_info(self, table_name: str, schema_name: str = "public") -> Dict[str, Any]:
        """
        Get detailed information about a specific table.

        Args:
            table_name: Name of the table
            schema_name: Schema containing the table

        Returns:
            Dictionary with table info, columns, indexes, and constraints
        """
        await self._ensure_connected()

        # Get columns
        columns_query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """
        columns = await self.query(columns_query, schema_name, table_name)

        # Get primary key
        pk_query = """
            SELECT a.attname as column_name
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = ($1 || '.' || $2)::regclass
            AND i.indisprimary
        """
        try:
            pk_columns = await self.query(pk_query, schema_name, table_name)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to fetch primary key for %s.%s: %s", schema_name, table_name, exc
            )
            pk_columns = []

        # Get foreign keys
        fk_query = """
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = $1
                AND tc.table_name = $2
        """
        foreign_keys = await self.query(fk_query, schema_name, table_name)

        # Get row count estimate
        count_query = """
            SELECT reltuples::bigint AS estimate
            FROM pg_class
            WHERE oid = ($1 || '.' || $2)::regclass
        """
        try:
            count_result = await self.query_one(count_query, schema_name, table_name)
            row_estimate = count_result["estimate"] if count_result else 0
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to fetch row estimate for %s.%s: %s", schema_name, table_name, exc
            )
            row_estimate = 0

        return {
            "table_name": table_name,
            "schema_name": schema_name,
            "columns": columns,
            "primary_key": [pk["column_name"] for pk in pk_columns],
            "foreign_keys": foreign_keys,
            "row_count_estimate": row_estimate,
        }

    def get_tools(self) -> List[BaseTool]:
        """
        Get LangChain-compatible tools for database operations.

        Returns:
            List of LangChain BaseTool instances
        """
        # Avoid circular import
        from seer.tools.postgres_tools import _create_tools_for_client

        return _create_tools_for_client(self)
