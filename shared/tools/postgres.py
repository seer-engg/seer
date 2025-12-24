"""
Async-native PostgreSQL connector with LangChain-compatible tools.

Provides dedicated tools for read and write operations to PostgreSQL databases,
with connection pooling and proper async handling.

Usage:
    from shared.tools.postgres import PostgresClient, get_postgres_tools
    
    # Get tools for an agent
    client = PostgresClient(connection_string="postgresql://...")
    tools = client.get_tools()
    
    # Or use standalone
    tools = get_postgres_tools(connection_string="postgresql://...")
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from langchain.tools import tool
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field
from shared.logger import get_logger

logger = get_logger("shared.tools.postgres")

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
        except Exception:
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
        except Exception:
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
        return _create_tools_for_client(self)


# ============================================================================
# LangChain Tools
# ============================================================================

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


def _create_tools_for_client(client: PostgresClient) -> List[BaseTool]:
    """Create LangChain tools bound to a specific PostgresClient instance."""
    
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
            # Validate query is read-only
            query_upper = query.strip().upper()
            if not query_upper.startswith("SELECT") and not query_upper.startswith("WITH"):
                return "Error: This tool only supports SELECT queries. Use postgres_execute for modifications."
            
            params = tuple(parameters) if parameters else ()
            results = await client.query(query, *params)
            
            if not results:
                return "Query returned no results."
            
            # Format results
            import json
            return json.dumps(results, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"PostgreSQL query error: {e}")
            return f"Query error: {str(e)}"
    
    @tool
    async def postgres_execute(
        statement: str,
        parameters: Optional[List[Any]] = None,
    ) -> str:
        """
        Execute a write SQL statement (INSERT, UPDATE, DELETE, CREATE, ALTER, DROP) on the PostgreSQL database.
        
        Use this tool when you need to modify data or schema in the database.
        
        Note: This operation may require human approval depending on configuration.
        If approval is required, the tool will pause and wait for confirmation.
        
        Args:
            statement: The SQL statement to execute. Use $1, $2, etc. for parameters.
            parameters: Optional list of statement parameters in order.
        
        Returns:
            Status message indicating the result (e.g., "INSERT 0 1", "UPDATE 5").
        
        Example:
            postgres_execute("INSERT INTO users (name, email) VALUES ($1, $2)", ["John", "john@example.com"])
            postgres_execute("UPDATE products SET price = $1 WHERE id = $2", [29.99, 123])
            postgres_execute("DELETE FROM sessions WHERE expires_at < NOW()")
        """
        try:
            # Check if write operations require approval
            from shared.config import config
            if config.postgres_write_requires_approval:
                logger.info(f"Write approval required for statement: {statement[:100]}...")
                response = _request_write_approval(statement, parameters)
                
                if not _is_write_approved(response):
                    logger.info(f"Write operation rejected by user: {response}")
                    return f"Operation cancelled: User rejected the write operation. Response: {response}"
                
                logger.info("Write operation approved by user")
            
            params = tuple(parameters) if parameters else ()
            result = await client.execute(statement, *params)
            return f"Statement executed successfully: {result}"
            
        except GraphInterrupt:
            # Re-raise GraphInterrupt for LangGraph human-in-the-loop handling
            raise
        except Exception as e:
            logger.error(f"PostgreSQL execute error: {e}")
            return f"Execution error: {str(e)}"
    
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
            
        except Exception as e:
            logger.error(f"PostgreSQL schema error: {e}")
            return f"Schema retrieval error: {str(e)}"
    
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
            # Check if write operations require approval
            from shared.config import config
            if config.postgres_write_requires_approval:
                # For batch operations, show a summary
                batch_size = len(parameters_list)
                sample_params = parameters_list[:3] if batch_size > 3 else parameters_list
                summary_note = f" (showing first 3 of {batch_size})" if batch_size > 3 else ""
                
                logger.info(f"Write approval required for batch statement: {statement[:100]}... ({batch_size} rows)")
                response = _request_write_approval(
                    f"{statement}\n\n-- Batch operation: {batch_size} rows{summary_note}",
                    sample_params
                )
                
                if not _is_write_approved(response):
                    logger.info(f"Batch write operation rejected by user: {response}")
                    return f"Operation cancelled: User rejected the batch write operation. Response: {response}"
                
                logger.info("Batch write operation approved by user")
            
            args_list = [tuple(params) for params in parameters_list]
            await client.execute_many(statement, args_list)
            return f"Batch executed successfully: {len(args_list)} operations completed."
            
        except GraphInterrupt:
            # Re-raise GraphInterrupt for LangGraph human-in-the-loop handling
            raise
        except Exception as e:
            logger.error(f"PostgreSQL batch execute error: {e}")
            return f"Batch execution error: {str(e)}"
    
    return [postgres_query, postgres_execute, postgres_get_schema, postgres_execute_batch]


# ============================================================================
# Convenience Functions
# ============================================================================

# Global client instance for simple usage
_default_client: Optional[PostgresClient] = None
_default_client_lock = asyncio.Lock()


async def get_default_client(connection_string: Optional[str] = None) -> PostgresClient:
    """
    Get or create a default PostgresClient instance.
    
    Args:
        connection_string: PostgreSQL connection URI. If not provided,
            uses the DATABASE_URL from shared.config.
    
    Returns:
        PostgresClient instance
    """
    global _default_client
    
    async with _default_client_lock:
        if _default_client is None:
            if connection_string is None:
                from shared.config import config
                connection_string = config.DATABASE_URL
                
            if not connection_string:
                raise ValueError(
                    "No PostgreSQL connection string provided. "
                    "Set DATABASE_URL environment variable or pass connection_string."
                )
            
            _default_client = PostgresClient(connection_string)
            await _default_client.connect()
        
        return _default_client


async def close_default_client() -> None:
    """Close the default PostgresClient if it exists."""
    global _default_client
    
    async with _default_client_lock:
        if _default_client is not None:
            await _default_client.close()
            _default_client = None


def get_postgres_tools(
    connection_string: Optional[str] = None,
    client: Optional[PostgresClient] = None,
) -> List[BaseTool]:
    """
    Get PostgreSQL LangChain tools.
    
    Convenience function to get PostgreSQL tools either from a provided client
    or by creating a new one with the given connection string.
    
    Args:
        connection_string: PostgreSQL connection URI
        client: Existing PostgresClient instance (takes precedence)
    
    Returns:
        List of LangChain BaseTool instances
    
    Example:
        # With connection string
        tools = get_postgres_tools("postgresql://user:pass@localhost/db")
        
        # With existing client
        client = PostgresClient("postgresql://...")
        tools = get_postgres_tools(client=client)
    """
    if client is not None:
        return client.get_tools()
    
    if connection_string is None:
        from shared.config import config
        connection_string = config.DATABASE_URL
    
    if not connection_string:
        raise ValueError(
            "No PostgreSQL connection string provided. "
            "Set DATABASE_URL environment variable or pass connection_string."
        )
    
    new_client = PostgresClient(connection_string)
    return new_client.get_tools()


# ============================================================================
# PostgreSQL Provider (for integration with eval_run pattern)
# ============================================================================

class PostgresProvider:
    """
    Provider class for PostgreSQL that matches the BaseProvider interface
    used by eval_run for resource provisioning and cleanup.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        if connection_string is None:
            from shared.config import config
            connection_string = config.DATABASE_URL
        self._connection_string = connection_string
        self._client: Optional[PostgresClient] = None
    
    @property
    def persistent_resource(self) -> Dict[str, Any]:
        """Return persistent resource metadata."""
        return {
            "type": "postgres",
            "connection_string": self._connection_string,
        }
    
    async def provision_resources(
        self,
        seed: str,
        user_id: str,
        tables: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Provision PostgreSQL resources (create tables, seed data).
        
        Args:
            seed: Unique seed for resource naming
            user_id: User ID for isolation
            tables: Optional list of table definitions to create
        
        Returns:
            Dictionary of provisioned resource metadata
        """
        return {}
        if not self._connection_string:
            logger.warning("No DATABASE_URL configured, skipping PostgreSQL provisioning")
            return {}
        
        self._client = PostgresClient(self._connection_string)
        await self._client.connect()
        
        resources = {
            "seed": seed,
            "user_id": user_id,
            "tables_created": [],
        }
        
        # Create tables if provided
        if tables:
            for table_def in tables:
                table_name = f"{table_def['name']}_{seed}"
                try:
                    # Create table with seed suffix for isolation
                    create_sql = table_def.get("create_sql", "").replace(
                        table_def["name"], table_name
                    )
                    if create_sql:
                        await self._client.execute(create_sql)
                        resources["tables_created"].append(table_name)
                        logger.info(f"Created table {table_name}")
                except Exception as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
        
        return resources
    
    async def cleanup_resources(
        self,
        resources: Dict[str, Any],
        user_id: str,
    ) -> None:
        """
        Cleanup PostgreSQL resources (drop tables created during provisioning).
        
        Args:
            resources: Resources metadata from provision_resources
            user_id: User ID for verification
        """
        return {}
        if not self._client:
            if not self._connection_string:
                return
            self._client = PostgresClient(self._connection_string)
            await self._client.connect()
        
        tables_created = resources.get("tables_created", [])
        for table_name in tables_created:
            try:
                await self._client.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
                logger.info(f"Dropped table {table_name}")
            except Exception as e:
                logger.error(f"Failed to drop table {table_name}: {e}")
        
        # Close client
        if self._client:
            await self._client.close()
            self._client = None
    
    def get_tools(self) -> List[BaseTool]:
        """Get LangChain tools for this provider."""
        if self._client is None:
            self._client = PostgresClient(self._connection_string)
        return self._client.get_tools()

