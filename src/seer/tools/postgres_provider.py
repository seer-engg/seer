"""
PostgreSQL provider for resource provisioning and cleanup.

This module contains the PostgresProvider class for integration with eval_run
patterns for resource provisioning and cleanup.
"""
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from seer.logger import get_logger

logger = get_logger("shared.tools.postgres_provider")


class PostgresProvider:
    """
    Provider class for PostgreSQL that matches the BaseProvider interface
    used by eval_run for resource provisioning and cleanup.
    """

    def __init__(self, connection_string: Optional[str] = None):
        if connection_string is None:
            from seer.config import config
            connection_string = config.DATABASE_URL
        self._connection_string = connection_string
        self._client: Optional[Any] = None

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
        if not self._connection_string:
            logger.warning("No DATABASE_URL configured, skipping PostgreSQL provisioning")
            return {}

        from seer.tools.postgres_client import PostgresClient

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
                        logger.info("Created table %s", table_name)
                except (OSError, ValueError) as exc:
                    logger.exception("Failed to create table %s: %s", table_name, exc)

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
        if not self._client:
            if not self._connection_string:
                return
            from seer.tools.postgres_client import PostgresClient
            self._client = PostgresClient(self._connection_string)
            await self._client.connect()

        tables_created = resources.get("tables_created", [])
        for table_name in tables_created:
            try:
                await self._client.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
                logger.info("Dropped table %s", table_name)
            except (OSError, ValueError) as exc:
                logger.exception("Failed to drop table %s: %s", table_name, exc)

        # Close client
        if self._client:
            await self._client.close()
            self._client = None

    def get_tools(self) -> List[BaseTool]:
        """Get LangChain tools for this provider."""
        if self._client is None:
            from seer.tools.postgres_client import PostgresClient
            self._client = PostgresClient(self._connection_string)
        return self._client.get_tools()
