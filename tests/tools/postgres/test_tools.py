"""Tests for PostgreSQL tools."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.postgres.tools import (
    PostgresAnalyzeDbHealthTool,
    PostgresExecuteSqlTool,
    PostgresExplainQueryTool,
    PostgresGetObjectDetailsTool,
    PostgresListObjectsTool,
    PostgresListSchemasTool,
)


def make_mock_credentials(access_mode: str = "restricted"):
    """Create mock credentials with specified access mode."""
    mock_resource = MagicMock()
    mock_resource.resource_metadata = {"access_mode": access_mode}

    mock_credentials = MagicMock()
    mock_credentials.resource = mock_resource
    mock_credentials.secrets = {"postgres_connection_string": "postgresql://test:test@localhost:5432/testdb"}

    return mock_credentials


def make_mock_connection():
    """Create mock asyncpg connection."""
    mock_conn = AsyncMock()
    mock_conn.close = AsyncMock()
    return mock_conn


class TestPostgresListSchemasTool:
    """Tests for postgres_list_schemas tool."""

    @pytest.fixture
    def tool(self):
        return PostgresListSchemasTool()

    def test_metadata(self, tool):
        assert tool.name == "postgres_list_schemas"
        assert tool.integration_type == "postgres"
        assert tool.provider == "postgres"
        assert "postgres_connection_string" in tool.required_secrets

    @patch("seer.tools.postgres.base.asyncpg.connect")
    async def test_list_schemas_success(self, mock_connect, tool):
        mock_conn = make_mock_connection()
        mock_conn.fetch.return_value = [
            {"schema_name": "public", "owner": "postgres"},
            {"schema_name": "auth", "owner": "supabase"},
        ]
        mock_connect.return_value = mock_conn

        result = await tool.execute(
            access_token=None,
            arguments={"integration_resource_id": 1},
            credentials=make_mock_credentials(),
        )

        assert len(result) == 2
        assert result[0]["schema_name"] == "public"
        assert result[1]["schema_name"] == "auth"
        mock_conn.close.assert_called_once()


class TestPostgresListObjectsTool:
    """Tests for postgres_list_objects tool."""

    @pytest.fixture
    def tool(self):
        return PostgresListObjectsTool()

    def test_metadata(self, tool):
        assert tool.name == "postgres_list_objects"
        schema = tool.get_parameters_schema()
        assert "schema" in schema["properties"]
        assert "object_type" in schema["properties"]

    @patch("seer.tools.postgres.base.asyncpg.connect")
    async def test_list_tables_success(self, mock_connect, tool):
        mock_conn = make_mock_connection()
        mock_conn.fetch.return_value = [
            {"name": "users", "type": "table", "owner": "postgres"},
            {"name": "orders", "type": "table", "owner": "postgres"},
        ]
        mock_connect.return_value = mock_conn

        result = await tool.execute(
            access_token=None,
            arguments={"integration_resource_id": 1, "schema": "public", "object_type": "table"},
            credentials=make_mock_credentials(),
        )

        assert len(result) == 2
        assert result[0]["name"] == "users"
        mock_conn.close.assert_called_once()


class TestPostgresGetObjectDetailsTool:
    """Tests for postgres_get_object_details tool."""

    @pytest.fixture
    def tool(self):
        return PostgresGetObjectDetailsTool()

    def test_metadata(self, tool):
        assert tool.name == "postgres_get_object_details"
        schema = tool.get_parameters_schema()
        assert "table_name" in schema["required"]

    @patch("seer.tools.postgres.base.asyncpg.connect")
    async def test_get_table_details_success(self, mock_connect, tool):
        mock_conn = make_mock_connection()

        # Mock column query
        mock_conn.fetch.side_effect = [
            # Columns
            [
                {"column_name": "id", "data_type": "integer", "not_null": True, "default_value": None, "description": None},
                {"column_name": "name", "data_type": "text", "not_null": False, "default_value": None, "description": None},
            ],
            # Constraints
            [
                {"constraint_name": "users_pkey", "constraint_type": "PRIMARY KEY", "definition": "PRIMARY KEY (id)"},
            ],
            # Indexes
            [
                {"index_name": "users_pkey", "index_type": "btree", "definition": "CREATE UNIQUE INDEX...", "is_unique": True, "is_primary": True},
            ],
        ]
        mock_conn.fetchval.return_value = 1000  # row estimate

        mock_connect.return_value = mock_conn

        result = await tool.execute(
            access_token=None,
            arguments={"integration_resource_id": 1, "table_name": "users"},
            credentials=make_mock_credentials(),
        )

        assert len(result["columns"]) == 2
        assert result["columns"][0]["column_name"] == "id"
        assert len(result["constraints"]) == 1
        assert result["row_estimate"] == 1000


class TestPostgresExecuteSqlTool:
    """Tests for postgres_execute_sql tool with access mode enforcement."""

    @pytest.fixture
    def tool(self):
        return PostgresExecuteSqlTool()

    def test_metadata(self, tool):
        assert tool.name == "postgres_execute_sql"
        schema = tool.get_parameters_schema()
        assert "sql" in schema["required"]

    @patch("seer.tools.postgres.base.asyncpg.connect")
    async def test_select_allowed_in_restricted_mode(self, mock_connect, tool):
        mock_conn = make_mock_connection()
        mock_conn.fetch.return_value = [{"id": 1, "name": "Test"}]
        mock_connect.return_value = mock_conn

        result = await tool.execute(
            access_token=None,
            arguments={"integration_resource_id": 1, "sql": "SELECT * FROM users"},
            credentials=make_mock_credentials(access_mode="restricted"),
        )

        assert result["row_count"] == 1
        assert result["rows"][0]["name"] == "Test"

    async def test_insert_blocked_in_restricted_mode(self, tool):
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                access_token=None,
                arguments={"integration_resource_id": 1, "sql": "INSERT INTO users (name) VALUES ('test')"},
                credentials=make_mock_credentials(access_mode="restricted"),
            )

        assert exc_info.value.status_code == 403
        assert "restricted access mode" in exc_info.value.detail.lower()

    async def test_update_blocked_in_restricted_mode(self, tool):
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                access_token=None,
                arguments={"integration_resource_id": 1, "sql": "UPDATE users SET name = 'new'"},
                credentials=make_mock_credentials(access_mode="restricted"),
            )

        assert exc_info.value.status_code == 403

    async def test_delete_blocked_in_restricted_mode(self, tool):
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                access_token=None,
                arguments={"integration_resource_id": 1, "sql": "DELETE FROM users WHERE id = 1"},
                credentials=make_mock_credentials(access_mode="restricted"),
            )

        assert exc_info.value.status_code == 403

    async def test_drop_blocked_in_restricted_mode(self, tool):
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                access_token=None,
                arguments={"integration_resource_id": 1, "sql": "DROP TABLE users"},
                credentials=make_mock_credentials(access_mode="restricted"),
            )

        assert exc_info.value.status_code == 403

    @patch("seer.tools.postgres.base.asyncpg.connect")
    async def test_insert_allowed_in_unrestricted_mode(self, mock_connect, tool):
        mock_conn = make_mock_connection()
        mock_conn.fetch.return_value = [{"id": 1, "name": "test"}]
        mock_connect.return_value = mock_conn

        result = await tool.execute(
            access_token=None,
            arguments={"integration_resource_id": 1, "sql": "INSERT INTO users (name) VALUES ('test') RETURNING *"},
            credentials=make_mock_credentials(access_mode="unrestricted"),
        )

        assert result["row_count"] == 1

    @patch("seer.tools.postgres.base.asyncpg.connect")
    async def test_parameterized_query(self, mock_connect, tool):
        mock_conn = make_mock_connection()
        mock_conn.fetch.return_value = [{"id": 1, "name": "Test"}]
        mock_connect.return_value = mock_conn

        result = await tool.execute(
            access_token=None,
            arguments={
                "integration_resource_id": 1,
                "sql": "SELECT * FROM users WHERE id = $1",
                "params": [1],
            },
            credentials=make_mock_credentials(),
        )

        # Verify parameterized call
        mock_conn.fetch.assert_called_with("SELECT * FROM users WHERE id = $1", 1)


class TestPostgresExplainQueryTool:
    """Tests for postgres_explain_query tool."""

    @pytest.fixture
    def tool(self):
        return PostgresExplainQueryTool()

    def test_metadata(self, tool):
        assert tool.name == "postgres_explain_query"
        schema = tool.get_parameters_schema()
        assert "analyze" in schema["properties"]
        assert "format" in schema["properties"]

    @patch("seer.tools.postgres.base.asyncpg.connect")
    async def test_explain_without_analyze(self, mock_connect, tool):
        mock_conn = make_mock_connection()
        mock_conn.fetch.return_value = [
            ([{"Plan": {"Node Type": "Seq Scan", "Relation Name": "users"}}],)
        ]
        mock_connect.return_value = mock_conn

        result = await tool.execute(
            access_token=None,
            arguments={"integration_resource_id": 1, "sql": "SELECT * FROM users"},
            credentials=make_mock_credentials(),
        )

        assert "plan" in result


class TestPostgresAnalyzeDbHealthTool:
    """Tests for postgres_analyze_db_health tool."""

    @pytest.fixture
    def tool(self):
        return PostgresAnalyzeDbHealthTool()

    def test_metadata(self, tool):
        assert tool.name == "postgres_analyze_db_health"

    @patch("seer.tools.postgres.base.asyncpg.connect")
    async def test_health_check_success(self, mock_connect, tool):
        mock_conn = make_mock_connection()

        # Mock multiple fetchval/fetchrow calls
        mock_conn.fetchval.side_effect = [
            "100 MB",  # database size
            "1 day 02:30:00",  # uptime
        ]
        mock_conn.fetchrow.side_effect = [
            {"active_connections": 5, "max_connections": 100},  # connection stats
            {"cache_hit_ratio": 99.5},  # cache stats
            {"deadlocks": 0, "conflicts": 0, "temp_files": 10},  # db stats
        ]

        mock_connect.return_value = mock_conn

        result = await tool.execute(
            access_token=None,
            arguments={"integration_resource_id": 1},
            credentials=make_mock_credentials(),
        )

        assert result["database_size"] == "100 MB"
        assert result["active_connections"] == 5
        assert result["cache_hit_ratio"] == 99.5


class TestCredentialValidation:
    """Tests for credential validation across tools."""

    async def test_missing_credentials_raises_error(self):
        tool = PostgresListSchemasTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                access_token=None,
                arguments={"integration_resource_id": 1},
                credentials=None,
            )

        assert exc_info.value.status_code == 400

    async def test_missing_connection_string_raises_error(self):
        tool = PostgresListSchemasTool()

        mock_credentials = MagicMock()
        mock_credentials.resource = MagicMock()
        mock_credentials.secrets = {}  # No connection string

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(
                access_token=None,
                arguments={"integration_resource_id": 1},
                credentials=mock_credentials,
            )

        assert exc_info.value.status_code == 400
        assert "connection string" in exc_info.value.detail.lower()
