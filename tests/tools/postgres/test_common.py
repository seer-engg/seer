"""Tests for PostgreSQL common utilities (SQL validation)."""
import pytest

from seer.tools.postgres.common import (
    AccessMode,
    build_connection_string,
    validate_sql_for_restricted_mode,
)


class TestAccessMode:
    """Tests for AccessMode enum."""

    def test_restricted_value(self):
        assert AccessMode.RESTRICTED.value == "restricted"

    def test_unrestricted_value(self):
        assert AccessMode.UNRESTRICTED.value == "unrestricted"

    def test_from_string(self):
        assert AccessMode("restricted") == AccessMode.RESTRICTED
        assert AccessMode("unrestricted") == AccessMode.UNRESTRICTED


class TestValidateSqlForRestrictedMode:
    """Tests for SQL validation in restricted mode."""

    # === ALLOWED STATEMENTS ===

    def test_simple_select_allowed(self):
        is_valid, error = validate_sql_for_restricted_mode("SELECT * FROM users")
        assert is_valid is True
        assert error is None

    def test_select_with_where_allowed(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "SELECT id, name FROM users WHERE status = 'active'"
        )
        assert is_valid is True
        assert error is None

    def test_select_with_join_allowed(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        )
        assert is_valid is True
        assert error is None

    def test_select_with_subquery_allowed(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)"
        )
        assert is_valid is True
        assert error is None

    def test_select_with_cte_allowed(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "WITH active_users AS (SELECT * FROM users WHERE active) SELECT * FROM active_users"
        )
        assert is_valid is True
        assert error is None

    def test_explain_allowed(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "EXPLAIN SELECT * FROM users"
        )
        assert is_valid is True
        assert error is None

    def test_explain_analyze_allowed(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "EXPLAIN ANALYZE SELECT * FROM users"
        )
        assert is_valid is True
        assert error is None

    def test_show_allowed(self):
        is_valid, error = validate_sql_for_restricted_mode("SHOW search_path")
        assert is_valid is True
        assert error is None

    # === BLOCKED DML STATEMENTS ===

    def test_insert_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "INSERT INTO users (name) VALUES ('test')"
        )
        assert is_valid is False
        assert "INSERT" in error

    def test_update_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "UPDATE users SET name = 'new' WHERE id = 1"
        )
        assert is_valid is False
        assert "UPDATE" in error

    def test_delete_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "DELETE FROM users WHERE id = 1"
        )
        assert is_valid is False
        assert "DELETE" in error

    def test_truncate_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode("TRUNCATE users")
        assert is_valid is False
        assert "TRUNCATE" in error

    # === BLOCKED DDL STATEMENTS ===

    def test_create_table_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "CREATE TABLE test (id int)"
        )
        assert is_valid is False
        assert "CREATE TABLE" in error

    def test_create_index_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "CREATE INDEX idx ON users (name)"
        )
        assert is_valid is False
        assert "CREATE INDEX" in error

    def test_create_function_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "CREATE FUNCTION test() RETURNS void AS $$ BEGIN END; $$ LANGUAGE plpgsql"
        )
        assert is_valid is False
        assert "CREATE FUNCTION" in error

    def test_alter_table_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "ALTER TABLE users ADD COLUMN email text"
        )
        assert is_valid is False
        assert "ALTER TABLE" in error

    def test_drop_table_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode("DROP TABLE users")
        assert is_valid is False
        assert "DROP" in error

    def test_drop_database_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode("DROP DATABASE test")
        assert is_valid is False
        assert "DROP DATABASE" in error

    # === BLOCKED PERMISSION STATEMENTS ===

    def test_grant_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "GRANT SELECT ON users TO reader"
        )
        assert is_valid is False
        assert "GRANT" in error

    def test_revoke_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "REVOKE SELECT ON users FROM reader"
        )
        assert is_valid is False
        # REVOKE is handled by GrantStmt in pglast, so error might say GRANT
        assert "GRANT" in error or "REVOKE" in error

    def test_create_role_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode("CREATE ROLE reader")
        assert is_valid is False
        assert "CREATE ROLE" in error

    # === BLOCKED DANGEROUS STATEMENTS ===

    def test_copy_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "COPY users TO '/tmp/users.csv'"
        )
        assert is_valid is False
        assert "COPY" in error

    def test_do_block_blocked(self):
        is_valid, error = validate_sql_for_restricted_mode(
            "DO $$ BEGIN RAISE NOTICE 'test'; END $$"
        )
        assert is_valid is False
        assert "DO" in error

    # === ERROR HANDLING ===

    def test_invalid_sql_returns_parse_error(self):
        is_valid, error = validate_sql_for_restricted_mode("SELECT * FORM users")
        assert is_valid is False
        assert "parse error" in error.lower()

    def test_empty_sql_returns_error(self):
        is_valid, error = validate_sql_for_restricted_mode("")
        assert is_valid is False
        assert "empty" in error.lower() or "parse error" in error.lower()


class TestBuildConnectionString:
    """Tests for connection string builder."""

    def test_basic_connection_string(self):
        result = build_connection_string(
            host="localhost",
            port=5432,
            database="mydb",
            user="myuser",
            password="mypass",
            ssl_mode="prefer",
        )
        assert result == "postgresql://myuser:mypass@localhost:5432/mydb?sslmode=prefer"

    def test_connection_string_with_special_password(self):
        result = build_connection_string(
            host="localhost",
            port=5432,
            database="mydb",
            user="myuser",
            password="p@ss:word/123",
            ssl_mode="require",
        )
        # Password should be URL-encoded
        assert "p%40ss%3Aword%2F123" in result
        assert "sslmode=require" in result

    def test_connection_string_custom_port(self):
        result = build_connection_string(
            host="db.example.com",
            port=5433,
            database="production",
            user="admin",
            password="secret",
            ssl_mode="verify-full",
        )
        assert "db.example.com:5433" in result
        assert "sslmode=verify-full" in result
