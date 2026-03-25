"""
PostgreSQL tools common utilities.

This module provides:
- AccessMode enum for read-only vs read-write access control
- SQL validation using pglast AST parsing to enforce restricted mode

The restricted mode blocks DDL, DML (except SELECT), and other write operations.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple
from urllib.parse import quote_plus

import pglast

from seer.tools.base import DefaultResourceRequirement, ResourcePickerConfig


class AccessMode(str, Enum):
    """Database access mode for credential binding."""
    RESTRICTED = "restricted"  # Read-only: SELECT, EXPLAIN, SHOW only
    UNRESTRICTED = "unrestricted"  # Full access: All SQL operations


# Statement types allowed in restricted mode (read-only operations)
RESTRICTED_MODE_ALLOWED_STATEMENTS = frozenset({
    "SelectStmt",
    "ExplainStmt",
    "VariableShowStmt",
})


def validate_sql_for_restricted_mode(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL against restricted mode rules using pglast AST parsing.

    In restricted mode, only SELECT, EXPLAIN, and SHOW statements are allowed.
    All DDL (CREATE, ALTER, DROP) and DML (INSERT, UPDATE, DELETE) are blocked.

    Args:
        sql: SQL statement(s) to validate

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if SQL is allowed
        - (False, reason) if SQL is blocked

    Examples:
        >>> validate_sql_for_restricted_mode("SELECT * FROM users")
        (True, None)
        >>> validate_sql_for_restricted_mode("INSERT INTO users VALUES (1)")
        (False, "INSERT statements are not allowed in restricted mode")
    """
    try:
        statements = pglast.parse_sql(sql)
    except pglast.parser.ParseError as e:  # pylint: disable=c-extension-no-member  # Reason: pglast.parser is a C extension;
        #ParseError exists at runtime but pylint cannot introspect it statically.
        return False, f"SQL parse error: {str(e)}"

    if not statements:
        return False, "Empty SQL statement"

    for raw_stmt in statements:
        # pglast returns RawStmt objects, get the actual statement
        stmt_node = raw_stmt.stmt if hasattr(raw_stmt, "stmt") else raw_stmt

        stmt_type = type(stmt_node).__name__

        # Check if statement type is in allowlist
        if stmt_type in RESTRICTED_MODE_ALLOWED_STATEMENTS:
            continue

        # Map statement types to human-readable names for error messages
        blocked_statements = {
            "InsertStmt": "INSERT",
            "UpdateStmt": "UPDATE",
            "DeleteStmt": "DELETE",
            "TruncateStmt": "TRUNCATE",
            "CreateStmt": "CREATE TABLE",
            "CreateTableAsStmt": "CREATE TABLE AS",
            "IndexStmt": "CREATE INDEX",
            "CreateFunctionStmt": "CREATE FUNCTION",
            "CreateSchemaStmt": "CREATE SCHEMA",
            "CreateSeqStmt": "CREATE SEQUENCE",
            "ViewStmt": "CREATE VIEW",
            "CreateTrigStmt": "CREATE TRIGGER",
            "CreateRoleStmt": "CREATE ROLE",
            "CreatedbStmt": "CREATE DATABASE",
            "CreatePolicyStmt": "CREATE POLICY",
            "CreateExtensionStmt": "CREATE EXTENSION",
            "AlterTableStmt": "ALTER TABLE",
            "AlterFunctionStmt": "ALTER FUNCTION",
            "AlterSeqStmt": "ALTER SEQUENCE",
            "AlterRoleStmt": "ALTER ROLE",
            "AlterDatabaseStmt": "ALTER DATABASE",
            "AlterObjectSchemaStmt": "ALTER SCHEMA",
            "AlterOwnerStmt": "ALTER OWNER",
            "DropStmt": "DROP",
            "DropRoleStmt": "DROP ROLE",
            "DropdbStmt": "DROP DATABASE",
            "GrantStmt": "GRANT",
            "GrantRoleStmt": "GRANT ROLE",
            "RevokeStmt": "REVOKE",
            "CopyStmt": "COPY",
            "CallStmt": "CALL",
            "DoStmt": "DO",
            "LockStmt": "LOCK",
            "VacuumStmt": "VACUUM",
            "ClusterStmt": "CLUSTER",
            "ReindexStmt": "REINDEX",
            "RefreshMatViewStmt": "REFRESH MATERIALIZED VIEW",
            "TransactionStmt": "Transaction control",
            "PrepareStmt": "PREPARE",
            "ExecuteStmt": "EXECUTE",
            "DeallocateStmt": "DEALLOCATE",
            "DeclareCursorStmt": "DECLARE CURSOR",
            "FetchStmt": "FETCH",
            "ClosePortalStmt": "CLOSE",
            "ListenStmt": "LISTEN",
            "NotifyStmt": "NOTIFY",
            "UnlistenStmt": "UNLISTEN",
            "LoadStmt": "LOAD",
            "CommentStmt": "COMMENT",
            "SecLabelStmt": "SECURITY LABEL",
        }

        if stmt_type in blocked_statements:
            return False, f"{blocked_statements[stmt_type]} statements are not allowed in restricted mode"

        # Catch-all for any other statement types not in allowlist
        return False, f"Statement type '{stmt_type}' is not allowed in restricted mode"

    return True, None


def build_connection_string(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    ssl_mode: str = "prefer",
) -> str:
    """
    Build a PostgreSQL connection string from individual components.

    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        ssl_mode: SSL mode (disable, allow, prefer, require, verify-ca, verify-full)

    Returns:
        PostgreSQL connection string (URI format)
    """
    # URL-encode password to handle special characters
    encoded_password = quote_plus(password)
    return f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}?sslmode={ssl_mode}"


# Resource picker configuration for Postgres database selection
POSTGRES_DATABASE_PICKER: ResourcePickerConfig = {
    "resource_type": "postgres_database",
    "display_field": "name",
    "value_field": "id",
    "search_enabled": True,
    "endpoint": "/integrations/postgres/resources/bindings",
}

POSTGRES_DEFAULT_RESOURCE: DefaultResourceRequirement = {
    "provider": "postgres",
    "resource_type": "database",
    "required": True,
}
