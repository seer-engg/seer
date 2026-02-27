"""
Unit tests for multi-account OAuth connection validation.

Tests the validation logic that ensures workflow tool nodes have valid OAuth
connections, handling both single-account (backward compatible) and multi-account
scenarios.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.compiler.validate_connections import (
    ConnectionValidationResult,
    validate_tool_connections,
    validate_connections_and_raise,
)
from seer.core.errors import ErrorCode, ValidationPhaseError
from seer.core.schema.models import ToolNode, WorkflowSpec

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock()
    user.user_id = "test_user_123"
    return user


@pytest.fixture
def mock_tool_with_oauth():
    """Create a mock tool that requires OAuth."""
    tool = MagicMock()
    tool.name = "google.gmail.send_email"
    tool.required_scopes = ["https://www.googleapis.com/auth/gmail.send"]
    tool.provider = "google"
    return tool


@pytest.fixture
def mock_tool_without_oauth():
    """Create a mock tool that doesn't require OAuth."""
    tool = MagicMock()
    tool.name = "test.no_oauth"
    tool.required_scopes = []
    tool.provider = None
    return tool


def make_oauth_connection(id_: int, provider: str, account_id: str):
    """Factory for creating mock OAuth connections."""
    conn = MagicMock()
    conn.id = id_
    conn.provider = provider
    conn.provider_account_id = account_id
    conn.status = "active"
    conn.scopes = "https://www.googleapis.com/auth/gmail.send"
    return conn


# =============================================================================
# Schema Tests: ToolNode with connection_id
# =============================================================================


def test_tool_node_accepts_connection_id():
    """Test that ToolNode schema accepts optional connection_id field."""
    node_dict = {
        "id": "send_email",
        "type": "tool",
        "tool": "google.gmail.send_email",
        "inputs": {"to": "test@example.com"},
        "connection_id": 123,
    }
    node = ToolNode(**node_dict)

    assert node.id == "send_email"
    assert node.tool == "google.gmail.send_email"
    assert node.connection_id == 123


def test_tool_node_connection_id_is_optional():
    """Test that ToolNode works without connection_id (backward compatible)."""
    node_dict = {
        "id": "send_email",
        "type": "tool",
        "tool": "google.gmail.send_email",
        "inputs": {"to": "test@example.com"},
    }
    node = ToolNode(**node_dict)

    assert node.connection_id is None


def test_tool_node_connection_id_allows_none():
    """Test that ToolNode accepts explicit None for connection_id."""
    node_dict = {
        "id": "send_email",
        "type": "tool",
        "tool": "google.gmail.send_email",
        "inputs": {},
        "connection_id": None,
    }
    node = ToolNode(**node_dict)

    assert node.connection_id is None


# =============================================================================
# Validation Tests: No Connections
# =============================================================================


@pytest.mark.anyio
async def test_validation_fails_when_no_connections(mock_user, mock_tool_with_oauth):
    """Test that validation fails when user has no OAuth connections for provider."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(
                id="send_email",
                type="tool",
                tool="google.gmail.send_email",
                inputs={},
            )
        ],
        edges=[],
        triggers=[],
    )

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_with_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=[])

            result = await validate_tool_connections(spec, mock_user)

    assert len(result.errors) == 1
    assert "No google account connected" in result.errors[0].message
    assert result.errors[0].node_id == "send_email"
    assert result.errors[0].code == ErrorCode.VALIDATION_ERROR


# =============================================================================
# Validation Tests: Single Account (Backward Compatible)
# =============================================================================


@pytest.mark.anyio
async def test_validation_auto_resolves_single_account(mock_user, mock_tool_with_oauth):
    """Test that single account is auto-resolved without connection_id."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(
                id="send_email",
                type="tool",
                tool="google.gmail.send_email",
                inputs={},
            )
        ],
        edges=[],
        triggers=[],
    )

    single_connection = make_oauth_connection(123, "google", "alice@gmail.com")

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_with_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=[single_connection])

            result = await validate_tool_connections(spec, mock_user)

    assert len(result.errors) == 0
    assert result.resolved_connections == {"send_email": 123}


@pytest.mark.anyio
async def test_validation_with_explicit_connection_id_single_account(mock_user, mock_tool_with_oauth):
    """Test that explicit connection_id works with single account."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(
                id="send_email",
                type="tool",
                tool="google.gmail.send_email",
                inputs={},
                connection_id=123,
            )
        ],
        edges=[],
        triggers=[],
    )

    single_connection = make_oauth_connection(123, "google", "alice@gmail.com")

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_with_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=[single_connection])

            result = await validate_tool_connections(spec, mock_user)

    assert len(result.errors) == 0
    # Not in resolved_connections because connection_id was explicit
    assert "send_email" not in result.resolved_connections


# =============================================================================
# Validation Tests: Multiple Accounts
# =============================================================================


@pytest.mark.anyio
async def test_validation_fails_multiple_accounts_no_selection(mock_user, mock_tool_with_oauth):
    """Test that multiple accounts without selection raises validation error."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(
                id="send_email",
                type="tool",
                tool="google.gmail.send_email",
                inputs={},
            )
        ],
        edges=[],
        triggers=[],
    )

    connections = [
        make_oauth_connection(123, "google", "alice@gmail.com"),
        make_oauth_connection(124, "google", "bob@gmail.com"),
    ]

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_with_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=connections)

            result = await validate_tool_connections(spec, mock_user)

    assert len(result.errors) == 1
    assert "Multiple google accounts connected" in result.errors[0].message
    assert "alice@gmail.com" in result.errors[0].message
    assert "bob@gmail.com" in result.errors[0].message
    assert result.errors[0].node_id == "send_email"


@pytest.mark.anyio
async def test_validation_passes_multiple_accounts_with_selection(mock_user, mock_tool_with_oauth):
    """Test that multiple accounts with explicit selection passes validation."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(
                id="send_email",
                type="tool",
                tool="google.gmail.send_email",
                inputs={},
                connection_id=124,  # User selected bob's account
            )
        ],
        edges=[],
        triggers=[],
    )

    connections = [
        make_oauth_connection(123, "google", "alice@gmail.com"),
        make_oauth_connection(124, "google", "bob@gmail.com"),
    ]

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_with_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=connections)

            result = await validate_tool_connections(spec, mock_user)

    assert len(result.errors) == 0


@pytest.mark.anyio
async def test_validation_fails_invalid_connection_id(mock_user, mock_tool_with_oauth):
    """Test that invalid connection_id raises validation error."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(
                id="send_email",
                type="tool",
                tool="google.gmail.send_email",
                inputs={},
                connection_id=999,  # Non-existent connection
            )
        ],
        edges=[],
        triggers=[],
    )

    connections = [
        make_oauth_connection(123, "google", "alice@gmail.com"),
    ]

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_with_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=connections)

            result = await validate_tool_connections(spec, mock_user)

    assert len(result.errors) == 1
    assert "not found or inactive" in result.errors[0].message
    assert "999" in result.errors[0].message


# =============================================================================
# Validation Tests: Non-OAuth Tools
# =============================================================================


@pytest.mark.anyio
async def test_validation_skips_non_oauth_tools(mock_user, mock_tool_without_oauth):
    """Test that tools without required_scopes are skipped."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(
                id="no_oauth_tool",
                type="tool",
                tool="test.no_oauth",
                inputs={},
            )
        ],
        edges=[],
        triggers=[],
    )

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_without_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=[])

            result = await validate_tool_connections(spec, mock_user)

    assert len(result.errors) == 0
    assert len(result.resolved_connections) == 0


# =============================================================================
# Validation Tests: Mixed Workflows
# =============================================================================


@pytest.mark.anyio
async def test_validation_handles_mixed_workflow(mock_user):
    """Test validation of workflow with both OAuth and non-OAuth tools."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(id="gmail", type="tool", tool="google.gmail.send_email", inputs={}),
            ToolNode(id="no_oauth", type="tool", tool="test.no_oauth", inputs={}),
        ],
        edges=[],
        triggers=[],
    )

    oauth_tool = MagicMock()
    oauth_tool.required_scopes = ["https://www.googleapis.com/auth/gmail.send"]
    oauth_tool.provider = "google"

    non_oauth_tool = MagicMock()
    non_oauth_tool.required_scopes = []
    non_oauth_tool.provider = None

    def tool_lookup(name):
        if name == "google.gmail.send_email":
            return oauth_tool
        return non_oauth_tool

    connection = make_oauth_connection(123, "google", "test@gmail.com")

    with patch("seer.core.compiler.validate_connections.get_tool", side_effect=tool_lookup):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=[connection])

            result = await validate_tool_connections(spec, mock_user)

    assert len(result.errors) == 0
    assert result.resolved_connections == {"gmail": 123}


# =============================================================================
# validate_connections_and_raise Tests
# =============================================================================


@pytest.mark.anyio
async def test_validate_and_raise_returns_resolved_on_success(mock_user, mock_tool_with_oauth):
    """Test that validate_connections_and_raise returns resolved connections."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(id="send_email", type="tool", tool="google.gmail.send_email", inputs={})
        ],
        edges=[],
        triggers=[],
    )

    connection = make_oauth_connection(123, "google", "test@gmail.com")

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_with_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=[connection])

            resolved = await validate_connections_and_raise(spec, mock_user)

    assert resolved == {"send_email": 123}


@pytest.mark.anyio
async def test_validate_and_raise_raises_on_errors(mock_user, mock_tool_with_oauth):
    """Test that validate_connections_and_raise raises ValidationPhaseError."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            ToolNode(id="send_email", type="tool", tool="google.gmail.send_email", inputs={})
        ],
        edges=[],
        triggers=[],
    )

    with patch("seer.core.compiler.validate_connections.get_tool", return_value=mock_tool_with_oauth):
        with patch("seer.database.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.filter.return_value.all = AsyncMock(return_value=[])

            with pytest.raises(ValidationPhaseError) as exc_info:
                await validate_connections_and_raise(spec, mock_user)

    assert "No google account connected" in str(exc_info.value)
    assert len(exc_info.value.errors) == 1
