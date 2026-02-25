"""
Integration tests for tool execution with credentials.

Tests:
- Tool executor with OAuth credentials
- Credential resolver functionality
- Integration with database models
- Error handling for missing credentials
- Scope validation
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.database import IntegrationResource, OAuthConnection
from seer.tools.base import BaseTool
from seer.tools.credential_resolver import CredentialResolver, ResolvedCredentials
from seer.tools.executor import execute_tool


# =============================================================================
# Mock Tool for Testing
# =============================================================================


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"
    required_scopes = ["test.scope"]

    def get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
            },
        }

    async def execute(self, access_token: str, arguments: dict, *, credentials=None, context=None):
        return {
            "success": True,
            "access_token": access_token,
            "arguments": arguments,
            "credentials_provided": credentials is not None,
        }


class MockToolNoScopes(BaseTool):
    """Mock tool that doesn't require OAuth."""

    name = "mock_tool_no_scopes"
    description = "A mock tool without OAuth"
    required_scopes = []

    def get_parameters_schema(self):
        return {"type": "object", "properties": {}}

    async def execute(self, access_token: str, arguments: dict, *, credentials=None, context=None):
        return {"success": True, "no_auth": True}


# =============================================================================
# CredentialResolver Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_credential_resolver_no_scopes(db_engine, test_user):
    """Test credential resolver for tool without required scopes."""
    tool = MockToolNoScopes()
    resolver = CredentialResolver(user=test_user, tool=tool)

    resolved = await resolver.resolve({})

    assert resolved.connection is None
    assert resolved.access_token is None
    assert resolved.resource is None
    assert resolved.secrets == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_credential_resolver_with_oauth(db_engine, test_user):
    """Test credential resolver with OAuth connection."""
    # Create OAuth connection
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="google_user_001",
        access_token_enc="test_token_123",
        refresh_token_enc="refresh_token_123",
        expires_at=expires_at,
        scopes="test.scope test.read",
    )

    tool = MockTool()

    # Mock get_oauth_token to return our connection
    with patch("seer.tools.credential_resolver.get_oauth_token") as mock_oauth:
        mock_oauth.return_value = (connection, "test_token_123")

        resolver = CredentialResolver(
            user=test_user,
            tool=tool,
            connection_id=str(connection.id),
        )

        resolved = await resolver.resolve({})

        assert resolved.connection is not None
        assert resolved.connection.id == connection.id
        assert resolved.access_token == "test_token_123"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_credential_resolver_missing_scopes(db_engine, test_user):
    """Test credential resolver fails when connection missing required scopes."""
    # Create OAuth connection with insufficient scopes
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="google_user_002",
        access_token_enc="test_token_123",
        refresh_token_enc="refresh_token_123",
        expires_at=expires_at,
        scopes="other.scope",  # Missing "test.scope"
    )

    tool = MockTool()

    with patch("seer.tools.credential_resolver.get_oauth_token") as mock_oauth:
        mock_oauth.return_value = (connection, "test_token_123")

        resolver = CredentialResolver(
            user=test_user,
            tool=tool,
            connection_id=str(connection.id),
        )

        # Should raise exception for missing scope
        with pytest.raises(HTTPException) as exc_info:
            await resolver.resolve({})

        assert exc_info.value.status_code == 403
        assert "missing required scope" in str(exc_info.value.detail).lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_credential_resolver_with_resource(db_engine, test_user):
    """Test credential resolver with integration resource."""
    # Create integration resource
    resource = await IntegrationResource.create(
        user=test_user,
        resource_id="res_test_123",
        provider="gmail",
        resource_type="account",
        resource_key="test@gmail.com",
        resource_metadata={"name": "Test Gmail"},
    )

    tool = MockToolNoScopes()
    resolver = CredentialResolver(user=test_user, tool=tool)

    # Pass resource_id in arguments
    with patch.object(resolver, "_resolve_resource") as mock_resolve_resource:
        mock_resolve_resource.return_value = resource

        resolved = await resolver.resolve({"resource_id": "res_test_123"})

        assert resolved.resource is not None
        assert resolved.resource.id == resource.id


# =============================================================================
# Tool Executor Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_tool_no_auth(db_engine, test_user):
    """Test executing tool that doesn't require authentication."""
    with patch("seer.tools.executor.get_tool") as mock_get_tool:
        mock_get_tool.return_value = MockToolNoScopes()

        result = await execute_tool(
            tool_name="mock_tool_no_scopes",
            user=test_user,
        )

        assert result["success"] is True
        assert result["no_auth"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_tool_with_oauth(db_engine, test_user):
    """Test executing tool with OAuth credentials."""
    # Create OAuth connection
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="google_user_003",
        access_token_enc="test_token_123",
        refresh_token_enc="refresh_token_123",
        expires_at=expires_at,
        scopes="test.scope",
    )

    with patch("seer.tools.executor.get_tool") as mock_get_tool, \
         patch("seer.tools.credential_resolver.get_oauth_token") as mock_oauth:

        mock_get_tool.return_value = MockTool()
        mock_oauth.return_value = (connection, "test_token_123")

        result = await execute_tool(
            tool_name="mock_tool",
            user=test_user,
            connection_id=str(connection.id),
            arguments={"param1": "value1"},
        )

        assert result["success"] is True
        assert result["access_token"] == "test_token_123"
        assert result["arguments"]["param1"] == "value1"
        assert result["credentials_provided"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_tool_not_found(db_engine, test_user):
    """Test executing non-existent tool raises error."""
    with patch("seer.tools.executor.get_tool") as mock_get_tool:
        mock_get_tool.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await execute_tool(
                tool_name="nonexistent_tool",
                user=test_user,
            )

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_tool_execution_failure(db_engine, test_user):
    """Test tool execution failure handling."""

    class FailingTool(BaseTool):
        name = "failing_tool"
        description = "A tool that fails"
        required_scopes = []

        def get_parameters_schema(self):
            return {"type": "object"}

        async def execute(self, access_token: str, arguments: dict, *, credentials=None, context=None):
            raise ValueError("Tool execution failed")

    with patch("seer.tools.executor.get_tool") as mock_get_tool:
        mock_get_tool.return_value = FailingTool()

        with pytest.raises(HTTPException) as exc_info:
            await execute_tool(
                tool_name="failing_tool",
                user=test_user,
            )

        assert exc_info.value.status_code == 500
        assert "execution failed" in str(exc_info.value.detail).lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_tool_credential_resolution_failure(db_engine, test_user):
    """Test credential resolution failure handling."""
    # Tool requires OAuth but no connection provided
    with patch("seer.tools.executor.get_tool") as mock_get_tool:
        mock_get_tool.return_value = MockTool()

        with pytest.raises(HTTPException):
            await execute_tool(
                tool_name="mock_tool",
                user=test_user,
                # No connection_id provided!
            )


# =============================================================================
# Integration Resource Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_integration_resource(db_engine, test_user):
    """Test creating integration resource."""
    resource = await IntegrationResource.create(
        user=test_user,
        resource_id="res_gmail_001",
        provider="gmail",
        resource_type="account",
        resource_key="user@gmail.com",
        resource_metadata={"name": "Personal Gmail", "email": "user@gmail.com"},
    )

    assert resource.id is not None
    assert resource.resource_id == "res_gmail_001"
    assert resource.provider == "gmail"
    assert resource.resource_type == "account"
    assert resource.resource_key == "user@gmail.com"
    assert resource.resource_metadata["name"] == "Personal Gmail"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_integration_resources_by_provider(db_engine, test_user):
    """Test querying integration resources by provider."""
    await IntegrationResource.create(
        user=test_user,
        resource_id="res_1",
        provider="gmail",
        resource_type="account",
        resource_key="user1@gmail.com",
    )
    await IntegrationResource.create(
        user=test_user,
        resource_id="res_2",
        provider="github",
        resource_type="account",
        resource_key="user",
    )
    await IntegrationResource.create(
        user=test_user,
        resource_id="res_3",
        provider="gmail",
        resource_type="account",
        resource_key="user2@gmail.com",
    )

    gmail_resources = await IntegrationResource.filter(
        user=test_user, provider="gmail"
    ).all()

    assert len(gmail_resources) == 2
    assert {r.resource_id for r in gmail_resources} == {"res_1", "res_3"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_resource_user_relationship(db_engine, test_user):
    """Test relationship between IntegrationResource and User."""
    resource = await IntegrationResource.create(
        user=test_user,
        resource_id="res_rel_test",
        provider="gmail",
        resource_type="account",
        resource_key="test@gmail.com",
    )

    # Access user from resource
    resource_user = await resource.user
    assert resource_user.id == test_user.id

    # Access resources from user (if reverse relationship exists)
    # Note: May need to check if this is defined in the model


# =============================================================================
# Real Tool Execution Tests (Gap 1: Executor → Tool.execute())
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_tool_calls_real_tool_with_correct_signature(db_engine, test_user):
    """
    Verify executor calls tool.execute() with credentials and context kwargs.

    This test addresses Gap 1 from the RCA: ensuring the executor actually
    passes kwargs correctly to real tool instances, not just mocked ones.
    """
    from seer.tools.base import register_tool, clear_registry

    # Track what arguments the tool received
    received_args = {}

    class SignatureVerifyingTool(BaseTool):
        name = "signature_verifying_tool"
        description = "Tool that records what arguments it receives"
        required_scopes = []  # No OAuth required

        def get_parameters_schema(self):
            return {
                "type": "object",
                "properties": {"input": {"type": "string"}},
            }

        async def execute(self, access_token, arguments, *, credentials=None, context=None):
            received_args["access_token"] = access_token
            received_args["arguments"] = arguments
            received_args["credentials"] = credentials
            received_args["context"] = context
            received_args["credentials_is_kwarg"] = True  # If we got here, it was passed correctly
            return {"success": True, "received_input": arguments.get("input")}

    clear_registry()
    register_tool(SignatureVerifyingTool())

    try:
        with patch("seer.tools.executor.get_tool") as mock_get_tool:
            mock_get_tool.return_value = SignatureVerifyingTool()

            result = await execute_tool(
                tool_name="signature_verifying_tool",
                user=test_user,
                arguments={"input": "test_value"},
            )

        assert result["success"] is True
        assert result["received_input"] == "test_value"
        assert received_args["arguments"]["input"] == "test_value"
        assert received_args["credentials_is_kwarg"] is True
        assert "credentials" in received_args  # Was passed as kwarg
        assert received_args["credentials"] is not None  # ResolvedCredentials object
    finally:
        clear_registry()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_tool_passes_context_to_tool(db_engine, test_user):
    """
    Verify executor passes WorkflowRuntimeContext to tool.execute().

    The context parameter allows tools to access workflow state and services
    during execution.
    """
    from seer.tools.base import register_tool, clear_registry
    from unittest.mock import MagicMock

    received_context = {}

    class ContextAwareTool(BaseTool):
        name = "context_aware_tool"
        description = "Tool that uses runtime context"
        required_scopes = []

        def get_parameters_schema(self):
            return {"type": "object", "properties": {}}

        async def execute(self, access_token, arguments, *, credentials=None, context=None):
            received_context["context"] = context
            return {"context_received": context is not None}

    clear_registry()
    register_tool(ContextAwareTool())

    # Create a mock context
    mock_context = MagicMock()
    mock_context.workflow_id = "wf_test_123"

    try:
        with patch("seer.tools.executor.get_tool") as mock_get_tool:
            mock_get_tool.return_value = ContextAwareTool()

            result = await execute_tool(
                tool_name="context_aware_tool",
                user=test_user,
                arguments={},
                context=mock_context,
            )

        assert result["context_received"] is True
        assert received_context["context"] is mock_context
        assert received_context["context"].workflow_id == "wf_test_123"
    finally:
        clear_registry()


# =============================================================================
# Tool Error Handling Tests (Gap 3)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_http_exception_preserved(db_engine, test_user):
    """
    Verify HTTPException from tool is passed through with correct status.

    Tools may raise HTTPException for business logic errors (e.g., 400 for
    invalid input, 403 for permission denied). These should propagate unchanged.
    """
    from seer.tools.base import register_tool, clear_registry

    class BadRequestTool(BaseTool):
        name = "bad_request_tool"
        description = "Tool that raises 400"
        required_scopes = []

        def get_parameters_schema(self):
            return {"type": "object"}

        async def execute(self, access_token, arguments, *, credentials=None, context=None):
            raise HTTPException(status_code=400, detail="Invalid input format")

    clear_registry()
    register_tool(BadRequestTool())

    try:
        with patch("seer.tools.executor.get_tool") as mock_get_tool:
            mock_get_tool.return_value = BadRequestTool()

            with pytest.raises(HTTPException) as exc_info:
                await execute_tool(tool_name="bad_request_tool", user=test_user)

            assert exc_info.value.status_code == 400
            assert "Invalid input format" in str(exc_info.value.detail)
    finally:
        clear_registry()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_unexpected_exception_returns_500(db_engine, test_user):
    """
    Verify unexpected exceptions are wrapped in 500 HTTPException.

    Non-HTTPException errors should be caught and converted to 500 status
    with the error message included in the detail.
    """
    from seer.tools.base import register_tool, clear_registry

    class UnexpectedErrorTool(BaseTool):
        name = "unexpected_error_tool"
        description = "Tool that raises unexpected error"
        required_scopes = []

        def get_parameters_schema(self):
            return {"type": "object"}

        async def execute(self, access_token, arguments, *, credentials=None, context=None):
            raise RuntimeError("Database connection lost")

    clear_registry()
    register_tool(UnexpectedErrorTool())

    try:
        with patch("seer.tools.executor.get_tool") as mock_get_tool:
            mock_get_tool.return_value = UnexpectedErrorTool()

            with pytest.raises(HTTPException) as exc_info:
                await execute_tool(tool_name="unexpected_error_tool", user=test_user)

            assert exc_info.value.status_code == 500
            assert "execution failed" in str(exc_info.value.detail).lower()
            assert "Database connection lost" in str(exc_info.value.detail)
    finally:
        clear_registry()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_returns_complex_output(db_engine, test_user):
    """
    Verify tool can return complex nested output structures.

    Tools often return nested dictionaries with arrays and multiple levels.
    The executor should pass these through without modification.
    """
    from seer.tools.base import register_tool, clear_registry

    class ComplexOutputTool(BaseTool):
        name = "complex_output_tool"
        description = "Tool that returns complex structure"
        required_scopes = []

        def get_parameters_schema(self):
            return {"type": "object", "properties": {"count": {"type": "integer"}}}

        def get_output_schema(self):
            return {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "object"}},
                    "metadata": {"type": "object"},
                },
            }

        async def execute(self, access_token, arguments, *, credentials=None, context=None):
            count = arguments.get("count", 3)
            return {
                "items": [{"id": i, "name": f"item_{i}"} for i in range(count)],
                "metadata": {
                    "total": count,
                    "page": 1,
                    "has_more": False,
                },
            }

    clear_registry()
    register_tool(ComplexOutputTool())

    try:
        with patch("seer.tools.executor.get_tool") as mock_get_tool:
            mock_get_tool.return_value = ComplexOutputTool()

            result = await execute_tool(
                tool_name="complex_output_tool",
                user=test_user,
                arguments={"count": 5},
            )

        assert len(result["items"]) == 5
        assert result["items"][0]["id"] == 0
        assert result["items"][4]["name"] == "item_4"
        assert result["metadata"]["total"] == 5
        assert result["metadata"]["has_more"] is False
    finally:
        clear_registry()
