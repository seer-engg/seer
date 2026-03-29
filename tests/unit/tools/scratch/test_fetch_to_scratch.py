"""Unit tests for fetch_to_scratch tool."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.scratch.models import ScratchDataRef
from seer.tools.scratch.fetch_to_scratch import FetchToScratchTool


@pytest.mark.asyncio
@pytest.mark.unit
class TestFetchToScratchTool:
    """Tests for the FetchToScratchTool wrapper."""

    @pytest.fixture
    def tool(self) -> FetchToScratchTool:
        """Create a FetchToScratchTool instance."""
        return FetchToScratchTool()

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """Create a mock WorkflowRuntimeContext."""
        ctx = MagicMock()
        ctx.user = MagicMock()
        ctx.user.user_id = 123
        ctx.workflow_run_id = "run_456"
        ctx.organization_id = 789
        ctx.has_scratch_store = True
        ctx.scratch_store = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_scratch_ref(self) -> ScratchDataRef:
        """Create a mock ScratchDataRef."""
        return ScratchDataRef(
            handle="oura_get_sleep_abc123",
            data_type="json",
            preview='[{"day": "2026-03-25", "score": 85}]',
            row_count=30,
            size_bytes=52000,
            storage_path="123/run_456/scratch/oura_get_sleep_abc123.json",
            workflow_run_id="run_456",
            created_at=datetime.now(timezone.utc),
        )

    async def test_executes_target_tool_and_stores_result(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
        mock_scratch_ref: ScratchDataRef,
    ) -> None:
        """Test that the tool executes the target tool and stores result in scratch."""
        # Setup mock target tool
        mock_target_tool = MagicMock()
        mock_target_tool.name = "oura_get_sleep"
        mock_target_tool.required_scopes = ["read"]
        mock_target_tool.requires_oauth = False
        mock_target_tool.required_secrets = []
        mock_target_tool.execute = AsyncMock(return_value=[{"day": "2026-03-25", "score": 85}])

        # Setup mock credential resolver
        mock_credentials = MagicMock()
        mock_credentials.access_token = "mock_token"

        mock_context.scratch_store.store = AsyncMock(return_value=mock_scratch_ref)

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(return_value=mock_credentials)
            mock_resolver_class.return_value = mock_resolver

            result = await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "oura_get_sleep",
                    "arguments": {"start_date": "2026-02-25", "end_date": "2026-03-25"},
                },
                context=mock_context,
            )

        # Verify target tool was executed
        mock_target_tool.execute.assert_called_once()

        # Verify scratch store was called
        mock_context.scratch_store.store.assert_called_once()
        store_call = mock_context.scratch_store.store.call_args
        assert store_call.kwargs["user_id"] == "123"
        assert store_call.kwargs["run_id"] == "run_456"
        assert "oura_get_sleep_" in store_call.kwargs["handle"]
        assert store_call.kwargs["data_type"] == "json"

        # Verify result is the scratch ref dict
        assert result["_type"] == "scratch_data_ref"
        assert result["handle"] == "oura_get_sleep_abc123"

    async def test_returns_error_for_unknown_tool(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
    ) -> None:
        """Test graceful error handling for unknown tool name."""
        with patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=None):
            result = await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "nonexistent_tool",
                    "arguments": {},
                },
                context=mock_context,
            )

        assert "error" in result
        assert "not found" in result["error"]

    async def test_returns_error_without_context(
        self,
        tool: FetchToScratchTool,
    ) -> None:
        """Test error when runtime context is not available."""
        result = await tool.execute(
            access_token=None,
            arguments={
                "tool_name": "oura_get_sleep",
                "arguments": {},
            },
            context=None,
        )

        assert "error" in result
        assert "context not available" in result["error"]

    async def test_returns_error_without_scratch_store(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
    ) -> None:
        """Test error when scratch store is not configured."""
        mock_context.has_scratch_store = False

        result = await tool.execute(
            access_token=None,
            arguments={
                "tool_name": "oura_get_sleep",
                "arguments": {},
            },
            context=mock_context,
        )

        assert "error" in result
        assert "not configured" in result["error"]

    async def test_passes_credentials_to_target_tool(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
        mock_scratch_ref: ScratchDataRef,
    ) -> None:
        """Test that credentials are properly resolved for OAuth tools."""
        mock_target_tool = MagicMock()
        mock_target_tool.name = "oura_get_sleep"
        mock_target_tool.required_scopes = ["email", "personal"]
        mock_target_tool.requires_oauth = True
        mock_target_tool.required_secrets = []
        mock_target_tool.execute = AsyncMock(return_value=[])

        mock_credentials = MagicMock()
        mock_credentials.access_token = "oauth_token_123"

        mock_context.scratch_store.store = AsyncMock(return_value=mock_scratch_ref)

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(return_value=mock_credentials)
            mock_resolver_class.return_value = mock_resolver

            await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "oura_get_sleep",
                    "arguments": {"start_date": "2026-02-25"},
                },
                context=mock_context,
            )

        # Verify resolver was created with correct params
        mock_resolver_class.assert_called_once()
        call_kwargs = mock_resolver_class.call_args.kwargs
        assert call_kwargs["user"] == mock_context.user
        assert call_kwargs["tool"] == mock_target_tool
        assert call_kwargs["organization_id"] == 789

        # Verify target tool received credentials
        execute_call = mock_target_tool.execute.call_args
        assert execute_call.kwargs["credentials"] == mock_credentials
        # access_token is passed as positional argument
        assert execute_call.args[0] == "oauth_token_123"

    async def test_custom_preview_rows_parameter(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
        mock_scratch_ref: ScratchDataRef,
    ) -> None:
        """Test that custom preview_rows is passed to scratch.store."""
        mock_target_tool = MagicMock()
        mock_target_tool.name = "oura_get_sleep"
        mock_target_tool.required_scopes = []
        mock_target_tool.requires_oauth = False
        mock_target_tool.required_secrets = []
        mock_target_tool.execute = AsyncMock(return_value=[{"data": "value"}])

        mock_credentials = MagicMock()
        mock_credentials.access_token = None

        mock_context.scratch_store.store = AsyncMock(return_value=mock_scratch_ref)

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(return_value=mock_credentials)
            mock_resolver_class.return_value = mock_resolver

            await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "oura_get_sleep",
                    "arguments": {},
                    "preview_rows": 10,
                },
                context=mock_context,
            )

        # Verify preview_rows was passed
        store_call = mock_context.scratch_store.store.call_args
        assert store_call.kwargs["preview_rows"] == 10

    async def test_handles_tool_execution_error(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
    ) -> None:
        """Test error handling when target tool raises exception."""
        mock_target_tool = MagicMock()
        mock_target_tool.name = "oura_get_sleep"
        mock_target_tool.required_scopes = []
        mock_target_tool.requires_oauth = False
        mock_target_tool.required_secrets = []
        mock_target_tool.execute = AsyncMock(side_effect=RuntimeError("API timeout"))

        mock_credentials = MagicMock()
        mock_credentials.access_token = None

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(return_value=mock_credentials)
            mock_resolver_class.return_value = mock_resolver

            result = await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "oura_get_sleep",
                    "arguments": {},
                },
                context=mock_context,
            )

        assert "error" in result
        assert "execution failed" in result["error"]

    async def test_handles_tool_error_response(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
    ) -> None:
        """Test handling when target tool returns an error dict."""
        mock_target_tool = MagicMock()
        mock_target_tool.name = "oura_get_sleep"
        mock_target_tool.required_scopes = []
        mock_target_tool.requires_oauth = False
        mock_target_tool.required_secrets = []
        mock_target_tool.execute = AsyncMock(return_value={"error": "Invalid date range"})

        mock_credentials = MagicMock()
        mock_credentials.access_token = None

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(return_value=mock_credentials)
            mock_resolver_class.return_value = mock_resolver

            result = await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "oura_get_sleep",
                    "arguments": {},
                },
                context=mock_context,
            )

        assert "error" in result
        assert "Invalid date range" in result["error"]

    async def test_handles_credential_resolution_error(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
    ) -> None:
        """Test error handling when credential resolution fails."""
        mock_target_tool = MagicMock()
        mock_target_tool.name = "oura_get_sleep"
        mock_target_tool.required_scopes = ["email"]
        mock_target_tool.requires_oauth = True
        mock_target_tool.required_secrets = []

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(
                side_effect=Exception("OAuth connection not found")
            )
            mock_resolver_class.return_value = mock_resolver

            result = await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "oura_get_sleep",
                    "arguments": {},
                },
                context=mock_context,
            )

        assert "error" in result
        assert "Credential resolution failed" in result["error"]

    def test_parameters_schema(self, tool: FetchToScratchTool) -> None:
        """Test that parameters schema is correctly defined."""
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "tool_name" in schema["properties"]
        assert "arguments" in schema["properties"]
        assert "preview_rows" in schema["properties"]
        assert schema["required"] == ["tool_name", "arguments"]

    def test_output_schema(self, tool: FetchToScratchTool) -> None:
        """Test that output schema is correctly defined."""
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "_type" in schema["properties"]
        assert "handle" in schema["properties"]
        assert "preview" in schema["properties"]
        assert "error" in schema["properties"]

    def test_tool_metadata(self, tool: FetchToScratchTool) -> None:
        """Test tool metadata is correct."""
        assert tool.name == "fetch_to_scratch"
        assert tool.integration_type == "scratch"
        assert tool.provider is None
        assert tool.required_scopes == []

    async def test_tool_bindings_override_resource_id(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
        mock_scratch_ref: ScratchDataRef,
    ) -> None:
        """Test that tool_bindings from workflow spec override LLM-provided resource IDs."""
        mock_target_tool = MagicMock()
        mock_target_tool.name = "postgres_execute_sql"
        mock_target_tool.required_scopes = []
        mock_target_tool.requires_oauth = False
        mock_target_tool.required_secrets = ["postgres_connection_string"]
        mock_target_tool.execute = AsyncMock(return_value={"rows": [], "row_count": 0})

        mock_credentials = MagicMock()
        mock_credentials.access_token = None

        mock_context.scratch_store.store = AsyncMock(return_value=mock_scratch_ref)
        # Simulate workflow-spec binding: postgres tool is bound to resource 7
        mock_context.tool_bindings = {
            "postgres_execute_sql": {"connection_id": None, "integration_resource_id": 7},
        }

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(return_value=mock_credentials)
            mock_resolver_class.return_value = mock_resolver

            await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "postgres_execute_sql",
                    # LLM provides resource_id=1 (wrong user), but binding should override to 7
                    "arguments": {"integration_resource_id": 1, "sql": "SELECT 1"},
                },
                context=mock_context,
            )

        # The tool args passed to execute should have resource_id=7 (from binding), not 1
        execute_call = mock_target_tool.execute.call_args
        assert execute_call.args[1]["integration_resource_id"] == 7

    async def test_no_tool_bindings_passes_args_unchanged(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
        mock_scratch_ref: ScratchDataRef,
    ) -> None:
        """Test that args pass through unchanged when no tool_bindings are set."""
        mock_target_tool = MagicMock()
        mock_target_tool.name = "postgres_execute_sql"
        mock_target_tool.required_scopes = []
        mock_target_tool.requires_oauth = False
        mock_target_tool.required_secrets = ["postgres_connection_string"]
        mock_target_tool.execute = AsyncMock(return_value={"rows": [], "row_count": 0})

        mock_credentials = MagicMock()
        mock_credentials.access_token = None

        mock_context.scratch_store.store = AsyncMock(return_value=mock_scratch_ref)
        mock_context.tool_bindings = None  # No bindings configured

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(return_value=mock_credentials)
            mock_resolver_class.return_value = mock_resolver

            await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "postgres_execute_sql",
                    "arguments": {"integration_resource_id": 1, "sql": "SELECT 1"},
                },
                context=mock_context,
            )

        # Resource ID should remain as the LLM provided it (1)
        execute_call = mock_target_tool.execute.call_args
        assert execute_call.args[1]["integration_resource_id"] == 1

    async def test_tool_bindings_for_unbound_tool_passes_unchanged(
        self,
        tool: FetchToScratchTool,
        mock_context: MagicMock,
        mock_scratch_ref: ScratchDataRef,
    ) -> None:
        """Test that args pass through when tool_bindings exist but not for this tool."""
        mock_target_tool = MagicMock()
        mock_target_tool.name = "oura_get_sleep"
        mock_target_tool.required_scopes = []
        mock_target_tool.requires_oauth = False
        mock_target_tool.required_secrets = []
        mock_target_tool.execute = AsyncMock(return_value=[{"score": 85}])

        mock_credentials = MagicMock()
        mock_credentials.access_token = None

        mock_context.scratch_store.store = AsyncMock(return_value=mock_scratch_ref)
        # Bindings exist for postgres, but not for oura
        mock_context.tool_bindings = {
            "postgres_execute_sql": {"connection_id": None, "integration_resource_id": 7},
        }

        with (
            patch("seer.tools.scratch.fetch_to_scratch.get_tool", return_value=mock_target_tool),
            patch(
                "seer.tools.credential_resolver.CredentialResolver"
            ) as mock_resolver_class,
        ):
            mock_resolver = MagicMock()
            mock_resolver.resolve = AsyncMock(return_value=mock_credentials)
            mock_resolver_class.return_value = mock_resolver

            result = await tool.execute(
                access_token=None,
                arguments={
                    "tool_name": "oura_get_sleep",
                    "arguments": {"start_date": "2026-03-25"},
                },
                context=mock_context,
            )

        # Should succeed without modification
        assert "_type" in result
        assert result["_type"] == "scratch_data_ref"
