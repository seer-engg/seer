"""Unit tests for E2B data loader tool."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.tools.e2b.data_loader import E2BLoadDataTool


@pytest.fixture
def mock_context():
    """Create a mock runtime context with scratch store."""
    ctx = MagicMock()
    ctx.has_scratch_store = True
    ctx.user = MagicMock()
    ctx.user.user_id = "user_123"
    ctx.workflow_run_id = "run_456"

    # Mock scratch store
    ctx.scratch_store = MagicMock()
    ctx.scratch_store.retrieve = AsyncMock(return_value=b'[{"score": 85}]')

    return ctx


@pytest.fixture
def mock_sandbox():
    """Create a mock AsyncSandbox with files API."""
    sandbox = AsyncMock()
    sandbox.sandbox_id = "sbx_test123"
    sandbox.files = MagicMock()
    sandbox.files.write = AsyncMock()
    return sandbox


@pytest.fixture
def mock_async_sandbox_class(mock_sandbox):
    """Create a mock AsyncSandbox class."""
    mock_class = MagicMock()
    mock_class.connect = AsyncMock(return_value=mock_sandbox)
    return mock_class


@pytest.mark.unit
class TestE2BLoadDataTool:
    """Test E2BLoadDataTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return E2BLoadDataTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "codebox_load_data"

    def test_tool_description(self, tool):
        """Test tool has description."""
        assert tool.description
        assert "load" in tool.description.lower()
        assert "sandbox" in tool.description.lower()

    def test_no_oauth_required(self, tool):
        """Test that no OAuth scopes are required."""
        assert tool.required_scopes == []

    def test_parameters_schema(self, tool):
        """Test parameter schema structure."""
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "sandbox_id" in schema["properties"]
        assert "handle" in schema["properties"]
        assert "filename" in schema["properties"]
        assert "sandbox_id" in schema["required"]
        assert "handle" in schema["required"]

    def test_output_schema(self, tool):
        """Test output schema structure."""
        schema = tool.get_output_schema()

        assert "success" in schema["properties"]
        assert "sandbox_path" in schema["properties"]
        assert "size_bytes" in schema["properties"]
        assert "error" in schema["properties"]

    @pytest.mark.asyncio
    async def test_successful_data_load(
        self, tool, mock_context, mock_sandbox, mock_async_sandbox_class
    ):
        """Test successful data loading to sandbox."""
        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={
                        "sandbox_id": "sbx_test123",
                        "handle": "oura_sleep_abc",
                    },
                    context=mock_context,
                )

                assert result["success"] is True
                assert result["sandbox_path"] == "/home/user/data/oura_sleep_abc.json"
                assert result["size_bytes"] == len(b'[{"score": 85}]')
                assert result["error"] is None

                # Verify files.write was called
                mock_sandbox.files.write.assert_called_once()
                call_args = mock_sandbox.files.write.call_args
                assert call_args[0][0] == "/home/user/data/oura_sleep_abc.json"
                assert call_args[0][1] == b'[{"score": 85}]'

    @pytest.mark.asyncio
    async def test_custom_filename(
        self, tool, mock_context, mock_sandbox, mock_async_sandbox_class
    ):
        """Test data loading with custom filename."""
        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={
                        "sandbox_id": "sbx_test123",
                        "handle": "oura_sleep_abc",
                        "filename": "sleep_data.json",
                    },
                    context=mock_context,
                )

                assert result["success"] is True
                assert result["sandbox_path"] == "/home/user/data/sleep_data.json"

    @pytest.mark.asyncio
    async def test_no_context(self, tool):
        """Test error when no context available."""
        result = await tool.execute(
            access_token=None,
            arguments={
                "sandbox_id": "sbx_test123",
                "handle": "test_handle",
            },
            context=None,
        )

        assert result["success"] is False
        assert "context not available" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_no_scratch_store(self, tool, mock_context):
        """Test error when scratch store not configured."""
        mock_context.has_scratch_store = False

        result = await tool.execute(
            access_token=None,
            arguments={
                "sandbox_id": "sbx_test123",
                "handle": "test_handle",
            },
            context=mock_context,
        )

        assert result["success"] is False
        assert "not configured" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_data_not_found(
        self, tool, mock_context, mock_async_sandbox_class
    ):
        """Test error when scratch data not found."""
        mock_context.scratch_store.retrieve.side_effect = FileNotFoundError("not found")

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"

            result = await tool.execute(
                access_token=None,
                arguments={
                    "sandbox_id": "sbx_test123",
                    "handle": "nonexistent",
                },
                context=mock_context,
            )

            assert result["success"] is False
            assert "not found" in result["error"].lower()


@pytest.mark.unit
class TestE2BLoadDataToolRegistration:
    """Test that E2B load data tool is registered."""

    def test_tool_registered(self):
        """Test that load data tool is registered."""
        from seer.tools.base import clear_registry, get_tool
        from seer.tools.e2b import register_e2b_tools

        clear_registry()
        register_e2b_tools()

        assert get_tool("codebox_load_data") is not None
