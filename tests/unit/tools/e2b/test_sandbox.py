"""Unit tests for E2B persistent sandbox tools."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.e2b.sandbox import (
    E2BCreateSandboxTool,
    E2BKillSandboxTool,
    E2BRunInSandboxTool,
)

from .conftest import MockExecution, MockLogs, MockResult


@pytest.mark.unit
class TestE2BCreateSandboxTool:
    """Test E2BCreateSandboxTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return E2BCreateSandboxTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "codebox_create_sandbox"

    def test_tool_description(self, tool):
        """Test tool has description."""
        assert tool.description
        assert "persistent" in tool.description.lower() or "create" in tool.description.lower()

    def test_no_oauth_required(self, tool):
        """Test that no OAuth scopes are required."""
        assert tool.required_scopes == []

    def test_parameters_schema(self, tool):
        """Test parameter schema structure."""
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "timeout" in schema["properties"]
        assert schema["required"] == []  # No required params

    def test_output_schema(self, tool):
        """Test output schema structure."""
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "sandbox_id" in schema["properties"]
        assert "timeout_seconds" in schema["properties"]

    @pytest.mark.asyncio
    async def test_successful_sandbox_creation(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test successful sandbox creation."""
        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            mock_config.e2b_sandbox_timeout_seconds = 300

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={},
                )

                assert result["sandbox_id"] == "sbx_test123"
                assert result["timeout_seconds"] == 300

                # Sandbox should NOT be killed (persistent)
                mock_sandbox.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_timeout(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test sandbox creation with custom timeout."""
        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            mock_config.e2b_sandbox_timeout_seconds = 300

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"timeout": 600},
                )

                assert result["timeout_seconds"] == 600
                call_kwargs = mock_async_sandbox_class.create.call_args.kwargs
                assert call_kwargs["timeout"] == 600


@pytest.mark.unit
class TestE2BRunInSandboxTool:
    """Test E2BRunInSandboxTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return E2BRunInSandboxTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "codebox_run_in_sandbox"

    def test_parameters_schema(self, tool):
        """Test parameter schema structure."""
        schema = tool.get_parameters_schema()

        assert "sandbox_id" in schema["properties"]
        assert "code" in schema["properties"]
        assert "sandbox_id" in schema["required"]
        assert "code" in schema["required"]

    @pytest.mark.asyncio
    async def test_successful_execution_in_sandbox(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test successful code execution in existing sandbox."""
        mock_sandbox.run_code.return_value = MockExecution(
            results=[MockResult(result_type="text", data="42")],
            logs=MockLogs(stdout="42\n", stderr=""),
            error=None,
        )

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={
                        "sandbox_id": "sbx_test123",
                        "code": "print(42)",
                    },
                )

                assert result["success"] is True
                assert result["stdout"] == "42\n"
                assert len(result["results"]) == 1

                # Should connect to existing sandbox
                mock_async_sandbox_class.connect.assert_called_once_with(
                    sandbox_id="sbx_test123",
                    api_key="test-api-key",
                )

    @pytest.mark.asyncio
    async def test_sandbox_not_found(self, tool, mock_async_sandbox_class):
        """Test error when sandbox not found."""
        mock_async_sandbox_class.connect.side_effect = Exception("Sandbox not found")

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                with pytest.raises(HTTPException) as exc_info:
                    await tool.execute(
                        access_token=None,
                        arguments={
                            "sandbox_id": "sbx_invalid",
                            "code": "print(1)",
                        },
                    )

                assert exc_info.value.status_code == 404
                assert "not found or expired" in str(exc_info.value.detail)


@pytest.mark.unit
class TestE2BKillSandboxTool:
    """Test E2BKillSandboxTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return E2BKillSandboxTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "codebox_kill_sandbox"

    def test_parameters_schema(self, tool):
        """Test parameter schema structure."""
        schema = tool.get_parameters_schema()

        assert "sandbox_id" in schema["properties"]
        assert "sandbox_id" in schema["required"]

    def test_output_schema(self, tool):
        """Test output schema structure."""
        schema = tool.get_output_schema()

        assert "success" in schema["properties"]
        assert "message" in schema["properties"]

    @pytest.mark.asyncio
    async def test_successful_sandbox_kill(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test successful sandbox termination."""
        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"sandbox_id": "sbx_test123"},
                )

                assert result["success"] is True
                assert "terminated" in result["message"]
                mock_sandbox.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_nonexistent_sandbox(self, tool, mock_async_sandbox_class):
        """Test killing a non-existent sandbox returns failure, not exception."""
        mock_async_sandbox_class.connect.side_effect = Exception("Sandbox not found")

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                # Kill should return failure response, not raise exception
                result = await tool.execute(
                    access_token=None,
                    arguments={"sandbox_id": "sbx_invalid"},
                )

                assert result["success"] is False
                assert "Sandbox not found" in result["message"] or "not found" in result["message"]


@pytest.mark.unit
class TestE2BSandboxToolsRegistration:
    """Test that all E2B sandbox tools are registered."""

    def test_all_tools_registered(self):
        """Test that all sandbox tools are registered."""
        from seer.tools.base import clear_registry, get_tool
        from seer.tools.e2b import register_e2b_tools

        clear_registry()
        register_e2b_tools()

        # Verify all tools are registered
        assert get_tool("codebox_create_sandbox") is not None
        assert get_tool("codebox_run_in_sandbox") is not None
        assert get_tool("codebox_kill_sandbox") is not None
        assert get_tool("codebox_run_code") is not None
