"""Unit tests for E2B run_code tool."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.e2b.run_code import E2BRunCodeTool

from .conftest import MockExecution, MockExecutionError, MockLogs, MockResult


@pytest.mark.unit
class TestE2BRunCodeToolMetadata:
    """Test E2BRunCodeTool metadata and schema."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return E2BRunCodeTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "codebox_run_code"

    def test_tool_description(self, tool):
        """Test tool has description."""
        assert tool.description
        assert "python" in tool.description.lower() or "sandbox" in tool.description.lower()

    def test_tool_integration_type(self, tool):
        """Test tool integration type."""
        assert tool.integration_type == "codebox"

    def test_no_oauth_required(self, tool):
        """Test that no OAuth scopes are required (uses platform API key)."""
        assert tool.required_scopes == []

    def test_parameters_schema(self, tool):
        """Test parameter schema structure."""
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Required parameters
        assert "code" in schema["required"]
        assert len(schema["required"]) == 1

        # Parameter definitions
        props = schema["properties"]
        assert "code" in props
        assert "timeout" in props

        # Type checks
        assert props["code"]["type"] == "string"
        assert props["timeout"]["type"] == "integer"

        # Default and limits
        assert props["timeout"]["default"] == 60
        assert props["timeout"]["minimum"] == 1
        assert props["timeout"]["maximum"] == 300

    def test_output_schema(self, tool):
        """Test output schema structure."""
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "success" in schema["properties"]
        assert "stdout" in schema["properties"]
        assert "stderr" in schema["properties"]
        assert "results" in schema["properties"]
        assert "error" in schema["properties"]
        assert "execution_time_ms" in schema["properties"]

    def test_get_metadata(self, tool):
        """Test that get_metadata returns complete metadata."""
        metadata = tool.get_metadata()

        assert metadata["name"] == "codebox_run_code"
        assert "description" in metadata
        assert metadata["integration_type"] == "codebox"
        assert metadata["required_scopes"] == []
        assert "parameters" in metadata
        assert "output_schema" in metadata


@pytest.mark.unit
class TestE2BRunCodeToolExecution:
    """Test E2BRunCodeTool execution."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return E2BRunCodeTool()

    @pytest.mark.asyncio
    async def test_error_when_api_key_not_configured(self, tool):
        """Test that HTTPException is raised when API key is not configured."""
        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = None

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    access_token=None,
                    arguments={"code": "print('hello')"},
                )

            assert exc_info.value.status_code == 503
            assert "E2B_API_KEY not configured" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_successful_code_execution(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test successful code execution with mocked sandbox."""
        # Set up mock execution result
        mock_sandbox.run_code.return_value = MockExecution(
            results=[MockResult(result_type="text", data="2", is_main_result=True)],
            logs=MockLogs(stdout="2\n", stderr=""),
            error=None,
        )

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            mock_config.e2b_sandbox_timeout_seconds = 300

            # Patch the import inside the method
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"code": "print(1 + 1)"},
                )

                # Verify result structure
                assert result["success"] is True
                assert result["stdout"] == "2\n"
                assert result["stderr"] == ""
                assert len(result["results"]) == 1
                assert result["results"][0]["data"] == "2"
                assert result["error"] is None
                assert "execution_time_ms" in result

                # Verify sandbox was created and killed
                mock_async_sandbox_class.create.assert_called_once()
                mock_sandbox.run_code.assert_called_once_with("print(1 + 1)")
                mock_sandbox.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_execution_with_error(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test code execution that results in an error."""
        mock_sandbox.run_code.return_value = MockExecution(
            results=[],
            logs=MockLogs(stdout="", stderr=""),
            error=MockExecutionError(
                name="NameError",
                value="name 'undefined_var' is not defined",
                traceback="...",
            ),
        )

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            mock_config.e2b_sandbox_timeout_seconds = 300

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"code": "print(undefined_var)"},
                )

                # Error should be in response, not raised as exception
                assert result["success"] is False
                assert result["error"] is not None
                assert "NameError" in result["error"]

    @pytest.mark.asyncio
    async def test_timeout_clamping(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test that timeout is clamped to maximum."""
        mock_sandbox.run_code.return_value = MockExecution()

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            mock_config.e2b_sandbox_timeout_seconds = 300

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                await tool.execute(
                    access_token=None,
                    arguments={"code": "x = 1", "timeout": 500},  # Exceeds max
                )

                # Should be clamped to 300 (max) + 30 (buffer)
                call_kwargs = mock_async_sandbox_class.create.call_args.kwargs
                assert call_kwargs["timeout"] == 330  # 300 + 30 buffer

    @pytest.mark.asyncio
    async def test_sandbox_creation_failure(self, tool, mock_async_sandbox_class):
        """Test handling of sandbox creation failure."""
        mock_async_sandbox_class.create.side_effect = Exception("E2B quota exceeded")

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            mock_config.e2b_sandbox_timeout_seconds = 300

            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys

                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                with pytest.raises(HTTPException) as exc_info:
                    await tool.execute(
                        access_token=None,
                        arguments={"code": "print('hello')"},
                    )

                assert exc_info.value.status_code == 502
                assert "Failed to create sandbox" in str(exc_info.value.detail)


@pytest.mark.unit
class TestE2BRunCodeToolRegistration:
    """Test that E2B tools can be registered."""

    def test_tools_can_be_imported(self):
        """Test that tools module can be imported."""
        from seer.tools.e2b import (
            E2BRunCodeTool,
            register_e2b_tools,
        )

        assert E2BRunCodeTool is not None
        assert callable(register_e2b_tools)

    def test_tool_registration(self):
        """Test that tool is registered in the registry."""
        from seer.tools.base import clear_registry, get_tool
        from seer.tools.e2b import register_e2b_tools

        # Clear and re-register to test
        clear_registry()
        register_e2b_tools()

        tool = get_tool("codebox_run_code")
        assert tool is not None
        assert tool.name == "codebox_run_code"
        assert isinstance(tool, E2BRunCodeTool)
