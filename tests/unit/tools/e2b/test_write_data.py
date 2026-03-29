"""Unit tests for E2B write data tool."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.tools.e2b.write_data import E2BWriteDataTool


@pytest.mark.unit
class TestE2BWriteDataTool:
    """Test E2BWriteDataTool."""

    @pytest.fixture
    def tool(self):
        return E2BWriteDataTool()

    def test_tool_name(self, tool):
        assert tool.name == "codebox_write_data"

    def test_tool_description(self, tool):
        assert "sandbox" in tool.description.lower()
        assert "fetch_to_scratch" in tool.description

    def test_parameters_schema(self, tool):
        schema = tool.get_parameters_schema()
        assert "sandbox_id" in schema["properties"]
        assert "data" in schema["properties"]
        assert "filename" in schema["properties"]
        assert "sandbox_id" in schema["required"]
        assert "data" in schema["required"]

    @pytest.mark.asyncio
    async def test_successful_json_write(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test writing JSON data to sandbox."""
        json_data = '[{"id": 1, "name": "test"}]'

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={
                        "sandbox_id": "sbx_test123",
                        "data": json_data,
                        "filename": "results.json",
                    },
                )

                assert result["success"] is True
                assert result["sandbox_path"] == "/home/user/data/results.json"
                assert result["size_bytes"] == len(json_data.encode("utf-8"))
                assert result["error"] is None

                mock_sandbox.files.write.assert_called_once_with(
                    "/home/user/data/results.json",
                    json_data.encode("utf-8"),
                )

    @pytest.mark.asyncio
    async def test_default_filename(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test default filename is data.json."""
        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={
                        "sandbox_id": "sbx_test123",
                        "data": "{}",
                    },
                )

                assert result["success"] is True
                assert result["sandbox_path"] == "/home/user/data/data.json"

    @pytest.mark.asyncio
    async def test_csv_write(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test writing CSV data to sandbox."""
        csv_data = "id,name\n1,test\n2,other"

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={
                        "sandbox_id": "sbx_test123",
                        "data": csv_data,
                        "filename": "data.csv",
                    },
                )

                assert result["success"] is True
                assert result["sandbox_path"] == "/home/user/data/data.csv"

    @pytest.mark.asyncio
    async def test_sandbox_write_error(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test graceful handling of sandbox write errors."""
        mock_sandbox.files.write = AsyncMock(side_effect=RuntimeError("disk full"))

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={
                        "sandbox_id": "sbx_test123",
                        "data": "some data",
                    },
                )

                assert result["success"] is False
                assert "disk full" in result["error"]


@pytest.mark.unit
class TestE2BWriteDataToolRegistration:
    """Test that write data tool is registered."""

    def test_tool_registered(self):
        from seer.tools.base import clear_registry, get_tool
        from seer.tools.e2b import register_e2b_tools

        clear_registry()
        register_e2b_tools()

        assert get_tool("codebox_write_data") is not None
