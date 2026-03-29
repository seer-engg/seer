"""Unit tests for E2B download file tool."""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.tools.e2b.download_file import E2BDownloadFileTool


@pytest.mark.unit
class TestE2BDownloadFileTool:
    """Test E2BDownloadFileTool."""

    @pytest.fixture
    def tool(self):
        return E2BDownloadFileTool()

    def test_tool_name(self, tool):
        assert tool.name == "codebox_download_file"

    def test_tool_description(self, tool):
        assert "data_uri" in tool.description
        assert "create_artifact" in tool.description

    def test_parameters_schema(self, tool):
        schema = tool.get_parameters_schema()
        assert "sandbox_id" in schema["properties"]
        assert "path" in schema["properties"]
        assert "max_size_bytes" in schema["properties"]
        assert "sandbox_id" in schema["required"]
        assert "path" in schema["required"]

    @pytest.mark.asyncio
    async def test_successful_png_download(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test downloading a PNG file returns correct data URI."""
        png_data = bytearray(b"\x89PNG\r\n\x1a\nfake_png_data")  # E2B returns bytearray with format="bytes"
        mock_sandbox.files.read = AsyncMock(return_value=png_data)

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"sandbox_id": "sbx_test123", "path": "/mnt/data/chart.png"},
                )

                assert result["success"] is True
                assert result["mime_type"] == "image/png"
                assert result["size_bytes"] == len(png_data)
                assert result["data_uri"].startswith("data:image/png;base64,")
                # Verify base64 roundtrip
                b64_part = result["data_uri"].split(",", 1)[1]
                assert base64.b64decode(b64_part) == bytes(png_data)
                # Verify format="bytes" was passed
                mock_sandbox.files.read.assert_called_once_with("/mnt/data/chart.png", format="bytes")

    @pytest.mark.asyncio
    async def test_successful_svg_download(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test downloading an SVG file."""
        svg_data = bytearray(b"<svg><circle/></svg>")
        mock_sandbox.files.read = AsyncMock(return_value=svg_data)

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"sandbox_id": "sbx_test123", "path": "/tmp/plot.svg"},
                )

                assert result["success"] is True
                assert result["mime_type"] == "image/svg+xml"

    @pytest.mark.asyncio
    async def test_file_too_large(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test rejection of files exceeding max_size_bytes."""
        large_data = bytearray(b"x" * 1000)
        mock_sandbox.files.read = AsyncMock(return_value=large_data)

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={
                        "sandbox_id": "sbx_test123",
                        "path": "/mnt/data/huge.png",
                        "max_size_bytes": 500,
                    },
                )

                assert result["success"] is False
                assert "exceeds" in result["error"]
                assert result["size_bytes"] == 1000

    @pytest.mark.asyncio
    async def test_unknown_extension_fallback(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test that unknown extensions default to application/octet-stream."""
        mock_sandbox.files.read = AsyncMock(return_value=bytearray(b"binary stuff"))

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"sandbox_id": "sbx_test123", "path": "/tmp/data.xyz"},
                )

                assert result["success"] is True
                assert result["mime_type"] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_bytearray_return_from_e2b(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test that bytearray from E2B format='bytes' is handled correctly."""
        # E2B SDK returns bytearray when format="bytes"
        binary_data = bytearray(b"\x89PNG\r\n\x1a\n\x00\xff\xfe\xfd")
        mock_sandbox.files.read = AsyncMock(return_value=binary_data)

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"sandbox_id": "sbx_test123", "path": "/mnt/data/chart.png"},
                )

                assert result["success"] is True
                assert result["data_uri"].startswith("data:image/png;base64,")
                # Verify binary data survives roundtrip (including bytes > 0x7F)
                b64_part = result["data_uri"].split(",", 1)[1]
                assert base64.b64decode(b64_part) == bytes(binary_data)

    @pytest.mark.asyncio
    async def test_sandbox_error(self, tool, mock_sandbox, mock_async_sandbox_class):
        """Test graceful handling of sandbox read errors."""
        mock_sandbox.files.read = AsyncMock(side_effect=RuntimeError("file not found"))

        with patch("seer.tools.e2b.base.config") as mock_config:
            mock_config.e2b_api_key = "test-api-key"
            with patch.dict("sys.modules", {"e2b_code_interpreter": MagicMock()}):
                import sys
                sys.modules["e2b_code_interpreter"].AsyncSandbox = mock_async_sandbox_class

                result = await tool.execute(
                    access_token=None,
                    arguments={"sandbox_id": "sbx_test123", "path": "/mnt/data/missing.png"},
                )

                assert result["success"] is False
                assert "file not found" in result["error"]


@pytest.mark.unit
class TestE2BDownloadFileToolRegistration:
    """Test that download file tool is registered."""

    def test_tool_registered(self):
        from seer.tools.base import clear_registry, get_tool
        from seer.tools.e2b import register_e2b_tools

        clear_registry()
        register_e2b_tools()

        assert get_tool("codebox_download_file") is not None
