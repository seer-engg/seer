"""
Tests for FileContentProcessor - the component that processes files for LLM input.

Tests cover:
- Image file processing (conversion to base64 data URLs)
- Document text extraction (PDF, DOCX, TXT)
- Mixed file handling
- Unsupported file type handling
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest

from seer.core.runtime.file_processor import (
    DOCUMENT_MIME_TYPES,
    IMAGE_MIME_TYPES,
    FileContentProcessor,
    is_supported_for_llm,
)


class TestFileContentProcessor:
    """Tests for FileContentProcessor class."""

    @pytest.mark.asyncio
    async def test_process_png_image(self):
        """PNG image is converted to base64 data URL content block."""
        processor = FileContentProcessor()
        image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"  # PNG header

        file_contents = [
            {
                "key": "image",
                "mime_type": "image/png",
                "filename": "test.png",
                "content": image_bytes,
            }
        ]

        result = await processor.process_files(file_contents)

        assert len(result["image_blocks"]) == 1
        assert result["image_blocks"][0]["type"] == "image_url"
        assert "image_url" in result["image_blocks"][0]

        # Verify data URL format
        data_url = result["image_blocks"][0]["image_url"]["url"]
        assert data_url.startswith("data:image/png;base64,")

        # Verify base64 encoding is correct
        b64_part = data_url.split(",")[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == image_bytes

        # No text extracted from images
        assert result["extracted_text"] == ""

    @pytest.mark.asyncio
    async def test_process_jpeg_image(self):
        """JPEG image is converted to base64 data URL content block."""
        processor = FileContentProcessor()
        image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # JPEG header

        file_contents = [
            {
                "key": "photo",
                "mime_type": "image/jpeg",
                "filename": "photo.jpg",
                "content": image_bytes,
            }
        ]

        result = await processor.process_files(file_contents)

        assert len(result["image_blocks"]) == 1
        data_url = result["image_blocks"][0]["image_url"]["url"]
        assert data_url.startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_process_pdf_document(self):
        """PDF document has text extracted via DocumentProcessor."""
        processor = FileContentProcessor()

        # Mock the document processor
        with patch.object(processor, "_get_document_processor") as mock_get_dp:
            mock_dp = AsyncMock()
            mock_dp.extract_text = AsyncMock(return_value="Extracted PDF content here.")
            mock_get_dp.return_value = mock_dp

            file_contents = [
                {
                    "key": "document",
                    "mime_type": "application/pdf",
                    "filename": "report.pdf",
                    "content": b"%PDF-1.4 fake pdf content",
                }
            ]

            result = await processor.process_files(file_contents)

            assert result["image_blocks"] == []
            assert "report.pdf" in result["extracted_text"]
            assert "Extracted PDF content here." in result["extracted_text"]

            mock_dp.extract_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_plain_text(self):
        """Plain text file has content extracted."""
        processor = FileContentProcessor()

        with patch.object(processor, "_get_document_processor") as mock_get_dp:
            mock_dp = AsyncMock()
            mock_dp.extract_text = AsyncMock(return_value="Hello, this is plain text content.")
            mock_get_dp.return_value = mock_dp

            file_contents = [
                {
                    "key": "textfile",
                    "mime_type": "text/plain",
                    "filename": "notes.txt",
                    "content": b"Hello, this is plain text content.",
                }
            ]

            result = await processor.process_files(file_contents)

            assert result["image_blocks"] == []
            assert "notes.txt" in result["extracted_text"]
            assert "Hello, this is plain text content." in result["extracted_text"]

    @pytest.mark.asyncio
    async def test_process_mixed_files(self):
        """Mix of images and documents are processed correctly."""
        processor = FileContentProcessor()

        with patch.object(processor, "_get_document_processor") as mock_get_dp:
            mock_dp = AsyncMock()
            mock_dp.extract_text = AsyncMock(return_value="Document text")
            mock_get_dp.return_value = mock_dp

            file_contents = [
                {
                    "key": "image",
                    "mime_type": "image/png",
                    "filename": "image.png",
                    "content": b"\x89PNG",
                },
                {
                    "key": "doc",
                    "mime_type": "application/pdf",
                    "filename": "doc.pdf",
                    "content": b"%PDF",
                },
                {
                    "key": "photo",
                    "mime_type": "image/webp",
                    "filename": "photo.webp",
                    "content": b"RIFF",
                },
            ]

            result = await processor.process_files(file_contents)

            # Two images
            assert len(result["image_blocks"]) == 2

            # One document
            assert "doc.pdf" in result["extracted_text"]
            assert "Document text" in result["extracted_text"]

    @pytest.mark.asyncio
    async def test_unsupported_mime_type(self):
        """Unsupported MIME types are logged as warning and skipped."""
        processor = FileContentProcessor()

        file_contents = [
            {
                "key": "video",
                "mime_type": "video/mp4",
                "filename": "movie.mp4",
                "content": b"\x00\x00\x00\x1c",
            }
        ]

        result = await processor.process_files(file_contents)

        # No content processed
        assert result["image_blocks"] == []
        assert result["extracted_text"] == ""

    @pytest.mark.asyncio
    async def test_document_extraction_error_handled(self):
        """Document extraction errors are handled gracefully."""
        processor = FileContentProcessor()

        with patch.object(processor, "_get_document_processor") as mock_get_dp:
            mock_dp = AsyncMock()
            mock_dp.extract_text = AsyncMock(side_effect=Exception("PDF parsing failed"))
            mock_get_dp.return_value = mock_dp

            file_contents = [
                {
                    "key": "broken",
                    "mime_type": "application/pdf",
                    "filename": "corrupt.pdf",
                    "content": b"not a real pdf",
                }
            ]

            result = await processor.process_files(file_contents)

            # Error is included in text instead of raising
            assert "Error extracting text from corrupt.pdf" in result["extracted_text"]

    @pytest.mark.asyncio
    async def test_empty_file_list(self):
        """Empty file list returns empty results."""
        processor = FileContentProcessor()

        result = await processor.process_files([])

        assert result["image_blocks"] == []
        assert result["extracted_text"] == ""


class TestIsSupportedForLLM:
    """Tests for is_supported_for_llm helper function."""

    @pytest.mark.parametrize("mime_type", list(IMAGE_MIME_TYPES))
    def test_image_types_supported(self, mime_type: str):
        """All image MIME types are supported."""
        assert is_supported_for_llm(mime_type) is True

    @pytest.mark.parametrize("mime_type", list(DOCUMENT_MIME_TYPES))
    def test_document_types_supported(self, mime_type: str):
        """All document MIME types are supported."""
        assert is_supported_for_llm(mime_type) is True

    def test_unsupported_types(self):
        """Unsupported types return False."""
        assert is_supported_for_llm("video/mp4") is False
        assert is_supported_for_llm("audio/mpeg") is False
        assert is_supported_for_llm("application/zip") is False

    def test_case_insensitive(self):
        """MIME type checking is case-insensitive."""
        assert is_supported_for_llm("IMAGE/PNG") is True
        assert is_supported_for_llm("Application/PDF") is True
