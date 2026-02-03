"""Unit tests for DocumentProcessor."""

import pytest

from seer.services.knowledge.document_processor import (
    DocumentProcessor,
    SUPPORTED_MIME_TYPES,
    MIME_TO_EXTENSION,
    get_document_processor,
)


class TestDocumentProcessor:
    """Test DocumentProcessor text extraction functionality."""

    @pytest.fixture
    def processor(self):
        """Create document processor instance."""
        return DocumentProcessor()

    def test_is_supported_valid_types(self, processor):
        """Test is_supported returns True for valid MIME types."""
        assert processor.is_supported("text/plain") is True
        assert processor.is_supported("application/pdf") is True
        assert processor.is_supported("application/vnd.openxmlformats-officedocument.wordprocessingml.document") is True

    def test_is_supported_invalid_types(self, processor):
        """Test is_supported returns False for unsupported MIME types."""
        assert processor.is_supported("image/png") is False
        assert processor.is_supported("video/mp4") is False
        assert processor.is_supported("application/octet-stream") is False

    @pytest.mark.asyncio
    async def test_extract_plain_text_utf8(self, processor):
        """Test plain text extraction with UTF-8 encoding."""
        content = "Hello, World!".encode("utf-8")
        text = await processor.extract_text(content, "text/plain")

        assert text == "Hello, World!"

    @pytest.mark.asyncio
    async def test_extract_plain_text_latin1_fallback(self, processor):
        """Test plain text extraction falls back to latin-1 for non-UTF8."""
        # Create content with latin-1 specific character
        content = "Caf\xe9".encode("latin-1")
        text = await processor.extract_text(content, "text/plain")

        assert "Caf" in text

    @pytest.mark.asyncio
    async def test_extract_unsupported_type_raises(self, processor):
        """Test that unsupported MIME type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported MIME type"):
            await processor.extract_text(b"test", "image/png")


class TestSupportedMimeTypes:
    """Test SUPPORTED_MIME_TYPES constant."""

    def test_supported_types_set(self):
        """Test that SUPPORTED_MIME_TYPES is a set with expected values."""
        assert isinstance(SUPPORTED_MIME_TYPES, set)
        assert "text/plain" in SUPPORTED_MIME_TYPES
        assert "application/pdf" in SUPPORTED_MIME_TYPES
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in SUPPORTED_MIME_TYPES


class TestMimeToExtension:
    """Test MIME_TO_EXTENSION mapping."""

    def test_mime_to_extension_mapping(self):
        """Test MIME type to extension mapping."""
        assert MIME_TO_EXTENSION["text/plain"] == ".txt"
        assert MIME_TO_EXTENSION["application/pdf"] == ".pdf"
        assert MIME_TO_EXTENSION["application/vnd.openxmlformats-officedocument.wordprocessingml.document"] == ".docx"


class TestGetDocumentProcessor:
    """Test get_document_processor singleton factory."""

    def test_returns_document_processor(self):
        """Test factory returns DocumentProcessor instance."""
        processor = get_document_processor()
        assert isinstance(processor, DocumentProcessor)

    def test_returns_same_instance(self):
        """Test factory returns singleton instance."""
        processor1 = get_document_processor()
        processor2 = get_document_processor()
        assert processor1 is processor2
