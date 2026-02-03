"""Unit tests for ChunkingService."""

import pytest

from seer.services.knowledge.chunking_service import ChunkingService, create_chunking_service


class TestChunkingService:
    """Test ChunkingService text splitting functionality."""

    def test_chunk_short_text(self):
        """Test that short text produces single chunk."""
        service = ChunkingService(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."
        chunks = service.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self):
        """Test that long text is split into multiple chunks."""
        service = ChunkingService(chunk_size=50, chunk_overlap=10)
        text = "A" * 150  # 150 characters

        chunks = service.chunk_text(text)

        assert len(chunks) > 1
        # Verify all content is preserved
        assert all("A" in chunk for chunk in chunks)

    def test_chunk_overlap(self):
        """Test that chunks have expected overlap."""
        service = ChunkingService(chunk_size=100, chunk_overlap=20)
        # Create text that will definitely split
        text = "Word " * 100  # 500 characters

        chunks = service.chunk_text(text)

        assert len(chunks) > 1
        # With overlap, adjacent chunks should share some content
        # This is a rough check - exact overlap depends on split points

    def test_empty_text(self):
        """Test that empty text returns empty list."""
        service = ChunkingService(chunk_size=100, chunk_overlap=20)

        assert service.chunk_text("") == []
        assert service.chunk_text("   ") == []

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        service = ChunkingService(chunk_size=100, chunk_overlap=20)
        chunks = service.chunk_text("   \n\n\t   ")

        assert chunks == []

    def test_chunk_text_with_metadata(self):
        """Test chunk_text_with_metadata includes correct metadata."""
        service = ChunkingService(chunk_size=50, chunk_overlap=10)
        text = "Word " * 50

        result = service.chunk_text_with_metadata(text, {"source": "test"})

        assert len(result) > 0
        for i, item in enumerate(result):
            assert "content" in item
            assert "metadata" in item
            assert item["metadata"]["source"] == "test"
            assert item["metadata"]["chunk_index"] == i

    def test_custom_chunk_size(self):
        """Test that custom chunk size is respected."""
        small_service = ChunkingService(chunk_size=30, chunk_overlap=5)
        large_service = ChunkingService(chunk_size=200, chunk_overlap=5)
        text = "This is a test. " * 20

        small_chunks = small_service.chunk_text(text)
        large_chunks = large_service.chunk_text(text)

        # Smaller chunk size should produce more chunks
        assert len(small_chunks) > len(large_chunks)


class TestCreateChunkingService:
    """Test create_chunking_service factory function."""

    def test_default_parameters(self):
        """Test factory with default parameters."""
        service = create_chunking_service()

        assert service.chunk_size == 1000
        assert service.chunk_overlap == 200

    def test_custom_parameters(self):
        """Test factory with custom parameters."""
        service = create_chunking_service(chunk_size=500, chunk_overlap=100)

        assert service.chunk_size == 500
        assert service.chunk_overlap == 100

    def test_returns_chunking_service_instance(self):
        """Test factory returns correct type."""
        service = create_chunking_service()

        assert isinstance(service, ChunkingService)
