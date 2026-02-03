"""Chunking service for splitting text into overlapping chunks."""
from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from seer.logger import get_logger

logger = get_logger("services.knowledge.chunking")


class ChunkingService:
    """Service for splitting text documents into chunks for embedding."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize chunking service.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text content to split

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = self._splitter.split_text(text)
        logger.debug(
            "Chunked text",
            extra={
                "input_length": len(text),
                "chunk_count": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )
        return chunks

    def chunk_text_with_metadata(self, text: str, base_metadata: dict | None = None) -> List[dict]:
        """Split text into chunks with metadata.

        Args:
            text: Text content to split
            base_metadata: Base metadata to include with each chunk

        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        chunks = self.chunk_text(text)
        base_metadata = base_metadata or {}

        return [
            {
                "content": chunk,
                "metadata": {
                    **base_metadata,
                    "chunk_index": i,
                    "char_start": sum(len(c) for c in chunks[:i]) - i * self.chunk_overlap if i > 0 else 0,
                },
            }
            for i, chunk in enumerate(chunks)
        ]


def create_chunking_service(chunk_size: int = 1000, chunk_overlap: int = 200) -> ChunkingService:
    """Factory function to create a chunking service with specified parameters.

    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Configured ChunkingService instance
    """
    return ChunkingService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


__all__ = ["ChunkingService", "create_chunking_service"]
