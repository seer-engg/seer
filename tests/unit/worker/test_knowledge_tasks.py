"""
Unit tests for worker.tasks.knowledge module.

Tests background document processing tasks including text extraction,
chunking, embedding generation, and vector storage.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Process Document Task Tests
# =============================================================================


@pytest.mark.unit
class TestProcessDocumentTask:
    """Tests for process_document_task function."""

    @pytest.mark.asyncio
    async def test_successful_document_processing(self, mock_knowledge_document):
        """Test full pipeline success: extract → chunk → embed → store."""
        from seer.worker.tasks.knowledge import process_document_task

        content = b"This is test document content."
        content_b64 = base64.b64encode(content).decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="Extracted text content")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk1", "chunk2"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=2)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            result = await process_document_task(document_id=1, content_b64=content_b64)

        assert result["success"] is True
        assert result["document_id"] == 1
        assert result["chunk_count"] == 2

    @pytest.mark.asyncio
    async def test_returns_error_when_document_not_found(self):
        """Test early return with error when document doesn't exist."""
        from seer.worker.tasks.knowledge import process_document_task

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc:
            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=None)
            MockDoc.get_or_none.return_value = mock_query

            result = await process_document_task(document_id=999, content_b64="dGVzdA==")

        assert result["success"] is False
        assert result["error"] == "Document not found"

    @pytest.mark.asyncio
    async def test_updates_status_to_processing(self, mock_knowledge_document):
        """Test that document status is set to 'processing' initially."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=1)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            # Track status changes
            status_changes = []
            original_save = mock_knowledge_document.save

            async def track_save(*args, **kwargs):
                status_changes.append(mock_knowledge_document.processing_status)
                return await original_save(*args, **kwargs)

            mock_knowledge_document.save = track_save

            await process_document_task(document_id=1, content_b64=content_b64)

        # First status change should be "processing"
        assert "processing" in status_changes

    @pytest.mark.asyncio
    async def test_updates_status_to_completed(self, mock_knowledge_document):
        """Test that document status is 'completed' on success."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=1)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(document_id=1, content_b64=content_b64)

        assert mock_knowledge_document.processing_status == "completed"

    @pytest.mark.asyncio
    async def test_decodes_base64_content(self, mock_knowledge_document):
        """Test that base64 content is correctly decoded."""
        from seer.worker.tasks.knowledge import process_document_task

        original_content = b"Original document bytes"
        content_b64 = base64.b64encode(original_content).decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=1)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(document_id=1, content_b64=content_b64)

        # Verify extract_text received decoded bytes
        mock_processor.extract_text.assert_called_once()
        call_args = mock_processor.extract_text.call_args
        assert call_args.args[0] == original_content

    @pytest.mark.asyncio
    async def test_extracts_text_from_document(self, mock_knowledge_document):
        """Test that text extraction is called with content and mime_type."""
        from seer.worker.tasks.knowledge import process_document_task

        mock_knowledge_document.mime_type = "application/pdf"
        content_b64 = base64.b64encode(b"pdf content").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="Extracted PDF text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=1)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(document_id=1, content_b64=content_b64)

        call_args = mock_processor.extract_text.call_args
        assert call_args.args[1] == "application/pdf"

    @pytest.mark.asyncio
    async def test_fails_on_empty_text(self, mock_knowledge_document):
        """Test that empty text extraction results in failure."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"empty doc").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="   ")  # Whitespace only

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor

            result = await process_document_task(document_id=1, content_b64=content_b64)

        assert result["success"] is False
        assert "No text content" in result["error"]
        assert mock_knowledge_document.processing_status == "failed"

    @pytest.mark.asyncio
    async def test_chunks_text_with_kb_settings(self, mock_knowledge_document):
        """Test that chunking uses knowledge base settings."""
        from seer.worker.tasks.knowledge import process_document_task

        mock_knowledge_document.knowledge_base.chunk_size = 500
        mock_knowledge_document.knowledge_base.chunk_overlap = 100
        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="Long text content")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk1"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=1)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(document_id=1, content_b64=content_b64)

        # Verify ChunkingService created with KB settings
        MockChunking.assert_called_once_with(chunk_size=500, chunk_overlap=100)

    @pytest.mark.asyncio
    async def test_chunks_text_with_overrides(self, mock_knowledge_document):
        """Test that parameter overrides take precedence over KB settings."""
        from seer.worker.tasks.knowledge import process_document_task

        mock_knowledge_document.knowledge_base.chunk_size = 1000
        mock_knowledge_document.knowledge_base.chunk_overlap = 200
        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=1)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(
                document_id=1,
                content_b64=content_b64,
                chunk_size=300,
                chunk_overlap=50,
            )

        # Verify ChunkingService created with override values
        MockChunking.assert_called_once_with(chunk_size=300, chunk_overlap=50)

    @pytest.mark.asyncio
    async def test_fails_on_empty_chunks(self, mock_knowledge_document):
        """Test that empty chunks result in failure."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = []  # Empty chunks

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking

            result = await process_document_task(document_id=1, content_b64=content_b64)

        assert result["success"] is False
        assert "no chunks" in result["error"]
        assert mock_knowledge_document.processing_status == "failed"

    @pytest.mark.asyncio
    async def test_generates_embeddings(self, mock_knowledge_document):
        """Test that embeddings are generated for all chunks."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk1", "chunk2", "chunk3"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1], [0.2], [0.3]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=3)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(document_id=1, content_b64=content_b64)

        mock_embedding.embed_texts.assert_called_once_with(["chunk1", "chunk2", "chunk3"])

    @pytest.mark.asyncio
    async def test_stores_in_vector_store(self, mock_knowledge_document):
        """Test that chunks and embeddings are stored in vector store."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk1", "chunk2"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1], [0.2]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=2)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(document_id=1, content_b64=content_b64)

        mock_vector_store.insert_chunks.assert_called_once_with(
            document_id=mock_knowledge_document.id,
            kb_id=mock_knowledge_document.knowledge_base.id,
            chunks=["chunk1", "chunk2"],
            embeddings=[[0.1], [0.2]],
        )

    @pytest.mark.asyncio
    async def test_updates_chunk_count(self, mock_knowledge_document):
        """Test that chunk_count is updated on document."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk1", "chunk2", "chunk3", "chunk4"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1], [0.2], [0.3], [0.4]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=4)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(document_id=1, content_b64=content_b64)

        assert mock_knowledge_document.chunk_count == 4

    @pytest.mark.asyncio
    async def test_handles_extraction_error(self, mock_knowledge_document):
        """Test that extraction errors are handled gracefully."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"bad doc").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(side_effect=RuntimeError("Failed to extract"))

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor

            result = await process_document_task(document_id=1, content_b64=content_b64)

        assert result["success"] is False
        assert "Failed to extract" in result["error"]
        assert mock_knowledge_document.processing_status == "failed"
        assert mock_knowledge_document.processing_error == "Failed to extract"

    @pytest.mark.asyncio
    async def test_handles_embedding_error(self, mock_knowledge_document):
        """Test that embedding errors are handled gracefully."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(side_effect=RuntimeError("Embedding API error"))

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding

            result = await process_document_task(document_id=1, content_b64=content_b64)

        assert result["success"] is False
        assert "Embedding API error" in result["error"]
        assert mock_knowledge_document.processing_status == "failed"

    @pytest.mark.asyncio
    async def test_handles_vector_store_error(self, mock_knowledge_document):
        """Test that vector store errors are handled gracefully."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(side_effect=RuntimeError("DB connection failed"))

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            result = await process_document_task(document_id=1, content_b64=content_b64)

        assert result["success"] is False
        assert "DB connection failed" in result["error"]
        assert mock_knowledge_document.processing_status == "failed"

    @pytest.mark.asyncio
    async def test_truncates_long_error_messages(self, mock_knowledge_document):
        """Test that error messages are truncated to 1000 chars."""
        from seer.worker.tasks.knowledge import process_document_task

        content_b64 = base64.b64encode(b"test").decode()

        long_error = "X" * 2000  # Error longer than 1000 chars

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(side_effect=RuntimeError(long_error))

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor

            await process_document_task(document_id=1, content_b64=content_b64)

        # processing_error should be truncated to 1000 chars
        assert len(mock_knowledge_document.processing_error) == 1000

    @pytest.mark.asyncio
    async def test_clears_processing_error_on_success(self, mock_knowledge_document):
        """Test that processing_error is cleared on success."""
        from seer.worker.tasks.knowledge import process_document_task

        # Set a previous error
        mock_knowledge_document.processing_error = "Previous error"
        content_b64 = base64.b64encode(b"test").decode()

        mock_processor = MagicMock()
        mock_processor.extract_text = AsyncMock(return_value="text")

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["chunk"]

        mock_embedding = MagicMock()
        mock_embedding.embed_texts = AsyncMock(return_value=[[0.1]])

        mock_vector_store = MagicMock()
        mock_vector_store.insert_chunks = AsyncMock(return_value=1)

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as MockDoc, \
             patch("seer.worker.tasks.knowledge.get_document_processor") as mock_get_processor, \
             patch("seer.worker.tasks.knowledge.ChunkingService") as MockChunking, \
             patch("seer.worker.tasks.knowledge.get_embedding_service") as mock_get_embedding, \
             patch("seer.worker.tasks.knowledge.get_vector_store") as mock_get_vector:

            mock_query = MagicMock()
            mock_query.prefetch_related = AsyncMock(return_value=mock_knowledge_document)
            MockDoc.get_or_none.return_value = mock_query

            mock_get_processor.return_value = mock_processor
            MockChunking.return_value = mock_chunking
            mock_get_embedding.return_value = mock_embedding
            mock_get_vector.return_value = mock_vector_store

            await process_document_task(document_id=1, content_b64=content_b64)

        assert mock_knowledge_document.processing_error is None
