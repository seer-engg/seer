"""
Unit tests for worker.tasks.knowledge module.

Tests:
- Document processing pipeline (extract → chunk → embed → store)
- Status transitions (processing → completed / failed)
- Error handling and error message truncation
- Missing document handling
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Process Document Task
# =============================================================================


@pytest.mark.unit
class TestProcessDocumentTask:
    """Tests for process_document_task."""

    @pytest.mark.asyncio
    async def test_missing_document_returns_error(self):
        from seer.worker.tasks.knowledge import process_document_task

        with patch("seer.worker.tasks.knowledge.KnowledgeDocument") as mock_model:
            mock_model.get_or_none.return_value.prefetch_related = AsyncMock(return_value=None)

            result = await process_document_task(document_id=999, content_b64="dGVzdA==")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_successful_processing(self):
        from seer.worker.tasks.knowledge import process_document_task

        doc = MagicMock()
        doc.id = 1
        doc.mime_type = "text/plain"
        doc.save = AsyncMock()
        kb = MagicMock()
        kb.id = 10
        kb.chunk_size = 500
        kb.chunk_overlap = 50
        doc.knowledge_base = kb

        content = base64.b64encode(b"Hello world test content").decode()

        mock_processor = AsyncMock()
        mock_processor.extract_text.return_value = "Hello world test content"

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["Hello world", "test content"]

        mock_embedding = AsyncMock()
        mock_embedding.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]

        mock_vector_store = AsyncMock()
        mock_vector_store.insert_chunks.return_value = 2

        with (
            patch("seer.worker.tasks.knowledge.KnowledgeDocument") as mock_model,
            patch("seer.worker.tasks.knowledge.get_document_processor", return_value=mock_processor),
            patch("seer.worker.tasks.knowledge.ChunkingService", return_value=mock_chunking),
            patch("seer.worker.tasks.knowledge.get_embedding_service", return_value=mock_embedding),
            patch("seer.worker.tasks.knowledge.get_vector_store", return_value=mock_vector_store),
        ):
            mock_model.get_or_none.return_value.prefetch_related = AsyncMock(return_value=doc)

            result = await process_document_task(document_id=1, content_b64=content)

        assert result["success"] is True
        assert result["chunk_count"] == 2
        assert doc.processing_status == "completed"
        assert doc.chunk_count == 2

    @pytest.mark.asyncio
    async def test_empty_text_extraction_fails(self):
        from seer.worker.tasks.knowledge import process_document_task

        doc = MagicMock()
        doc.id = 1
        doc.mime_type = "text/plain"
        doc.save = AsyncMock()
        doc.knowledge_base = MagicMock()

        mock_processor = AsyncMock()
        mock_processor.extract_text.return_value = "   "  # Whitespace only

        with (
            patch("seer.worker.tasks.knowledge.KnowledgeDocument") as mock_model,
            patch("seer.worker.tasks.knowledge.get_document_processor", return_value=mock_processor),
        ):
            mock_model.get_or_none.return_value.prefetch_related = AsyncMock(return_value=doc)

            result = await process_document_task(
                document_id=1, content_b64=base64.b64encode(b"x").decode(),
            )

        assert result["success"] is False
        assert doc.processing_status == "failed"
        assert "No text content" in doc.processing_error

    @pytest.mark.asyncio
    async def test_error_message_truncated_to_1000_chars(self):
        from seer.worker.tasks.knowledge import process_document_task

        doc = MagicMock()
        doc.id = 1
        doc.mime_type = "text/plain"
        doc.save = AsyncMock()
        doc.knowledge_base = MagicMock()

        mock_processor = AsyncMock()
        mock_processor.extract_text.side_effect = RuntimeError("x" * 2000)

        with (
            patch("seer.worker.tasks.knowledge.KnowledgeDocument") as mock_model,
            patch("seer.worker.tasks.knowledge.get_document_processor", return_value=mock_processor),
        ):
            mock_model.get_or_none.return_value.prefetch_related = AsyncMock(return_value=doc)

            result = await process_document_task(
                document_id=1, content_b64=base64.b64encode(b"x").decode(),
            )

        assert doc.processing_status == "failed"
        assert len(doc.processing_error) <= 1000

    @pytest.mark.asyncio
    async def test_custom_chunk_size_overrides_kb_settings(self):
        from seer.worker.tasks.knowledge import process_document_task

        doc = MagicMock()
        doc.id = 1
        doc.mime_type = "text/plain"
        doc.save = AsyncMock()
        kb = MagicMock()
        kb.id = 10
        kb.chunk_size = 500
        kb.chunk_overlap = 50
        doc.knowledge_base = kb

        mock_processor = AsyncMock()
        mock_processor.extract_text.return_value = "test content"

        captured_args = {}

        def capture_chunking_init(chunk_size, chunk_overlap):
            captured_args["chunk_size"] = chunk_size
            captured_args["chunk_overlap"] = chunk_overlap
            mock = MagicMock()
            mock.chunk_text.return_value = ["test"]
            return mock

        mock_embedding = AsyncMock()
        mock_embedding.embed_texts.return_value = [[0.1]]
        mock_vector_store = AsyncMock()
        mock_vector_store.insert_chunks.return_value = 1

        with (
            patch("seer.worker.tasks.knowledge.KnowledgeDocument") as mock_model,
            patch("seer.worker.tasks.knowledge.get_document_processor", return_value=mock_processor),
            patch("seer.worker.tasks.knowledge.ChunkingService", side_effect=capture_chunking_init),
            patch("seer.worker.tasks.knowledge.get_embedding_service", return_value=mock_embedding),
            patch("seer.worker.tasks.knowledge.get_vector_store", return_value=mock_vector_store),
        ):
            mock_model.get_or_none.return_value.prefetch_related = AsyncMock(return_value=doc)

            await process_document_task(
                document_id=1,
                content_b64=base64.b64encode(b"test").decode(),
                chunk_size=200,
                chunk_overlap=20,
            )

        assert captured_args["chunk_size"] == 200
        assert captured_args["chunk_overlap"] == 20

    @pytest.mark.asyncio
    async def test_sets_processing_status_before_work(self):
        from seer.worker.tasks.knowledge import process_document_task

        doc = MagicMock()
        doc.id = 1
        doc.mime_type = "text/plain"
        doc.knowledge_base = MagicMock(id=10, chunk_size=500, chunk_overlap=50)
        statuses = []

        async def track_save(*args, **kwargs):
            statuses.append(doc.processing_status)

        doc.save = AsyncMock(side_effect=track_save)

        mock_processor = AsyncMock()
        mock_processor.extract_text.return_value = "content"

        mock_chunking = MagicMock()
        mock_chunking.chunk_text.return_value = ["content"]

        mock_embedding = AsyncMock()
        mock_embedding.embed_texts.return_value = [[0.1]]

        mock_vector_store = AsyncMock()
        mock_vector_store.insert_chunks.return_value = 1

        with (
            patch("seer.worker.tasks.knowledge.KnowledgeDocument") as mock_model,
            patch("seer.worker.tasks.knowledge.get_document_processor", return_value=mock_processor),
            patch("seer.worker.tasks.knowledge.ChunkingService", return_value=mock_chunking),
            patch("seer.worker.tasks.knowledge.get_embedding_service", return_value=mock_embedding),
            patch("seer.worker.tasks.knowledge.get_vector_store", return_value=mock_vector_store),
        ):
            mock_model.get_or_none.return_value.prefetch_related = AsyncMock(return_value=doc)

            await process_document_task(
                document_id=1, content_b64=base64.b64encode(b"content").decode(),
            )

        # First save should set "processing", last should set "completed"
        assert statuses[0] == "processing"
        assert statuses[-1] == "completed"
