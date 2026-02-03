"""Background tasks for knowledge base document processing."""
from __future__ import annotations

import base64
from typing import Optional

from seer.database.knowledge_models import KnowledgeDocument
from seer.logger import get_logger
from seer.services.knowledge.chunking_service import ChunkingService
from seer.services.knowledge.document_processor import get_document_processor
from seer.services.knowledge.embedding_service import get_embedding_service
from seer.services.knowledge.vector_store import get_vector_store
from seer.worker.broker_instance import broker

logger = get_logger(__name__)


@broker.task
async def process_document_task(
    document_id: int,
    content_b64: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    """Process a document in the background: extract text, chunk, embed, and store.

    This task handles the async processing of uploaded documents:
    1. Extract text from the document based on MIME type
    2. Split text into overlapping chunks
    3. Generate embeddings for each chunk
    4. Store chunks and embeddings in pgvector

    Args:
        document_id: Internal ID of the KnowledgeDocument
        content_b64: Base64-encoded document content
        chunk_size: Optional override for chunk size (uses KB settings if not provided)
        chunk_overlap: Optional override for chunk overlap (uses KB settings if not provided)

    Returns:
        dict with processing result including chunk_count
    """
    logger.info("Starting document processing", extra={"document_id": document_id})

    # Fetch the document with its knowledge base
    doc = await KnowledgeDocument.get_or_none(id=document_id).prefetch_related("knowledge_base")
    if not doc:
        logger.error("Document not found", extra={"document_id": document_id})
        return {"success": False, "error": "Document not found"}

    # Update status to processing
    doc.processing_status = "processing"
    await doc.save(update_fields=["processing_status", "updated_at"])

    try:
        # Decode content
        content = base64.b64decode(content_b64)

        # 1. Extract text
        document_processor = get_document_processor()
        text = await document_processor.extract_text(content, doc.mime_type)

        if not text.strip():
            raise ValueError("No text content extracted from document")

        logger.debug(
            "Extracted text from document",
            extra={"document_id": document_id, "text_length": len(text)},
        )

        # 2. Chunk text
        kb = doc.knowledge_base
        effective_chunk_size = chunk_size or kb.chunk_size
        effective_chunk_overlap = chunk_overlap or kb.chunk_overlap

        chunking_service = ChunkingService(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
        )
        chunks = chunking_service.chunk_text(text)

        if not chunks:
            raise ValueError("Text chunking produced no chunks")

        logger.debug(
            "Chunked document",
            extra={
                "document_id": document_id,
                "chunk_count": len(chunks),
                "chunk_size": effective_chunk_size,
            },
        )

        # 3. Embed chunks
        embedding_service = get_embedding_service()
        embeddings = await embedding_service.embed_texts(chunks)

        logger.debug(
            "Generated embeddings",
            extra={"document_id": document_id, "embedding_count": len(embeddings)},
        )

        # 4. Store in vector store
        vector_store = get_vector_store()
        inserted_count = await vector_store.insert_chunks(
            document_id=doc.id,
            kb_id=kb.id,
            chunks=chunks,
            embeddings=embeddings,
        )

        # 5. Update document status
        doc.chunk_count = inserted_count
        doc.processing_status = "completed"
        doc.processing_error = None
        await doc.save(update_fields=["chunk_count", "processing_status", "processing_error", "updated_at"])

        logger.info(
            "Document processing completed",
            extra={
                "document_id": document_id,
                "chunk_count": inserted_count,
            },
        )

        return {
            "success": True,
            "document_id": document_id,
            "chunk_count": inserted_count,
        }

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Catch-all for document processing errors to update status
        # Update document with error status
        logger.exception("Document processing failed", extra={"document_id": document_id})
        doc.processing_status = "failed"
        doc.processing_error = str(e)[:1000]  # Truncate long errors
        await doc.save(update_fields=["processing_status", "processing_error", "updated_at"])

        return {
            "success": False,
            "document_id": document_id,
            "error": str(e),
        }


__all__ = ["process_document_task"]
