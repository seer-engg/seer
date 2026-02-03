"""Service layer for Knowledge Base API operations."""
from __future__ import annotations

import base64
import hashlib

from fastapi import HTTPException, UploadFile, status
from tortoise.functions import Count

from seer.api.knowledge.models import (
    AddTextRequest,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdateRequest,
    QueryRequest,
    QueryResponse,
    QueryResultItem,
)
from seer.database import User
from seer.database.knowledge_models import (
    DOC_ID_PREFIX,
    KnowledgeBase,
    KnowledgeDocument,
)
from seer.logger import get_logger
from seer.services.knowledge.chunking_service import ChunkingService
from seer.services.knowledge.document_processor import SUPPORTED_MIME_TYPES
from seer.services.knowledge.embedding_service import get_embedding_service
from seer.services.knowledge.vector_store import get_vector_store
from seer.worker.tasks.knowledge import process_document_task

logger = get_logger("api.knowledge.services")

# Maximum file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024


async def _get_kb_for_user(user: User, kb_id: str) -> KnowledgeBase:
    """Get knowledge base by public ID, ensuring user ownership."""
    try:
        internal_id = KnowledgeBase.parse_public_id(kb_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    kb = await KnowledgeBase.get_or_none(id=internal_id, user=user)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base not found: {kb_id}",
        )
    return kb


async def _get_doc_for_kb(kb: KnowledgeBase, doc_id: str) -> KnowledgeDocument:
    """Get document by public ID, ensuring it belongs to the knowledge base."""
    try:
        internal_id = KnowledgeDocument.parse_public_id(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    doc = await KnowledgeDocument.get_or_none(id=internal_id, knowledge_base=kb)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {doc_id}",
        )
    return doc


def _kb_to_response(kb: KnowledgeBase, document_count: int = 0) -> KnowledgeBaseResponse:
    """Convert KnowledgeBase model to API response."""
    return KnowledgeBaseResponse(
        kb_id=kb.public_id,
        name=kb.name,
        description=kb.description,
        embedding_model=kb.embedding_model,
        chunk_size=kb.chunk_size,
        chunk_overlap=kb.chunk_overlap,
        document_count=document_count,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


def _doc_to_response(doc: KnowledgeDocument, kb_id: str) -> DocumentResponse:
    """Convert KnowledgeDocument model to API response."""
    return DocumentResponse(
        doc_id=doc.public_id,
        kb_id=kb_id,
        name=doc.name,
        mime_type=doc.mime_type,
        file_size=doc.file_size,
        chunk_count=doc.chunk_count,
        processing_status=doc.processing_status,
        processing_error=doc.processing_error,
        metadata=doc.metadata,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


# Knowledge Base CRUD


async def create_knowledge_base(user: User, request: KnowledgeBaseCreateRequest) -> KnowledgeBaseResponse:
    """Create a new knowledge base."""
    kb = await KnowledgeBase.create(
        user=user,
        name=request.name,
        description=request.description,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    logger.info("Created knowledge base", extra={"kb_id": kb.public_id, "user_id": user.id})
    return _kb_to_response(kb, document_count=0)


async def list_knowledge_bases(user: User) -> KnowledgeBaseListResponse:
    """List all knowledge bases for a user."""
    kbs = await KnowledgeBase.filter(user=user).annotate(doc_count=Count("documents")).order_by("-created_at")

    items = [_kb_to_response(kb, document_count=getattr(kb, "doc_count", 0)) for kb in kbs]
    return KnowledgeBaseListResponse(items=items, total=len(items))


async def get_knowledge_base(user: User, kb_id: str) -> KnowledgeBaseResponse:
    """Get a single knowledge base."""
    kb = await _get_kb_for_user(user, kb_id)
    doc_count = await KnowledgeDocument.filter(knowledge_base=kb).count()
    return _kb_to_response(kb, document_count=doc_count)


async def update_knowledge_base(user: User, kb_id: str, request: KnowledgeBaseUpdateRequest) -> KnowledgeBaseResponse:
    """Update a knowledge base."""
    kb = await _get_kb_for_user(user, kb_id)

    if request.name is not None:
        kb.name = request.name
    if request.description is not None:
        kb.description = request.description

    await kb.save()
    doc_count = await KnowledgeDocument.filter(knowledge_base=kb).count()
    logger.info("Updated knowledge base", extra={"kb_id": kb_id})
    return _kb_to_response(kb, document_count=doc_count)


async def delete_knowledge_base(user: User, kb_id: str) -> None:
    """Delete a knowledge base and all its documents/chunks."""
    kb = await _get_kb_for_user(user, kb_id)

    # Chunks will be cascade-deleted via foreign key
    await kb.delete()
    logger.info("Deleted knowledge base", extra={"kb_id": kb_id})


# Document Operations


async def upload_document(user: User, kb_id: str, file: UploadFile) -> DocumentUploadResponse:
    """Upload and process a document."""
    kb = await _get_kb_for_user(user, kb_id)

    # Validate MIME type
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {mime_type}. Supported: {', '.join(SUPPORTED_MIME_TYPES)}",
        )

    # Read and validate file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
        )

    # Compute content hash for deduplication
    content_hash = hashlib.sha256(content).hexdigest()

    # Check for duplicate
    existing_doc = await KnowledgeDocument.get_or_none(knowledge_base=kb, content_hash=content_hash)
    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document with same content already exists: {existing_doc.public_id}",
        )

    # Create document record
    doc = await KnowledgeDocument.create(
        knowledge_base=kb,
        name=file.filename or "Untitled",
        mime_type=mime_type,
        file_size=len(content),
        content_hash=content_hash,
        processing_status="pending",
    )

    # Enqueue background processing task
    content_b64 = base64.b64encode(content).decode("ascii")
    await process_document_task.kiq(
        document_id=doc.id,
        content_b64=content_b64,
        chunk_size=kb.chunk_size,
        chunk_overlap=kb.chunk_overlap,
    )

    logger.info(
        "Document uploaded and queued for processing",
        extra={"doc_id": doc.public_id, "kb_id": kb_id, "file_size": len(content)},
    )

    return DocumentUploadResponse(
        doc_id=doc.public_id,
        kb_id=kb.public_id,
        name=doc.name,
        processing_status=doc.processing_status,
        message="Document uploaded and queued for processing",
    )


async def list_documents(user: User, kb_id: str) -> DocumentListResponse:
    """List all documents in a knowledge base."""
    kb = await _get_kb_for_user(user, kb_id)
    docs = await KnowledgeDocument.filter(knowledge_base=kb).order_by("-created_at")

    items = [_doc_to_response(doc, kb.public_id) for doc in docs]
    return DocumentListResponse(items=items, total=len(items))


async def get_document(user: User, kb_id: str, doc_id: str) -> DocumentResponse:
    """Get a single document."""
    kb = await _get_kb_for_user(user, kb_id)
    doc = await _get_doc_for_kb(kb, doc_id)
    return _doc_to_response(doc, kb.public_id)


async def delete_document(user: User, kb_id: str, doc_id: str) -> None:
    """Delete a document and its chunks."""
    kb = await _get_kb_for_user(user, kb_id)
    doc = await _get_doc_for_kb(kb, doc_id)

    # Chunks will be cascade-deleted via foreign key
    await doc.delete()
    logger.info("Deleted document", extra={"doc_id": doc_id, "kb_id": kb_id})


# Query Operations


async def query_knowledge_base(user: User, kb_id: str, request: QueryRequest) -> QueryResponse:
    """Perform semantic search on a knowledge base."""
    kb = await _get_kb_for_user(user, kb_id)

    # Generate embedding for query
    embedding_service = get_embedding_service()
    query_embedding = await embedding_service.embed_query(request.query)

    # Search vector store
    vector_store = get_vector_store()
    results = await vector_store.similarity_search(
        kb_id=kb.id,
        embedding=query_embedding,
        top_k=request.top_k,
        min_score=request.min_score,
    )

    logger.debug(
        "Query completed",
        extra={"kb_id": kb_id, "query_length": len(request.query), "result_count": len(results)},
    )

    return QueryResponse(
        results=[
            QueryResultItem(
                chunk_id=r["chunk_id"],
                doc_id=f"{DOC_ID_PREFIX}{r['doc_id']}",
                doc_name=r["doc_name"],
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in results
        ],
        query=request.query,
        kb_id=kb.public_id,
    )


async def add_text_to_knowledge_base(user: User, kb_id: str, request: AddTextRequest) -> DocumentUploadResponse:
    """Add text content directly to a knowledge base (sync processing)."""
    kb = await _get_kb_for_user(user, kb_id)

    # Compute content hash
    content_bytes = request.content.encode("utf-8")
    content_hash = hashlib.sha256(content_bytes).hexdigest()

    # Check for duplicate
    existing_doc = await KnowledgeDocument.get_or_none(knowledge_base=kb, content_hash=content_hash)
    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document with same content already exists: {existing_doc.public_id}",
        )

    # Create document record
    doc = await KnowledgeDocument.create(
        knowledge_base=kb,
        name=request.name,
        mime_type="text/plain",
        file_size=len(content_bytes),
        content_hash=content_hash,
        processing_status="processing",
        metadata=request.metadata,
    )

    try:
        # Process synchronously for text content (it's fast)
        chunking_service = ChunkingService(
            chunk_size=kb.chunk_size,
            chunk_overlap=kb.chunk_overlap,
        )
        chunks = chunking_service.chunk_text(request.content)

        if not chunks:
            raise ValueError("Text chunking produced no chunks")

        # Generate embeddings
        embedding_service = get_embedding_service()
        embeddings = await embedding_service.embed_texts(chunks)

        # Store in vector store
        vector_store = get_vector_store()
        await vector_store.insert_chunks(
            document_id=doc.id,
            kb_id=kb.id,
            chunks=chunks,
            embeddings=embeddings,
        )

        # Update document status
        doc.chunk_count = len(chunks)
        doc.processing_status = "completed"
        await doc.save()

        logger.info(
            "Text added to knowledge base",
            extra={"doc_id": doc.public_id, "kb_id": kb_id, "chunk_count": len(chunks)},
        )

        return DocumentUploadResponse(
            doc_id=doc.public_id,
            kb_id=kb.public_id,
            name=doc.name,
            processing_status="completed",
            message=f"Text added successfully with {len(chunks)} chunks",
        )

    except Exception as e:
        # Update document with error
        doc.processing_status = "failed"
        doc.processing_error = str(e)[:1000]
        await doc.save()
        logger.exception("Failed to add text", extra={"kb_id": kb_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process text: {str(e)}",
        ) from e


__all__ = [
    "create_knowledge_base",
    "list_knowledge_bases",
    "get_knowledge_base",
    "update_knowledge_base",
    "delete_knowledge_base",
    "upload_document",
    "list_documents",
    "get_document",
    "delete_document",
    "query_knowledge_base",
    "add_text_to_knowledge_base",
]
