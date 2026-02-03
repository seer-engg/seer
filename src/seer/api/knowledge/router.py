"""API router for Knowledge Base endpoints."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from seer.api.knowledge import models as api_models
from seer.api.knowledge import services
from seer.database import User

router = APIRouter(prefix="/v1/knowledge-bases", tags=["knowledge"])


def _require_user(request: Request) -> User:
    """Get authenticated user from request state."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


# Knowledge Base CRUD


@router.post("", response_model=api_models.KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(request: Request, payload: api_models.KnowledgeBaseCreateRequest):
    """Create a new knowledge base."""
    user = _require_user(request)
    return await services.create_knowledge_base(user, payload)


@router.get("", response_model=api_models.KnowledgeBaseListResponse)
async def list_knowledge_bases(request: Request):
    """List all knowledge bases for the current user."""
    user = _require_user(request)
    return await services.list_knowledge_bases(user)


@router.get("/{kb_id}", response_model=api_models.KnowledgeBaseResponse)
async def get_knowledge_base(request: Request, kb_id: str):
    """Get a single knowledge base by ID."""
    user = _require_user(request)
    return await services.get_knowledge_base(user, kb_id)


@router.put("/{kb_id}", response_model=api_models.KnowledgeBaseResponse)
async def update_knowledge_base(request: Request, kb_id: str, payload: api_models.KnowledgeBaseUpdateRequest):
    """Update a knowledge base."""
    user = _require_user(request)
    return await services.update_knowledge_base(user, kb_id, payload)


@router.delete("/{kb_id}", status_code=status.HTTP_200_OK)
async def delete_knowledge_base(request: Request, kb_id: str):
    """Delete a knowledge base and all its documents."""
    user = _require_user(request)
    await services.delete_knowledge_base(user, kb_id)
    return {"ok": True}


# Document Operations


@router.post("/{kb_id}/documents", response_model=api_models.DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, kb_id: str, file: UploadFile = File(...)):
    """Upload a document to a knowledge base.

    Supported file types: PDF, TXT, DOCX
    Maximum file size: 10MB

    The document will be processed asynchronously. Use the GET endpoint to check processing status.
    """
    user = _require_user(request)
    return await services.upload_document(user, kb_id, file)


@router.get("/{kb_id}/documents", response_model=api_models.DocumentListResponse)
async def list_documents(request: Request, kb_id: str):
    """List all documents in a knowledge base."""
    user = _require_user(request)
    return await services.list_documents(user, kb_id)


@router.get("/{kb_id}/documents/{doc_id}", response_model=api_models.DocumentResponse)
async def get_document(request: Request, kb_id: str, doc_id: str):
    """Get a single document by ID."""
    user = _require_user(request)
    return await services.get_document(user, kb_id, doc_id)


@router.delete("/{kb_id}/documents/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(request: Request, kb_id: str, doc_id: str):
    """Delete a document and its chunks."""
    user = _require_user(request)
    await services.delete_document(user, kb_id, doc_id)
    return {"ok": True}


# Query Operations


@router.post("/{kb_id}/query", response_model=api_models.QueryResponse)
async def query_knowledge_base(request: Request, kb_id: str, payload: api_models.QueryRequest):
    """Query a knowledge base using semantic search.

    Returns the most relevant chunks based on cosine similarity to the query.
    """
    user = _require_user(request)
    return await services.query_knowledge_base(user, kb_id, payload)


@router.post("/{kb_id}/text", response_model=api_models.DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def add_text(request: Request, kb_id: str, payload: api_models.AddTextRequest):
    """Add text content directly to a knowledge base.

    The text will be chunked, embedded, and stored synchronously.
    Use this for programmatic text addition (e.g., from workflow tools).
    """
    user = _require_user(request)
    return await services.add_text_to_knowledge_base(user, kb_id, payload)


__all__ = ["router"]
