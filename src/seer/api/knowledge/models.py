"""Pydantic models for Knowledge Base API requests and responses."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Request Models


class KnowledgeBaseCreateRequest(BaseModel):
    """Request to create a new knowledge base."""

    name: str = Field(..., min_length=1, max_length=255, description="Knowledge base name")
    description: Optional[str] = Field(None, max_length=2000, description="Optional description")
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Chunk size in characters")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Overlap between chunks")


class KnowledgeBaseUpdateRequest(BaseModel):
    """Request to update a knowledge base."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)


class QueryRequest(BaseModel):
    """Request to query a knowledge base."""

    query: str = Field(..., min_length=1, max_length=10000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum results to return")
    min_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class AddTextRequest(BaseModel):
    """Request to add text content to a knowledge base."""

    content: str = Field(..., min_length=1, description="Text content to add")
    name: str = Field(..., min_length=1, max_length=255, description="Document name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


# Response Models


class KnowledgeBaseResponse(BaseModel):
    """Response for a single knowledge base."""

    kb_id: str
    name: str
    description: Optional[str] = None
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    document_count: int = 0
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseListResponse(BaseModel):
    """Response for listing knowledge bases."""

    items: List[KnowledgeBaseResponse]
    total: int


class DocumentResponse(BaseModel):
    """Response for a single document."""

    doc_id: str
    kb_id: str
    name: str
    mime_type: str
    file_size: int
    chunk_count: int
    processing_status: str
    processing_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    items: List[DocumentResponse]
    total: int


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    doc_id: str
    kb_id: str
    name: str
    processing_status: str
    message: str


class QueryResultItem(BaseModel):
    """Single result from semantic search."""

    chunk_id: int
    doc_id: str
    doc_name: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response from knowledge base query."""

    results: List[QueryResultItem]
    query: str
    kb_id: str


__all__ = [
    "KnowledgeBaseCreateRequest",
    "KnowledgeBaseUpdateRequest",
    "QueryRequest",
    "AddTextRequest",
    "KnowledgeBaseResponse",
    "KnowledgeBaseListResponse",
    "DocumentResponse",
    "DocumentListResponse",
    "DocumentUploadResponse",
    "QueryResultItem",
    "QueryResponse",
]
