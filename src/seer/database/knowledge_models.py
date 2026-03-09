"""Database models for knowledge base feature."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict
from tortoise import fields, models
from tortoise.fields import CASCADE


# Public ID prefixes
KB_ID_PREFIX = "kb_"
DOC_ID_PREFIX = "doc_"


class KnowledgeBase(models.Model):
    """Knowledge base entity for storing document collections and their embeddings."""

    id = fields.IntField(primary_key=True)
    user = fields.ForeignKeyField("models.User", related_name="knowledge_bases", on_delete=CASCADE)
    user_id: int  # Tortoise ORM FK shadow attribute
    organization = fields.ForeignKeyField(
        "models.Organization",
        related_name="knowledge_bases",
        on_delete=CASCADE,
        null=True,
        description="Organization this KB belongs to (for team access)",
    )
    organization_id: Optional[int]  # Tortoise ORM FK shadow attribute
    name = fields.CharField(max_length=255)
    description = fields.TextField(null=True)
    embedding_model = fields.CharField(max_length=100, default="text-embedding-3-small")
    embedding_dims = fields.IntField(default=1536)
    chunk_size = fields.IntField(default=1000)
    chunk_overlap = fields.IntField(default=200)
    metadata = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "knowledge_bases"

    def __str__(self) -> str:
        return f"KnowledgeBase<{self.id}:{self.name}>"

    @property
    def public_id(self) -> str:
        """Return public ID with prefix."""
        return f"{KB_ID_PREFIX}{self.id}"

    @classmethod
    def parse_public_id(cls, public_id: str) -> int:
        """Parse public ID to internal ID."""
        if not public_id.startswith(KB_ID_PREFIX):
            raise ValueError(f"Invalid knowledge base ID format: {public_id}")
        return int(public_id[len(KB_ID_PREFIX):])


class KnowledgeDocument(models.Model):
    """Document within a knowledge base."""

    id = fields.IntField(primary_key=True)
    knowledge_base = fields.ForeignKeyField("models.KnowledgeBase", related_name="documents", on_delete=CASCADE)
    name = fields.CharField(max_length=255)
    mime_type = fields.CharField(max_length=100)
    file_size = fields.IntField()
    content_hash = fields.CharField(max_length=64)  # SHA-256 for deduplication
    chunk_count = fields.IntField(default=0)
    processing_status = fields.CharField(max_length=20, default="pending")  # pending/processing/completed/failed
    processing_error = fields.TextField(null=True)
    metadata = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "knowledge_documents"
        unique_together = (("knowledge_base", "content_hash"),)

    def __str__(self) -> str:
        return f"KnowledgeDocument<{self.id}:{self.name}>"

    @property
    def public_id(self) -> str:
        """Return public ID with prefix."""
        return f"{DOC_ID_PREFIX}{self.id}"

    @classmethod
    def parse_public_id(cls, public_id: str) -> int:
        """Parse public ID to internal ID."""
        if not public_id.startswith(DOC_ID_PREFIX):
            raise ValueError(f"Invalid document ID format: {public_id}")
        return int(public_id[len(DOC_ID_PREFIX):])


class KnowledgeChunk(models.Model):
    """Text chunk with embedding for semantic search."""

    id = fields.IntField(primary_key=True)
    document = fields.ForeignKeyField("models.KnowledgeDocument", related_name="chunks", on_delete=CASCADE)
    knowledge_base = fields.ForeignKeyField("models.KnowledgeBase", related_name="chunks", on_delete=CASCADE)
    chunk_index = fields.IntField()
    content = fields.TextField()
    # embedding column added via migration (pgvector) - not managed by Tortoise ORM
    metadata = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "knowledge_chunks"

    def __str__(self) -> str:
        # pylint: disable=no-member  # Reason: document_id is dynamically created by Tortoise ORM ForeignKeyField
        return f"KnowledgeChunk<{self.id}:doc={self.document_id}:idx={self.chunk_index}>"


# Pydantic models for API responses


class KnowledgeBasePublic(BaseModel):
    """Pydantic model for KnowledgeBase API responses."""

    model_config = ConfigDict(from_attributes=True)

    kb_id: str
    name: str
    description: Optional[str] = None
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    document_count: int = 0
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_orm_with_count(cls, kb: KnowledgeBase, document_count: int = 0) -> "KnowledgeBasePublic":
        """Create from ORM model with document count."""
        return cls(
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


__all__ = [
    "KB_ID_PREFIX",
    "DOC_ID_PREFIX",
    "KnowledgeBase",
    "KnowledgeDocument",
    "KnowledgeChunk",
    "KnowledgeBasePublic",
]
