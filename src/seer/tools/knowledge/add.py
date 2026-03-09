"""Knowledge base add text tool."""
# pylint: disable=duplicate-code  # Reason: Similar error handling and config patterns across platform adapters (Discord/Slack) is intentional
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.database import OrganizationMembership
from seer.database.knowledge_models import KnowledgeBase, KnowledgeDocument
from seer.database.organization_models import OrganizationRole
from seer.logger import get_logger
from seer.services.knowledge.chunking_service import ChunkingService
from seer.services.knowledge.embedding_service import get_embedding_service
from seer.services.knowledge.vector_store import get_vector_store
from seer.tools.base import BaseTool, ResourcePickerConfig
from seer.tools.knowledge.common import KNOWLEDGE_BASE_PICKER

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.knowledge.add")


class KnowledgeBaseAddTextTool(BaseTool):
    """Add text content to a knowledge base."""

    name = "kb_add_text"
    description = "Add text content to a knowledge base. The text will be automatically chunked and embedded for semantic search."
    integration_type = "knowledge"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kb_id": {
                    "type": "string",
                    "description": "Knowledge base ID (format: kb_*). Use kb_list tool to see available knowledge bases.",
                },
                "content": {
                    "type": "string",
                    "description": "The text content to add to the knowledge base.",
                },
                "name": {
                    "type": "string",
                    "description": "A name/title for this content (e.g., 'Meeting Notes 2024-01-15').",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to attach to the document (e.g., source, date, tags).",
                    "additionalProperties": True,
                },
            },
            "required": ["kb_id", "content", "name"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether the operation succeeded"},
                "doc_id": {"type": "string", "description": "ID of the created document"},
                "kb_id": {"type": "string", "description": "Knowledge base ID"},
                "name": {"type": "string", "description": "Document name"},
                "chunk_count": {"type": "integer", "description": "Number of chunks created"},
                "message": {"type": "string", "description": "Status message"},
            },
        }

    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        return {"kb_id": dict(KNOWLEDGE_BASE_PICKER)}

    async def _validate_kb_access(self, kb: KnowledgeBase, kb_id: str, context: Optional["WorkflowRuntimeContext"]) -> None:
        """Verify the requesting user has access to the knowledge base. Raises HTTPException if not."""
        if not context or not context.user:
            raise HTTPException(status_code=401, detail="User context required")

        has_access = kb.user_id == context.user.id  # pylint: disable=no-member  # Reason: Tortoise FK _id attribute generated dynamically
        if not has_access and kb.organization_id:  # pylint: disable=no-member  # Reason: Tortoise FK _id attribute generated dynamically
            membership = await OrganizationMembership.get_or_none(
                organization_id=kb.organization_id, user=context.user  # pylint: disable=no-member  # Reason: Tortoise FK _id attribute generated dynamically
            )
            if membership and membership.role in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
                has_access = True

        if not has_access:
            raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_id}")

    async def _process_and_store(  # pylint: disable=too-many-positional-arguments  # Reason: all params are distinct required inputs for chunking/storage pipeline
        self, doc: KnowledgeDocument, content: str, kb: KnowledgeBase, kb_id: str, internal_id: int
    ) -> Dict[str, Any]:
        """Chunk, embed, and store text content. Updates doc status. Returns result dict."""
        try:
            chunking_service = ChunkingService(chunk_size=kb.chunk_size, chunk_overlap=kb.chunk_overlap)
            chunks = chunking_service.chunk_text(content)

            if not chunks:
                raise ValueError("Text chunking produced no chunks")

            embedding_service = get_embedding_service()
            embeddings = await embedding_service.embed_texts(chunks)

            vector_store = get_vector_store()
            await vector_store.insert_chunks(document_id=doc.id, kb_id=internal_id, chunks=chunks, embeddings=embeddings)

            doc.chunk_count = len(chunks)
            doc.processing_status = "completed"
            await doc.save()

            logger.info("Text added to knowledge base via tool", extra={"doc_id": doc.public_id, "kb_id": kb_id, "chunk_count": len(chunks)})

            return {
                "success": True,
                "doc_id": doc.public_id,
                "kb_id": kb_id,
                "name": doc.name,
                "chunk_count": len(chunks),
                "message": f"Successfully added text with {len(chunks)} chunks",
            }

        except Exception as e:
            doc.processing_status = "failed"
            doc.processing_error = str(e)[:1000]
            await doc.save()
            logger.exception("Failed to add text via tool", extra={"kb_id": kb_id})
            raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}") from e

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        # access_token, credentials unused but required for interface consistency
        _ = access_token, credentials
        kb_id = arguments["kb_id"]
        content = arguments["content"]
        name = arguments["name"]
        metadata = arguments.get("metadata")

        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")

        try:
            internal_id = KnowledgeBase.parse_public_id(kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        kb = await KnowledgeBase.get_or_none(id=internal_id)
        if not kb:
            raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_id}")

        await self._validate_kb_access(kb, kb_id, context)

        content_bytes = content.encode("utf-8")
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        existing_doc = await KnowledgeDocument.get_or_none(knowledge_base_id=internal_id, content_hash=content_hash)
        if existing_doc:
            return {
                "success": False,
                "doc_id": existing_doc.public_id,
                "kb_id": kb_id,
                "name": existing_doc.name,
                "chunk_count": existing_doc.chunk_count,
                "message": f"Document with same content already exists: {existing_doc.public_id}",
            }

        doc = await KnowledgeDocument.create(
            knowledge_base_id=internal_id,
            name=name,
            mime_type="text/plain",
            file_size=len(content_bytes),
            content_hash=content_hash,
            processing_status="processing",
            metadata=metadata,
        )

        return await self._process_and_store(doc, content, kb, kb_id, internal_id)


__all__ = ["KnowledgeBaseAddTextTool"]
