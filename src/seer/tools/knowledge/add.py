"""Knowledge base add text tool."""
# pylint: disable=duplicate-code  # Reason: Similar error handling and config patterns across platform adapters (Discord/Slack) is intentional
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.database.knowledge_models import KnowledgeBase, KnowledgeDocument
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

    async def execute(  # pylint: disable=too-many-locals  # Reason: Knowledge base ingestion requires many intermediate variables for chunking/embedding
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        # access_token, credentials, context unused but required for interface consistency
        _ = access_token, credentials, context
        kb_id = arguments["kb_id"]
        content = arguments["content"]
        name = arguments["name"]
        metadata = arguments.get("metadata")

        # Validate content
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")

        # Parse KB ID
        try:
            internal_id = KnowledgeBase.parse_public_id(kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # Get KB
        kb = await KnowledgeBase.get_or_none(id=internal_id)
        if not kb:
            raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_id}")

        # Compute content hash for deduplication
        content_bytes = content.encode("utf-8")
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Check for duplicate
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

        # Create document record
        doc = await KnowledgeDocument.create(
            knowledge_base_id=internal_id,
            name=name,
            mime_type="text/plain",
            file_size=len(content_bytes),
            content_hash=content_hash,
            processing_status="processing",
            metadata=metadata,
        )

        try:
            # Chunk text
            chunking_service = ChunkingService(
                chunk_size=kb.chunk_size,
                chunk_overlap=kb.chunk_overlap,
            )
            chunks = chunking_service.chunk_text(content)

            if not chunks:
                raise ValueError("Text chunking produced no chunks")

            # Generate embeddings
            embedding_service = get_embedding_service()
            embeddings = await embedding_service.embed_texts(chunks)

            # Store in vector store
            vector_store = get_vector_store()
            await vector_store.insert_chunks(
                document_id=doc.id,
                kb_id=internal_id,
                chunks=chunks,
                embeddings=embeddings,
            )

            # Update document status
            doc.chunk_count = len(chunks)
            doc.processing_status = "completed"
            await doc.save()

            logger.info(
                "Text added to knowledge base via tool",
                extra={"doc_id": doc.public_id, "kb_id": kb_id, "chunk_count": len(chunks)},
            )

            return {
                "success": True,
                "doc_id": doc.public_id,
                "kb_id": kb_id,
                "name": name,
                "chunk_count": len(chunks),
                "message": f"Successfully added text with {len(chunks)} chunks",
            }

        except Exception as e:
            # Update document with error
            doc.processing_status = "failed"
            doc.processing_error = str(e)[:1000]
            await doc.save()
            logger.exception("Failed to add text via tool", extra={"kb_id": kb_id})
            raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}") from e


__all__ = ["KnowledgeBaseAddTextTool"]
