"""Knowledge base query tool for semantic search."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.database.knowledge_models import DOC_ID_PREFIX, KnowledgeBase
from seer.logger import get_logger
from seer.services.knowledge.embedding_service import get_embedding_service
from seer.services.knowledge.vector_store import get_vector_store
from seer.tools.base import BaseTool, ResourcePickerConfig
from seer.tools.knowledge.common import KNOWLEDGE_BASE_PICKER

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.knowledge.query")


class KnowledgeBaseQueryTool(BaseTool):
    """Query a knowledge base to find relevant documents using semantic search."""

    name = "kb_query"
    description = "Query a knowledge base to find relevant documents using semantic search. Returns text chunks ranked by relevance to the query."
    integration_type = "knowledge"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kb_id": {
                    "type": "string",
                    "description": "Knowledge base ID (format: kb_*). Use kb_list tool to see available knowledge bases.",
                },
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and descriptive for better results.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum similarity score threshold (0-1). Higher values return more relevant results.",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["kb_id", "query"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "description": "Search results ranked by relevance",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Text content of the chunk"},
                            "doc_name": {"type": "string", "description": "Name of the source document"},
                            "doc_id": {"type": "string", "description": "Document ID"},
                            "score": {"type": "number", "description": "Similarity score (0-1)"},
                            "metadata": {"type": ["object", "null"], "description": "Optional metadata"},
                        },
                    },
                },
                "query": {"type": "string", "description": "The original query"},
                "kb_id": {"type": "string", "description": "Knowledge base ID"},
            },
        }

    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        return {"kb_id": dict(KNOWLEDGE_BASE_PICKER)}

    async def execute(
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
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        min_score = arguments.get("min_score", 0.3)

        # Parse KB ID
        try:
            internal_id = KnowledgeBase.parse_public_id(kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # Verify KB exists (user auth is handled at workflow level)
        kb = await KnowledgeBase.get_or_none(id=internal_id)
        if not kb:
            raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_id}")

        # Generate query embedding
        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.embed_query(query)

        # Search vector store
        vector_store = get_vector_store()
        results = await vector_store.similarity_search(
            kb_id=internal_id,
            embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        logger.debug(
            "Knowledge base query executed",
            extra={"kb_id": kb_id, "query_length": len(query), "result_count": len(results)},
        )

        return {
            "results": [
                {
                    "content": r["content"],
                    "doc_name": r["doc_name"],
                    "doc_id": f"{DOC_ID_PREFIX}{r['doc_id']}",
                    "score": round(r["score"], 4),
                    "metadata": r["metadata"],
                }
                for r in results
            ],
            "query": query,
            "kb_id": kb_id,
        }


__all__ = ["KnowledgeBaseQueryTool"]
