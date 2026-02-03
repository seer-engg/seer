"""Knowledge base list tool."""
from __future__ import annotations

from typing import Any, Dict, Optional

from tortoise.functions import Count

from seer.database.knowledge_models import KnowledgeBase
from seer.logger import get_logger
from seer.tools.base import BaseTool

logger = get_logger("tools.knowledge.list")


class KnowledgeBaseListTool(BaseTool):
    """List available knowledge bases."""

    name = "kb_list"
    description = "List all knowledge bases available to the current user. Use this to find the kb_id needed for kb_query and kb_add_text tools."
    integration_type = "knowledge"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "knowledge_bases": {
                    "type": "array",
                    "description": "List of available knowledge bases",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kb_id": {"type": "string", "description": "Knowledge base ID (use this in other kb_* tools)"},
                            "name": {"type": "string", "description": "Knowledge base name"},
                            "description": {"type": "string", "description": "Knowledge base description"},
                            "document_count": {"type": "integer", "description": "Number of documents in the knowledge base"},
                        },
                    },
                },
                "total": {"type": "integer", "description": "Total number of knowledge bases"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        _credentials: Optional[Any] = None,
    ) -> Any:
        # Note: In workflow context, we need to get user_id from the credentials or workflow context
        # For now, list all KBs (access control should be handled at workflow level)
        # In production, this would filter by user from workflow context

        kbs = await KnowledgeBase.all().annotate(doc_count=Count("documents")).order_by("-created_at")

        result = {
            "knowledge_bases": [
                {
                    "kb_id": kb.public_id,
                    "name": kb.name,
                    "description": kb.description,
                    "document_count": getattr(kb, "doc_count", 0),
                }
                for kb in kbs
            ],
            "total": len(kbs),
        }

        logger.debug("Listed knowledge bases", extra={"count": len(kbs)})
        return result


__all__ = ["KnowledgeBaseListTool"]
