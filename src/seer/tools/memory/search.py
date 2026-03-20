"""Workflow memory search tool."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.tools.base import BaseTool
from seer.tools.memory.base import require_memory_access

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials


class MemorySearchTool(BaseTool):
    """Search a specific memory bank."""

    name = "memory_search"
    description = "Search a memory bank using semantic retrieval."
    integration_type = "memory"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_bank_id": {"type": "string", "description": "Target memory bank ID (format: mb_*)"},
                "query": {"type": "string", "description": "Semantic search query"},
                "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
            },
            "required": ["memory_bank_id", "query"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "array", "items": {"type": "object", "additionalProperties": True}}

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        _ = access_token, credentials
        memory_access = require_memory_access(context)
        query = str(arguments["query"]).strip()
        if not query:
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        return await memory_access.search(arguments["memory_bank_id"], query=query, limit=int(arguments.get("limit", 5)))


__all__ = ["MemorySearchTool"]
