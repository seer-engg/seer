"""Workflow memory update tool."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.tools.base import BaseTool
from seer.tools.memory.base import require_memory_access

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials


class MemoryUpdateTool(BaseTool):
    """Update an existing memory in a bank."""

    name = "memory_update"
    description = "Update an existing memory in a specific memory bank."
    integration_type = "memory"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_bank_id": {"type": "string", "description": "Target memory bank ID (format: mb_*)"},
                "memory_id": {"type": "string", "description": "Memory ID to update"},
                "content": {"type": "string", "description": "Replacement content"},
                "metadata": {"type": "object", "additionalProperties": True, "description": "Optional metadata payload"},
            },
            "required": ["memory_bank_id", "memory_id", "content"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "object", "additionalProperties": True}

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
        content = str(arguments["content"]).strip()
        if not content:
            raise HTTPException(status_code=400, detail="Memory content cannot be empty")
        return await memory_access.update(
            arguments["memory_bank_id"],
            arguments["memory_id"],
            content,
            metadata=arguments.get("metadata"),
        )


__all__ = ["MemoryUpdateTool"]
