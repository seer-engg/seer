"""
E2B write data tool.

Writes data from the agent's context directly into a sandbox filesystem,
enabling agents to push query results or other data to a sandbox without
routing through the scratch store.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.e2b.base import E2BToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.e2b.write_data")


class E2BWriteDataTool(E2BToolBase):
    """Write data from context directly into a sandbox filesystem."""

    name = "codebox_write_data"
    description = (
        "Write data directly into a sandbox filesystem for analysis. "
        "Use when you already have data in context (e.g., from a postgres_execute_sql "
        "or web_search result) and want to analyze it with Python code.\n\n"
        "For large datasets from external tools, prefer the fetch_to_scratch + "
        "codebox_load_data pipeline instead — it avoids passing data through the conversation."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sandbox_id": {
                    "type": "string",
                    "description": "Sandbox ID from codebox_create_sandbox",
                },
                "data": {
                    "type": "string",
                    "description": "Data to write (JSON string, CSV string, or plain text)",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename for the data file (default: 'data.json'). Include extension.",
                    "default": "data.json",
                },
            },
            "required": ["sandbox_id", "data"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether write succeeded"},
                "sandbox_path": {
                    "type": "string",
                    "description": "Path to the data file in the sandbox",
                },
                "size_bytes": {"type": "integer", "description": "Size of the written data"},
                "error": {"type": ["string", "null"], "description": "Error message if failed"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, credentials, context

        sandbox_id: str = arguments["sandbox_id"]
        data: str = arguments["data"]
        filename: str = arguments.get("filename", "data.json")

        try:
            sandbox = await self._connect_sandbox(sandbox_id)

            sandbox_path = f"/home/user/data/{filename}"
            data_bytes = data.encode("utf-8")

            await sandbox.files.write(sandbox_path, data_bytes)

            logger.info("Wrote data to sandbox: path=%s size=%d", sandbox_path, len(data_bytes))

            return {
                "success": True,
                "sandbox_path": sandbox_path,
                "size_bytes": len(data_bytes),
                "error": None,
            }

        except Exception as e:  # pylint: disable=broad-exception-caught  # E2B SDK raises undocumented exception types; return structured error
            logger.exception("Failed to write data to sandbox")
            return {
                "success": False,
                "sandbox_path": "",
                "size_bytes": 0,
                "error": str(e),
            }


__all__ = ["E2BWriteDataTool"]
