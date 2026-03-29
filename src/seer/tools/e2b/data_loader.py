"""
E2B data loader tool.

Loads scratch data into sandbox filesystem for analysis.
This enables tools to pass large datasets to sandboxes without
routing through the LLM context window.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.e2b.base import E2BToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.e2b.data_loader")


class E2BLoadDataTool(E2BToolBase):
    """Load scratch data into an E2B sandbox filesystem."""

    name = "codebox_load_data"
    description = (
        "Load data into sandbox filesystem for analysis. "
        "Use with data handles returned by fetch_to_scratch. "
        "Pair with fetch_to_scratch to move large query results directly into "
        "the sandbox without passing through the conversation.\n\n"
        "Example flow: fetch_to_scratch(tool_name='postgres_execute_sql', ...) → "
        "codebox_load_data(sandbox_id=..., handle=<returned handle>)"
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sandbox_id": {
                    "type": "string",
                    "description": "Sandbox ID from codebox_create_sandbox",
                },
                "handle": {
                    "type": "string",
                    "description": "Data handle from a tool's scratch_data_ref output",
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Optional filename for the data file (default: derived from handle). "
                        "Must end with .json or .csv to match the data format."
                    ),
                },
            },
            "required": ["sandbox_id", "handle"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether load succeeded"},
                "sandbox_path": {
                    "type": "string",
                    "description": "Path to the data file in the sandbox",
                },
                "size_bytes": {"type": "integer", "description": "Size of the loaded data"},
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
        _ = access_token, credentials

        sandbox_id: str = arguments["sandbox_id"]
        handle: str = arguments["handle"]
        filename: Optional[str] = arguments.get("filename")

        if not context:
            return {
                "success": False,
                "sandbox_path": "",
                "size_bytes": 0,
                "error": "Runtime context not available",
            }

        if not context.has_scratch_store:
            return {
                "success": False,
                "sandbox_path": "",
                "size_bytes": 0,
                "error": "Scratch store not configured",
            }

        try:
            # Get data from scratch store
            scratch = context.scratch_store
            user_id = str(context.user.user_id) if context.user else "anonymous"
            run_id = context.workflow_run_id or "unknown_run"

            data = await scratch.retrieve(handle, user_id, run_id)

            # Determine filename from data content heuristics
            if not filename:
                if isinstance(data, bytes) and (data.startswith(b"{") or data.startswith(b"[")):
                    filename = f"{handle}.json"
                elif isinstance(data, bytes) and (b"," in data[:1024] and b"\n" in data[:1024]):
                    filename = f"{handle}.csv"
                else:
                    filename = f"{handle}.dat"

            # Connect to sandbox and upload file
            sandbox = await self._connect_sandbox(sandbox_id)

            # Write to sandbox filesystem
            sandbox_path = f"/home/user/data/{filename}"

            # E2B's files.write expects the path and content
            await sandbox.files.write(sandbox_path, data)

            logger.info(
                "Loaded data to sandbox: handle=%s path=%s size=%d",
                handle, sandbox_path, len(data)
            )

            return {
                "success": True,
                "sandbox_path": sandbox_path,
                "size_bytes": len(data),
                "error": None,
            }

        except FileNotFoundError as e:
            logger.warning("Scratch data not found: handle=%s error=%s", handle, e)
            return {
                "success": False,
                "sandbox_path": "",
                "size_bytes": 0,
                "error": f"Data not found for handle: {handle}",
            }
        except Exception as e:  # pylint: disable=broad-exception-caught  # E2B SDK raises undocumented exception types; return structured error
            logger.exception("Failed to load data to sandbox")
            return {
                "success": False,
                "sandbox_path": "",
                "size_bytes": 0,
                "error": str(e),
            }


__all__ = ["E2BLoadDataTool"]
