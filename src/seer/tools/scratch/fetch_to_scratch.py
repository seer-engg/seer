"""
Fetch-to-scratch wrapper tool.

Executes any data-fetching tool and stores the result in scratch storage
for later sandbox analysis. This gives the agent explicit control over
when to use scratch storage vs returning data directly in context.

Usage flow:
1. Agent calls fetch_to_scratch(tool_name="oura_get_sleep", arguments={...})
2. This tool executes oura_get_sleep with proper credential resolution
3. Result is stored in S3 scratch storage
4. Agent receives lightweight ScratchDataRef (handle + preview)
5. Agent can later load full data into sandbox via codebox_load_data(handle)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

from seer.logger import get_logger
from seer.tools.base import BaseTool, get_tool

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.scratch.fetch_to_scratch")


class FetchToScratchTool(BaseTool):
    """
    Execute a data-fetching tool and store the result in scratch storage.

    Use this when you plan to analyze large datasets with Python code in a
    sandbox - it keeps the data out of the conversation context. Returns a
    scratch reference with a data preview. Use codebox_load_data to load
    the full data into a sandbox for analysis.
    """

    name = "fetch_to_scratch"
    description = (
        "Execute any data-fetching tool and store the result in scratch storage "
        "for later sandbox analysis. Use this when you plan to analyze data with "
        "Python code in a sandbox - it keeps large datasets out of the conversation.\n\n"
        "Returns a scratch reference with a data preview. Use codebox_load_data "
        "to load the full data into a sandbox for analysis.\n\n"
        "Example: fetch_to_scratch(tool_name='oura_get_sleep', "
        "arguments={'start_date': '2026-02-25', 'end_date': '2026-03-25'})"
    )

    # No OAuth - this tool orchestrates other tools
    provider = None
    integration_type = "scratch"
    required_scopes: list[str] = []

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": (
                        "Name of the tool to execute (e.g., 'oura_get_sleep', "
                        "'postgres_query', 'google_sheets_get_values')"
                    ),
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments to pass to the tool",
                    "additionalProperties": True,
                },
                "preview_rows": {
                    "type": "integer",
                    "description": "Number of rows to include in preview (default: 5)",
                    "default": 5,
                },
            },
            "required": ["tool_name", "arguments"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "_type": {
                    "type": "string",
                    "const": "scratch_data_ref",
                    "description": "Type marker for scratch reference",
                },
                "handle": {
                    "type": "string",
                    "description": "Unique handle for loading data into sandbox",
                },
                "data_type": {
                    "type": "string",
                    "description": "Format of stored data (json or csv)",
                },
                "preview": {
                    "type": "string",
                    "description": "Human-readable preview of the data",
                },
                "row_count": {
                    "type": ["integer", "null"],
                    "description": "Number of records (if applicable)",
                },
                "size_bytes": {
                    "type": "integer",
                    "description": "Size of the stored data in bytes",
                },
                "error": {
                    "type": ["string", "null"],
                    "description": "Error message if execution failed",
                },
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
        """
        Execute target tool and store result in scratch storage.

        Args:
            access_token: Not used (credentials resolved per target tool)
            arguments: Contains tool_name, arguments, and optional preview_rows
            credentials: Not used (credentials resolved per target tool)
            context: Runtime context for user info and scratch store access

        Returns:
            ScratchDataRef dict with handle, preview, and metadata,
            or error dict if execution failed.
        """
        _ = access_token, credentials  # Unused - we resolve per target tool

        tool_name: str = arguments["tool_name"]
        tool_args: Dict[str, Any] = arguments.get("arguments", {})
        preview_rows: int = arguments.get("preview_rows", 5)

        context_error = self._validate_context(context)
        if context_error:
            return {"error": context_error}
        assert context is not None  # _validate_context guarantees this

        target_tool = get_tool(tool_name)
        if not target_tool:
            return {"error": f"Tool '{tool_name}' not found"}

        result = await self._run_target_tool(target_tool, tool_name, tool_args, context)
        if isinstance(result, dict) and "error" in result:
            return result

        return await self._store_result(result, tool_name, preview_rows, context)

    def _validate_context(self, context: Optional["WorkflowRuntimeContext"]) -> Optional[str]:
        """Return an error message string if context is unusable, else None."""
        if not context:
            return "Runtime context not available"
        if not context.has_scratch_store:
            return "Scratch storage not configured"
        return None

    async def _run_target_tool(
        self,
        target_tool: BaseTool,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: "WorkflowRuntimeContext",
    ) -> Dict[str, Any]:
        """Resolve credentials, execute the target tool, and surface any errors."""
        try:
            resolved = await self._resolve_target_credentials(target_tool, tool_args, context)
        except Exception as e:  # pylint: disable=broad-exception-caught  # Credential resolution can fail in many ways; return structured error
            logger.warning("Failed to resolve credentials for %s: %s", tool_name, e)
            return {"error": f"Credential resolution failed: {e}"}

        try:
            result = await target_tool.execute(
                resolved.access_token,
                tool_args,
                credentials=resolved,
                context=context,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught  # Tool execution can fail in many ways; return structured error
            logger.exception("Target tool %s execution failed", tool_name)
            return {"error": f"Tool execution failed: {e}"}

        if isinstance(result, dict) and result.get("error"):
            return {"error": f"Tool returned error: {result['error']}"}
        return result

    async def _store_result(
        self,
        result: Any,
        tool_name: str,
        preview_rows: int,
        context: "WorkflowRuntimeContext",
    ) -> Dict[str, Any]:
        """Store the tool result in scratch and return a ScratchDataRef dict."""
        try:
            handle = f"{tool_name}_{uuid4().hex[:8]}"
            user_id = str(context.user.user_id) if context.user else "anonymous"
            run_id = context.workflow_run_id or "unknown_run"

            ref = await context.scratch_store.store(
                user_id=user_id,
                run_id=run_id,
                handle=handle,
                data=result,
                data_type="json",
                preview_rows=preview_rows,
            )
            logger.info("Stored tool result in scratch: tool=%s handle=%s size=%d", tool_name, handle, ref.size_bytes)
            return ref.to_dict()
        except Exception as e:  # pylint: disable=broad-exception-caught  # Storage can fail in various ways; return structured error
            logger.exception("Failed to store result in scratch")
            return {"error": f"Scratch storage failed: {e}"}

    async def _resolve_target_credentials(
        self,
        target_tool: BaseTool,
        tool_args: Dict[str, Any],
        context: "WorkflowRuntimeContext",
    ) -> "ResolvedCredentials":
        """
        Resolve credentials for the target tool.

        Args:
            target_tool: The tool to resolve credentials for
            tool_args: Arguments that may contain connection_id or resource hints
            context: Runtime context with user and organization info

        Returns:
            ResolvedCredentials with access_token, secrets, etc.
        """
        # pylint: disable=import-outside-toplevel  # Avoid circular imports
        from seer.tools.credential_resolver import CredentialResolver

        # Extract connection_id from args if provided
        connection_id = tool_args.pop("connection_id", None)

        resolver = CredentialResolver(
            user=context.user,
            tool=target_tool,
            connection_id=connection_id,
            organization_id=context.organization_id,
        )

        return await resolver.resolve(tool_args)


__all__ = ["FetchToScratchTool"]
