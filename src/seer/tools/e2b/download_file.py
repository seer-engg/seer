"""
E2B download file tool.

Reads a file from a sandbox and returns it as a base64 data URI,
enabling agents to embed sandbox-generated images in HTML artifacts.
"""
from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.e2b.base import E2BToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.e2b.download_file")

# File extension → MIME type mapping for common sandbox outputs
_EXT_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
    ".html": "text/html",
    ".csv": "text/csv",
    ".json": "application/json",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}

_DEFAULT_MAX_SIZE = 5 * 1024 * 1024  # 5 MB


class E2BDownloadFileTool(E2BToolBase):
    """Download a file from a sandbox as a base64 data URI."""

    name = "codebox_download_file"
    description = (
        "Download a file from a sandbox and return it as a base64 data URI. "
        "Use this to extract images, charts, or other files generated in a sandbox.\n\n"
        "The returned data_uri can be embedded directly in HTML for create_artifact:\n"
        '  <img src="{data_uri}">\n\n'
        "Example: codebox_download_file(sandbox_id=..., path='/mnt/data/chart.png')"
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sandbox_id": {
                    "type": "string",
                    "description": "Sandbox ID from codebox_create_sandbox",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file in the sandbox (e.g., '/mnt/data/chart.png')",
                },
                "max_size_bytes": {
                    "type": "integer",
                    "description": "Maximum file size in bytes (default: 5MB). Prevents context bloat.",
                    "default": _DEFAULT_MAX_SIZE,
                },
            },
            "required": ["sandbox_id", "path"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether download succeeded"},
                "data_uri": {
                    "type": "string",
                    "description": "Base64 data URI (e.g., 'data:image/png;base64,...') for embedding in HTML",
                },
                "mime_type": {"type": "string", "description": "Detected MIME type of the file"},
                "size_bytes": {"type": "integer", "description": "File size in bytes"},
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
        path: str = arguments["path"]
        max_size: int = arguments.get("max_size_bytes", _DEFAULT_MAX_SIZE)

        try:
            sandbox = await self._connect_sandbox(sandbox_id)

            # Read file as raw bytes — format="bytes" avoids lossy UTF-8 text decoding
            # that corrupts binary files (PNG, PDF) with replacement chars
            raw = await sandbox.files.read(path, format="bytes")
            data: bytes = bytes(raw)  # bytearray → bytes

            if len(data) > max_size:
                return {
                    "success": False,
                    "data_uri": "",
                    "mime_type": "",
                    "size_bytes": len(data),
                    "error": f"File size {len(data)} bytes exceeds max_size_bytes {max_size}",
                }

            # Infer MIME type from extension
            mime_type = "application/octet-stream"
            for ext, mt in _EXT_MIME.items():
                if path.lower().endswith(ext):
                    mime_type = mt
                    break

            # Encode as data URI
            b64 = base64.b64encode(data).decode("ascii")
            data_uri = f"data:{mime_type};base64,{b64}"

            logger.debug("Downloaded file from sandbox: path=%s size=%d mime=%s", path, len(data), mime_type)

            return {
                "success": True,
                "data_uri": data_uri,
                "mime_type": mime_type,
                "size_bytes": len(data),
                "error": None,
            }

        except Exception as e:  # pylint: disable=broad-exception-caught  # E2B SDK raises undocumented exception types; return structured error
            logger.exception("Failed to download file from sandbox: path=%s", path)
            return {
                "success": False,
                "data_uri": "",
                "mime_type": "",
                "size_bytes": 0,
                "error": str(e),
            }


__all__ = ["E2BDownloadFileTool"]
