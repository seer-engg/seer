"""Generic HTTP request tool for workflow nodes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.http")

MAX_RESPONSE_BYTES = 1_048_576  # 1 MB


class HttpRequestTool(BaseTool):
    """Make HTTP requests to external APIs."""

    name = "http_request"
    description = (
        "Make an HTTP request to any URL. Supports GET, POST, PUT, PATCH, DELETE. "
        "Use this to call external APIs that don't have a dedicated tool."
    )
    integration_type = "http"
    required_scopes: List[str] = []

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "description": "HTTP method.",
                    "default": "GET",
                },
                "url": {
                    "type": "string",
                    "description": "The URL to request.",
                },
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Request headers.",
                    "default": {},
                },
                "query_params": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "URL query parameters.",
                    "default": {},
                },
                "body": {
                    "description": "JSON request body (for POST/PUT/PATCH).",
                },
            },
            "required": ["url"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status_code": {"type": "integer"},
                "body": {
                    "description": "Parsed JSON body (object, array, or primitive), or raw text if not JSON.",
                    "additionalProperties": True,
                },
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
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
        _ = access_token, credentials, context

        url: str = arguments["url"]
        method: str = arguments.get("method", "GET").upper()
        headers: Dict[str, str] = arguments.get("headers") or {}
        query_params: Dict[str, str] = arguments.get("query_params") or {}
        body: Any = arguments.get("body")

        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise HTTPException(status_code=400, detail=f"Unsupported HTTP method: {method}")

        # Validate header values — expression resolver puts dicts with __error__ on failure
        for key, val in list(headers.items()):
            if isinstance(val, dict):
                error_msg = val.get("__error__", f"Header '{key}' value is not a string")
                raise HTTPException(status_code=400, detail=f"Header resolution failed: {error_msg}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                kwargs: Dict[str, Any] = {"headers": headers, "params": query_params}
                if body is not None and method in {"POST", "PUT", "PATCH"}:
                    kwargs["json"] = body

                resp = await client.request(method, url, **kwargs)

        except httpx.TimeoutException as e:
            raise HTTPException(status_code=504, detail=f"Request timed out: {url}") from e
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Request failed: {e}") from e

        # Parse response
        resp_headers = dict(resp.headers)
        try:
            resp_body = resp.json()
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: fallback to text for non-JSON responses
            text = resp.text
            resp_body = text[:MAX_RESPONSE_BYTES] if len(text) > MAX_RESPONSE_BYTES else text

        logger.debug("HTTP %s %s → %d", method, url, resp.status_code)
        return {
            "status_code": resp.status_code,
            "body": resp_body,
            "headers": resp_headers,
        }


__all__ = ["HttpRequestTool"]
