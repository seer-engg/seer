# pylint: disable=too-complex,try-except-raise
# Reason: _make_request handles multiple HTTP methods and error conditions; re-raises HTTPException to avoid broader catch
"""
Base class for Notion API tools.

Provides shared functionality for all Notion API integrations:
- HTTP client with OAuth token authentication
- Notion-Version header on all requests (required by Notion API)
- Consistent error handling and Notion error code mapping
"""
from abc import ABC
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool
from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.notion.base")

# Notion API base URL and required version header
NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


class NotionAPIClient(BaseTool, ABC):
    """
    Abstract base class for all Notion API tools.

    Provides centralized HTTP request handling, OAuth token authentication,
    Notion-Version header injection, and error translation.

    All Notion API calls require the Notion-Version header to be set.
    """

    provider = "notion"
    requires_oauth = True  # Notion uses dashboard capabilities, not URL scopes
    default_timeout: float = 30.0

    def _get_access_token(self, credentials: Optional[ResolvedCredentials] = None) -> str:
        """
        Get Notion access token from credentials.

        Args:
            credentials: ResolvedCredentials containing the access token

        Returns:
            Access token string

        Raises:
            HTTPException: 401 if no token is available
        """
        if credentials and credentials.access_token:
            return credentials.access_token

        raise HTTPException(
            status_code=401,
            detail=f"{self.name} requires a Notion access token. Please connect your Notion account."
        )

    def _build_headers(self, access_token: str) -> Dict[str, str]:
        """Build HTTP headers for Notion API request."""
        return {
            "Authorization": f"Bearer {access_token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        credentials: Optional[ResolvedCredentials] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make authenticated HTTP request to Notion API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint path (e.g., "pages/abc123" or "search")
            credentials: ResolvedCredentials with access token
            params: Query parameters
            json_body: JSON request body
            timeout: Request timeout in seconds

        Returns:
            Notion API response dict

        Raises:
            HTTPException: On API errors
        """
        access_token = self._get_access_token(credentials)
        headers = self._build_headers(access_token)
        timeout_value = timeout or self.default_timeout
        url = f"{NOTION_API_BASE}/{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                if method.upper() == "GET":
                    resp = await client.get(url, headers=headers, params=params)
                elif method.upper() == "POST":
                    resp = await client.post(url, headers=headers, params=params, json=json_body)
                elif method.upper() == "PATCH":
                    resp = await client.patch(url, headers=headers, params=params, json=json_body)
                elif method.upper() == "DELETE":
                    resp = await client.delete(url, headers=headers, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                resp.raise_for_status()
                return resp.json()

        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"{self.name} request timed out after {timeout_value}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise self._handle_notion_error(exc) from exc
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            logger.exception("Unexpected error in %s", self.name)
            raise HTTPException(
                status_code=500,
                detail=f"{self.name} error: {str(exc)}"
            ) from exc

    def _handle_notion_error(self, exc: httpx.HTTPStatusError) -> HTTPException:
        """
        Translate Notion API error responses to HTTPExceptions.

        Args:
            exc: The HTTP status error from httpx

        Returns:
            HTTPException with appropriate status code and detail
        """
        status_code = exc.response.status_code
        try:
            error_data = exc.response.json()
            error_code = error_data.get("code", "unknown_error")
            error_message = error_data.get("message", "Unknown error occurred")
        except (ValueError, KeyError, TypeError):
            error_code = "parse_error"
            error_message = exc.response.text[:500]

        # Map Notion error codes to HTTP status codes
        error_mapping = {
            "unauthorized": (401, "Authentication required. Please reconnect your Notion account."),
            "restricted_resource": (403, "Insufficient permissions for this operation."),
            "object_not_found": (404, "The requested resource was not found."),
            "validation_error": (400, f"Invalid request: {error_message}"),
            "invalid_json": (400, f"Invalid JSON in request: {error_message}"),
            "invalid_request_url": (400, f"Invalid request URL: {error_message}"),
            "invalid_request": (400, f"Invalid request: {error_message}"),
            "rate_limited": (429, "Rate limit exceeded. Please try again later."),
            "conflict_error": (409, f"Conflict: {error_message}"),
            "internal_server_error": (500, "Notion internal server error. Please try again."),
            "service_unavailable": (503, "Notion service is unavailable. Please try again."),
        }

        mapped_status, mapped_message = error_mapping.get(
            error_code,
            (status_code, f"Notion API error: {error_code} - {error_message}")
        )

        return HTTPException(status_code=mapped_status, detail=f"{self.name}: {mapped_message}")
