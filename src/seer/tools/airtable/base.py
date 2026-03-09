"""
Base class for Airtable API tools.

Provides shared functionality for all Airtable API integrations:
- HTTP client with OAuth token authentication
- Consistent error handling
- Common response parsing
"""
from abc import ABC
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.http_dispatch import dispatch_http

logger = get_logger("shared.tools.airtable.base")

# Airtable API base URL
AIRTABLE_API_BASE = "https://api.airtable.com/v0"


class AirtableAPIClient(BaseTool, ABC):
    """
    Abstract base class for all Airtable API tools.

    Provides centralized HTTP request handling, OAuth token authentication,
    error translation, and consistent behavior.

    Airtable API limits:
    - Max 10 records per create/update/delete request
    - Max 100 records per list request (with pagination)
    - Rate limit: 5 requests per second per base
    """

    provider = "airtable"
    default_timeout: float = 30.0

    def _get_access_token(self, credentials: Optional[ResolvedCredentials] = None) -> str:
        """
        Get Airtable access token from credentials.

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
            detail=f"{self.name} requires an Airtable access token. Please connect your Airtable account."
        )

    def _build_headers(self, access_token: str) -> Dict[str, str]:
        """Build HTTP headers for Airtable API request."""
        return {
            "Authorization": f"Bearer {access_token}",
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
        Make authenticated HTTP request to Airtable API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (e.g., "meta/bases" or "{baseId}/{tableIdOrName}")
            credentials: ResolvedCredentials with access token
            params: Query parameters
            json_body: JSON request body
            timeout: Request timeout in seconds

        Returns:
            Airtable API response dict

        Raises:
            HTTPException: On API errors
        """
        access_token = self._get_access_token(credentials)
        headers = self._build_headers(access_token)
        timeout_value = timeout or self.default_timeout
        url = f"{AIRTABLE_API_BASE}/{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                resp = await dispatch_http(client, method, url, headers, params=params, json_body=json_body)

                if resp.status_code == 204:
                    # No content (success for DELETE)
                    return {"deleted": True}

                resp.raise_for_status()
                return resp.json()

        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"{self.name} request timed out after {timeout_value}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise self._handle_airtable_error(exc) from exc
        except (ValueError, KeyError, TypeError) as exc:
            logger.exception("Unexpected error in %s", self.name)
            raise HTTPException(
                status_code=500,
                detail=f"{self.name} error: {str(exc)}"
            ) from exc

    def _handle_airtable_error(self, exc: httpx.HTTPStatusError) -> HTTPException:
        """
        Translate Airtable API error responses to HTTPExceptions.

        Args:
            exc: The HTTP status error from httpx

        Returns:
            HTTPException with appropriate status code and detail
        """
        status_code = exc.response.status_code
        try:
            error_data = exc.response.json()
            error = error_data.get("error", {})
            error_type = error.get("type", "UNKNOWN_ERROR")
            error_message = error.get("message", "Unknown error occurred")
        except (ValueError, KeyError, TypeError):
            # JSON parsing failed or unexpected structure
            error_type = "PARSE_ERROR"
            error_message = exc.response.text[:500]

        # Map Airtable error types to HTTP status codes
        error_mapping = {
            "AUTHENTICATION_REQUIRED": (401, "Authentication required. Please reconnect your Airtable account."),
            "INVALID_PERMISSIONS": (403, "Insufficient permissions for this operation."),
            "NOT_FOUND": (404, "The requested resource was not found."),
            "INVALID_REQUEST_UNKNOWN": (400, f"Invalid request: {error_message}"),
            "INVALID_REQUEST_DUPLICATE": (409, f"Duplicate record: {error_message}"),
            "MODEL_ID_NOT_FOUND": (404, f"Base or table not found: {error_message}"),
            "INVALID_VALUE_FOR_COLUMN": (400, f"Invalid field value: {error_message}"),
            "UNKNOWN_FIELD_NAME": (400, f"Unknown field name: {error_message}"),
            "RATE_LIMIT_EXCEEDED": (429, "Rate limit exceeded. Please try again later."),
        }

        mapped_status, mapped_message = error_mapping.get(
            error_type,
            (status_code, f"Airtable API error: {error_type} - {error_message}")
        )

        return HTTPException(status_code=mapped_status, detail=f"{self.name}: {mapped_message}")

    def _format_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format Airtable records for output.

        Flattens the record structure to include id and fields at the top level.

        Args:
            records: List of Airtable record objects

        Returns:
            List of formatted record dicts
        """
        return [
            {
                "id": record.get("id"),
                "createdTime": record.get("createdTime"),
                **record.get("fields", {})
            }
            for record in records
        ]
