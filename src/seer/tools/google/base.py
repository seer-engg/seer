"""
Base class for Google API tools.

Provides shared functionality for all Google API integrations:
- HTTP client with OAuth token management
- Consistent error handling
- Pagination utilities
- Common response parsing

All Google tools (Gmail, Drive, Sheets, etc.) should inherit from GoogleAPIClient
to eliminate code duplication and ensure consistent behavior.
"""

from abc import ABC
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool

logger = get_logger("shared.tools.google.base")


class GoogleAPIClient(BaseTool, ABC):
    """
    Abstract base class for all Google API tools.

    Provides centralized HTTP request handling, OAuth token validation,
    error translation, and pagination utilities.

    Subclasses only need to implement:
    - Class attributes (name, description, required_scopes, integration_type)
    - execute() method with business logic

    Example:
        class GmailReadTool(GoogleAPIClient):
            name = "gmail_read_emails"
            description = "Read emails from Gmail"
            required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
            integration_type = "gmail"

            async def execute(self, access_token, arguments):
                resp = await self._make_request(
                    "GET",
                    "https://www.googleapis.com/gmail/v1/users/me/messages",
                    access_token,
                    params={"maxResults": 10}
                )
                return resp.json()
    """

    provider = "google"
    default_timeout: float = 30.0

    def _validate_token(self, access_token: Optional[str]) -> str:
        """
        Validate and return OAuth access token.

        Args:
            access_token: OAuth token from connection

        Returns:
            Validated token string

        Raises:
            HTTPException: 401 if token is None or empty
        """
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail=f"{self.name} requires OAuth access token"
            )
        return access_token

    def _build_headers(self, access_token: str) -> Dict[str, str]:
        """
        Build HTTP headers for Google API request.

        Args:
            access_token: OAuth token

        Returns:
            Dict with Authorization and Accept headers
        """
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }

    async def _make_request(
        self,
        method: str,
        url: str,
        access_token: Optional[str],
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """
        Make authenticated HTTP request to Google API.

        Handles:
        - Token validation
        - Header construction
        - Timeout management
        - Error translation to HTTPException

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE, etc.)
            url: Full API endpoint URL
            access_token: OAuth token (will be validated)
            params: Query parameters
            json_body: JSON request body
            content: Raw bytes content (for file uploads)
            timeout: Request timeout in seconds (default: 30.0)

        Returns:
            httpx.Response object (already checked for errors)

        Raises:
            HTTPException: 401 for auth errors, 403 for permissions,
                          404 for not found, 429 for rate limits,
                          504 for timeouts, 500 for other errors

        Example:
            resp = await self._make_request(
                "GET",
                "https://www.googleapis.com/gmail/v1/users/me/messages",
                access_token,
                params={"q": "is:unread"}
            )
            data = resp.json()
        """
        token = self._validate_token(access_token)
        headers = self._build_headers(token)
        timeout_value = timeout or self.default_timeout

        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                resp = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    content=content
                )

                if resp.is_error:
                    raise self._handle_api_error(resp)

                return resp

        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"{self.name} request timed out after {timeout_value}s"
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in %s", self.name)
            raise HTTPException(
                status_code=500,
                detail=f"{self.name} error: {str(exc)}"
            ) from exc

    def _handle_api_error(self, response: httpx.Response) -> HTTPException:
        """
        Translate Google API HTTP error responses to FastAPI HTTPExceptions.

        Provides consistent, user-friendly error messages across all Google tools.

        Args:
            response: httpx.Response with error status code

        Returns:
            HTTPException with appropriate status code and detail message

        Common Status Codes:
            401: Authentication failed (expired/invalid token)
            403: Permission denied (insufficient OAuth scopes)
            404: Resource not found
            429: Rate limit exceeded
            500+: Google API server errors
        """
        body_snippet = response.text[:500] if response.text else ""

        if response.status_code == 401:
            return HTTPException(
                status_code=401,
                detail=f"{self.name}: Authentication failed. Token may be expired or invalid."
            )

        if response.status_code == 403:
            return HTTPException(
                status_code=403,
                detail=f"{self.name}: Permission denied. Check OAuth scopes."
            )

        if response.status_code == 404:
            return HTTPException(
                status_code=404,
                detail=f"{self.name}: Resource not found."
            )

        if response.status_code == 429:
            return HTTPException(
                status_code=429,
                detail=f"{self.name}: Rate limit exceeded. Try again later."
            )

        if response.status_code >= 500:
            return HTTPException(
                status_code=response.status_code,
                detail=f"{self.name}: Google API server error (status {response.status_code})."
            )

        return HTTPException(
            status_code=response.status_code,
            detail=f"{self.name}: API error (status {response.status_code}): {body_snippet}"
        )

    async def _paginate(
        self,
        url: str,
        access_token: Optional[str],
        *,
        params: Optional[Dict[str, Any]] = None,
        page_token_param: str = "pageToken",
        items_key: str = "items",
        max_pages: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Paginate through Google API results using async iteration.

        Yields individual items from paginated responses, handling nextPageToken
        automatically. Memory efficient - streams results instead of loading all
        pages into memory.

        Args:
            url: API endpoint URL
            access_token: OAuth token
            params: Query parameters (will be modified with page token)
            page_token_param: Name of pagination token parameter (default: "pageToken")
            items_key: Key containing items array in response (default: "items")
            max_pages: Maximum pages to fetch (None = fetch all)

        Yields:
            Individual items from the paginated response

        Example:
            # Fetch all Gmail messages
            async for message in self._paginate(
                "https://www.googleapis.com/gmail/v1/users/me/messages",
                access_token,
                params={"q": "is:unread"},
                items_key="messages"
            ):
                print(message["id"])

        Note:
            - Google APIs typically use "pageToken"/"nextPageToken" convention
            - Some APIs use different keys (e.g., Drive uses "files" not "items")
            - Always specify items_key to match the API response structure
        """
        params = params.copy() if params else {}
        page_count = 0

        while True:
            resp = await self._make_request("GET", url, access_token, params=params)
            data = resp.json()

            items = data.get(items_key, [])
            for item in items:
                yield item

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            page_count += 1
            if max_pages and page_count >= max_pages:
                logger.debug(
                    "%s: Reached max_pages limit (%d), stopping pagination",
                    self.name,
                    max_pages
                )
                break

            params[page_token_param] = page_token
