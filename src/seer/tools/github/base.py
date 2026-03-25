"""
Base class for GitHub API tools.

Provides shared functionality for all GitHub API integrations:
- HTTP client with OAuth token authentication
- Consistent error handling and mapping
- Common response parsing
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool
from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.github.base")

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com"

# Common parameter schemas shared across GitHub tools
OWNER_PARAM = {
    "type": "string",
    "description": "Repository owner (username or organization)",
}
REPO_PARAM = {
    "type": "string",
    "description": "Repository name",
}
PAGINATION_PARAMS = {
    "per_page": {
        "type": "integer",
        "description": "Number of items per page (max 100)",
        "default": 30,
        "maximum": 100,
    },
    "page": {
        "type": "integer",
        "description": "Page number for pagination",
        "default": 1,
    },
}


@dataclass
class _RequestArgs:
    """Bundles per-request data passed to _dispatch_request."""

    url: str
    headers: Dict[str, str]
    params: Optional[Dict[str, Any]] = None
    json_body: Optional[Dict[str, Any]] = None


class GitHubAPIClient(BaseTool, ABC):
    """
    Abstract base class for all GitHub API tools.

    Provides centralized HTTP request handling, OAuth token authentication,
    error translation, and consistent behavior.
    """

    provider = "github"
    default_timeout: float = 30.0

    def _get_access_token(self, credentials: Optional[ResolvedCredentials] = None) -> str:
        """
        Get GitHub access token from credentials.

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
            detail=f"{self.name} requires a GitHub access token. Please connect your GitHub account."
        )

    def _build_headers(self, access_token: str) -> Dict[str, str]:
        """Build HTTP headers for GitHub API request."""
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
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
    ) -> Any:
        """
        Make authenticated HTTP request to GitHub API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (e.g., "repos/{owner}/{repo}/commits")
            credentials: ResolvedCredentials with access token
            params: Query parameters
            json_body: JSON request body
            timeout: Request timeout in seconds

        Returns:
            GitHub API response (dict or list depending on endpoint)

        Raises:
            HTTPException: On API errors
        """
        access_token = self._get_access_token(credentials)
        headers = self._build_headers(access_token)
        timeout_value = timeout or self.default_timeout
        url = f"{GITHUB_API_BASE}/{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                req = _RequestArgs(url=url, headers=headers, params=params, json_body=json_body)
                resp = await self._dispatch_request(client, method, req)

                if not resp.is_success:
                    raise self._handle_github_error(resp)

                # Some endpoints return empty body on success (e.g., DELETE)
                if resp.status_code == 204:
                    return {"success": True}

                return resp.json()

        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"{self.name} request timed out after {timeout_value}s"
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("Unexpected error in %s", self.name)
            raise HTTPException(
                status_code=500,
                detail=f"{self.name} error: {str(exc)}"
            ) from exc

    async def _dispatch_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        req: _RequestArgs,
    ) -> httpx.Response:
        """Dispatch an HTTP request using the given method."""
        method_upper = method.upper()
        if method_upper == "GET":
            return await client.get(req.url, headers=req.headers, params=req.params)
        if method_upper == "POST":
            return await client.post(req.url, headers=req.headers, params=req.params, json=req.json_body)
        if method_upper == "PATCH":
            return await client.patch(req.url, headers=req.headers, params=req.params, json=req.json_body)
        if method_upper == "DELETE":
            return await client.delete(req.url, headers=req.headers, params=req.params)
        raise HTTPException(status_code=400, detail=f"Unsupported HTTP method: {method}")

    def _handle_github_error(self, response: httpx.Response) -> HTTPException:
        """
        Translate GitHub API error responses to HTTPExceptions.

        Args:
            response: httpx Response object

        Returns:
            HTTPException with appropriate status code and message
        """
        status_code = response.status_code

        try:
            error_data = response.json()
            message = error_data.get("message", "Unknown error")
            documentation_url = error_data.get("documentation_url", "")
        except ValueError:
            message = response.text or "Unknown error"
            documentation_url = ""

        # Map GitHub-specific error codes to user-friendly messages
        error_messages = {
            401: "Authentication failed. Token may be expired or invalid.",
            403: "Access forbidden. Check repository permissions or rate limits.",
            404: "Resource not found. Check owner, repo, and resource identifiers.",
            409: "Conflict. The resource may already exist or be in an invalid state.",
            422: f"Validation failed: {message}",
        }

        # Special handling for rate limiting
        if status_code == 403 and "rate limit" in message.lower():
            retry_after = response.headers.get("X-RateLimit-Reset", "")
            detail = f"Rate limited. Retry after timestamp: {retry_after}" if retry_after else "Rate limited. Please try again later."
            return HTTPException(status_code=429, detail=f"{self.name}: {detail}")

        # Get mapped message or use the API response
        detail = error_messages.get(status_code, f"GitHub API error: {message}")

        if documentation_url:
            detail = f"{detail} See: {documentation_url}"

        return HTTPException(status_code=status_code, detail=f"{self.name}: {detail}")
