"""HTTP API client for Seer cloud mode."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from seer_mcp.client.auth import OAuthError, OAuthHandler
from seer_mcp.client.token_store import TokenStore


class SeerAPIClient:
    """Async HTTP client for Seer API with OAuth token management.

    This client handles:
    - OAuth authentication and token refresh
    - Automatic retry with exponential backoff
    - Token expiration handling
    - Rate limiting (429 responses)
    """

    def __init__(self, api_url: str, client_id: str = "seer-mcp-client"):
        """Initialize API client.

        Args:
            api_url: Seer API base URL
            client_id: OAuth client identifier
        """
        self.api_url = api_url.rstrip("/")
        self.client_id = client_id
        self.token_store = TokenStore()
        self.auth_handler = OAuthHandler(api_url, self.token_store, client_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def has_valid_token(self) -> bool:
        """Check if a valid access token exists.

        Returns:
            True if valid token exists
        """
        return self.token_store.has_valid_token()

    async def authenticate(self) -> None:
        """Trigger OAuth flow to get tokens.

        Opens browser for user authorization.
        """
        # Run OAuth flow in thread pool (it's blocking)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.auth_handler.start_auth_flow)

    async def _refresh_if_needed(self) -> None:
        """Refresh access token if expired or about to expire."""
        token_data = self.token_store.load_tokens()
        if not token_data:
            raise ValueError("No tokens available - please authenticate first")

        # Check if token is expired or will expire soon (60 second buffer)
        now = int(datetime.now(timezone.utc).timestamp())
        expires_at = token_data.get("expires_at", 0)

        if expires_at <= (now + 60):
            # Token expired or about to expire - refresh it
            refresh_token = token_data.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh token available - please re-authenticate")

            # Refresh tokens (blocking call)
            loop = asyncio.get_event_loop()
            token_response = await loop.run_in_executor(
                None,
                self.auth_handler.refresh_tokens,
                refresh_token,
            )

            # Save new tokens
            self.token_store.save_tokens(
                access_token=token_response["access_token"],
                refresh_token=token_response["refresh_token"],
                expires_in=token_response["expires_in"],
                api_url=self.api_url,
                scope=token_response["scope"],
            )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        retry_count: int = 0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with automatic token refresh and retry logic.

        Args:
            method: HTTP method
            path: API path (e.g., "/api/v1/workflows")
            retry_count: Current retry attempt
            max_retries: Maximum retry attempts
            **kwargs: Additional arguments for httpx.request

        Returns:
            httpx.Response

        Raises:
            Exception: If request fails after retries
        """
        # Refresh token if needed
        await self._refresh_if_needed()

        # Get access token
        token_data = self.token_store.load_tokens()
        if not token_data:
            raise ValueError("No tokens available")

        access_token = token_data["access_token"]

        # Add authorization header
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {access_token}"

        # Make request
        client = await self._get_client()
        response = await client.request(method, path, headers=headers, **kwargs)

        # Handle 401 Unauthorized - token might be invalid
        if response.status_code == 401 and retry_count < max_retries:
            # Try refreshing token once more
            await self._refresh_if_needed()
            return await self._request(
                method, path, retry_count=retry_count + 1, max_retries=max_retries, **kwargs
            )

        # Handle 429 Rate Limit - exponential backoff
        if response.status_code == 429 and retry_count < max_retries:
            retry_after = int(response.headers.get("Retry-After", 2 ** retry_count))
            await asyncio.sleep(retry_after)
            return await self._request(
                method, path, retry_count=retry_count + 1, max_retries=max_retries, **kwargs
            )

        # Raise on 4xx/5xx errors (except already handled)
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except Exception:  # pylint: disable=broad-exception-caught  # Reason: catch any JSON parsing error
                error_detail = response.text or "Unknown error"
            raise OAuthError(f"API request failed ({response.status_code}): {error_detail}")

        return response

    # =========================================================================
    # Workflow Methods
    # =========================================================================

    async def create_workflow(self, name: str, description: str, spec: dict) -> dict:
        """Create a new workflow.

        Args:
            name: Workflow name
            description: Workflow description
            spec: Workflow specification dict

        Returns:
            Created workflow dict
        """
        response = await self._request(
            "POST",
            "/api/v1/workflows",
            json={"name": name, "description": description, "spec": spec},
        )
        return response.json()

    async def list_workflows(self, limit: int = 50, cursor: Optional[str] = None) -> dict:
        """List workflows.

        Args:
            limit: Maximum number of workflows to return
            cursor: Pagination cursor

        Returns:
            Workflows list with pagination info
        """
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        response = await self._request("GET", "/api/v1/workflows", params=params)
        return response.json()

    async def get_workflow(self, workflow_id: str) -> dict:
        """Get workflow by ID.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow dict
        """
        response = await self._request("GET", f"/api/v1/workflows/{workflow_id}")
        return response.json()

    async def update_workflow(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        spec: Optional[dict] = None,
    ) -> dict:
        """Update workflow metadata or spec.

        Args:
            workflow_id: Workflow identifier
            name: New workflow name
            description: New workflow description
            spec: New workflow spec (will create new draft)

        Returns:
            Updated workflow dict
        """
        # Update metadata if provided
        if name is not None or description is not None:
            metadata_payload = {}
            if name is not None:
                metadata_payload["name"] = name
            if description is not None:
                metadata_payload["description"] = description

            response = await self._request(
                "PATCH",
                f"/api/v1/workflows/{workflow_id}",
                json=metadata_payload,
            )
            result = response.json()
        else:
            # Get current workflow
            result = await self.get_workflow(workflow_id)

        # Update spec if provided
        if spec is not None:
            response = await self._request(
                "PATCH",
                f"/api/v1/workflows/{workflow_id}/draft",
                json={"spec": spec},
            )
            result = response.json()

        return result

    async def delete_workflow(self, workflow_id: str) -> None:
        """Delete workflow.

        Args:
            workflow_id: Workflow identifier
        """
        await self._request("DELETE", f"/api/v1/workflows/{workflow_id}")

    # =========================================================================
    # Execution Methods
    # =========================================================================

    async def run_workflow(
        self,
        workflow_id: str,
        inputs: Optional[dict] = None,
        test_mode: bool = False,
    ) -> dict:
        """Run a workflow.

        Args:
            workflow_id: Workflow identifier
            inputs: Workflow input variables
            test_mode: Whether to run in test mode

        Returns:
            Run result dict
        """
        payload = {"workflow_id": workflow_id, "inputs": inputs or {}}
        if test_mode:
            payload["test_mode"] = True

        response = await self._request("POST", "/api/v1/runs", json=payload)
        return response.json()

    async def get_execution(self, run_id: str) -> dict:
        """Get execution details.

        Args:
            run_id: Run identifier

        Returns:
            Execution dict
        """
        response = await self._request("GET", f"/api/v1/runs/{run_id}")
        return response.json()

    async def get_execution_history(self, run_id: str) -> dict:
        """Get execution history/events.

        Args:
            run_id: Run identifier

        Returns:
            Execution history dict
        """
        response = await self._request("GET", f"/api/v1/runs/{run_id}/history")
        return response.json()

    async def list_executions(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> dict:
        """List workflow executions.

        Args:
            workflow_id: Optional workflow ID filter
            limit: Maximum number of executions to return
            cursor: Pagination cursor

        Returns:
            Executions list with pagination info
        """
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        if workflow_id:
            # List runs for specific workflow
            response = await self._request(
                "GET",
                f"/api/v1/workflows/{workflow_id}/runs",
                params=params,
            )
        else:
            # List all runs (if supported by API)
            response = await self._request("GET", "/api/v1/runs", params=params)

        return response.json()

    # =========================================================================
    # Integration Methods
    # =========================================================================

    async def list_integrations(self) -> list:
        """List available integrations.

        Returns:
            List of integration dicts
        """
        response = await self._request("GET", "/api/integrations")
        return response.json()

    async def get_oauth_url(self, integration_name: str) -> str:
        """Get OAuth authorization URL for an integration.

        Args:
            integration_name: Integration name (e.g., "google", "github")

        Returns:
            OAuth authorization URL
        """
        response = await self._request(
            "POST",
            f"/api/integrations/{integration_name}/connect",
        )
        data = response.json()
        return data.get("url", "")
