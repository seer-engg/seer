"""
Base class for E2B sandbox tools.

Provides shared functionality for all E2B API integrations:
- AsyncSandbox client management
- Platform API key authentication
- Consistent error handling
"""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, List, Optional

from fastapi import HTTPException

from seer.config import config
from seer.logger import get_logger
from seer.tools.base import BaseTool

if TYPE_CHECKING:
    from e2b_code_interpreter import AsyncSandbox

logger = get_logger("tools.e2b.base")


class E2BToolBase(BaseTool, ABC):
    """
    Abstract base class for E2B sandbox tools.

    Provides:
    - API key validation from config
    - AsyncSandbox factory method
    - Consistent error handling

    E2B uses platform-owned API keys (not user OAuth), so these tools
    are always available for all customers without connection setup.
    """

    provider = None  # No OAuth provider
    integration_type = "codebox"
    required_scopes: List[str] = []  # Platform-owned API key, no OAuth

    def _require_api_key(self) -> str:
        """Get E2B API key from config or raise 503."""
        if not config.e2b_api_key:
            raise HTTPException(
                status_code=503,
                detail="CodeBox sandbox execution is not available: E2B_API_KEY not configured.",
            )
        return config.e2b_api_key

    async def _create_sandbox(
        self,
        timeout: Optional[int] = None,
    ) -> "AsyncSandbox":
        """
        Create a new E2B sandbox with configured API key.

        Args:
            timeout: Sandbox timeout in seconds. Defaults to config value.

        Returns:
            AsyncSandbox instance ready for code execution.

        Raises:
            HTTPException: 503 if package not installed, 502 if creation fails.
        """
        try:
            from e2b_code_interpreter import AsyncSandbox  # pylint: disable=import-outside-toplevel  # optional dependency, handle ImportError
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail="CodeBox sandbox not available: e2b-code-interpreter package not installed.",
            ) from e

        api_key = self._require_api_key()
        effective_timeout = timeout or config.e2b_sandbox_timeout_seconds

        try:
            sandbox = await AsyncSandbox.create(
                api_key=api_key,
                timeout=effective_timeout,
            )
            logger.debug("Created E2B sandbox: id=%s, timeout=%d", sandbox.sandbox_id, effective_timeout)
            return sandbox
        except Exception as e:
            logger.exception("Failed to create E2B sandbox")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to create sandbox: {e!s}",
            ) from e

    async def _connect_sandbox(self, sandbox_id: str) -> "AsyncSandbox":
        """
        Connect to an existing E2B sandbox by ID.

        Args:
            sandbox_id: The sandbox identifier from a previous create call.

        Returns:
            AsyncSandbox instance connected to the existing sandbox.

        Raises:
            HTTPException: 503 if package not installed, 404 if sandbox not found.
        """
        try:
            from e2b_code_interpreter import AsyncSandbox  # pylint: disable=import-outside-toplevel  # optional dependency, handle ImportError
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail="E2B sandbox not available: e2b-code-interpreter package not installed.",
            ) from e

        api_key = self._require_api_key()

        try:
            sandbox = await AsyncSandbox.connect(  # pylint: disable=no-value-for-parameter  # false positive: connect is a classmethod
                sandbox_id=sandbox_id,
                api_key=api_key,
            )
            logger.debug("Connected to E2B sandbox: id=%s", sandbox_id)
            return sandbox
        except Exception as e:
            logger.exception("Failed to connect to sandbox %s", sandbox_id)
            raise HTTPException(
                status_code=404,
                detail=f"Sandbox '{sandbox_id}' not found or expired: {e!s}",
            ) from e


__all__ = ["E2BToolBase"]
