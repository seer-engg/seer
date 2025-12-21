import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

import logging
COMPOSIO_BASE_URL = os.getenv("COMPOSIO_BASE_URL") or "https://backend.composio.dev"
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
logger = logging.getLogger(__name__)


def _get_composio_headers() -> Dict[str, str]:
    """Build headers for Composio API requests."""
    if not COMPOSIO_API_KEY:
        logger.error("COMPOSIO_API_KEY is not set in environment")
        raise HTTPException(
            status_code=500,
            detail="COMPOSIO_API_KEY must be set in the backend environment",
        )
    return {
        "x-api-key": COMPOSIO_API_KEY,
        "Content-Type": "application/json",
    }


async def list_connected_accounts(
    user_ids: Optional[List[str]] = None,
    toolkit_slugs: Optional[List[str]] = None,
    auth_config_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    List Composio connected accounts with optional filtering.

    This wraps the Composio `GET /api/v3/connected_accounts` endpoint and
    normalizes the response into the simplified shape expected by the frontend.
    """
    logger.info(
        "Listing connected accounts",
        extra={
            "user_ids": user_ids,
            "toolkit_slugs": toolkit_slugs,
            "auth_config_ids": auth_config_ids,
        },
    )

    params: Dict[str, str] = {}

    if user_ids:
        params["user_ids"] = ",".join(user_ids)
    if toolkit_slugs:
        params["toolkit_slugs"] = ",".join(toolkit_slugs)
    if auth_config_ids:
        params["auth_config_ids"] = ",".join(auth_config_ids)

    try:
        async with httpx.AsyncClient(base_url=COMPOSIO_BASE_URL, timeout=30.0) as client:
            logger.debug(
                "Sending request to Composio API",
                extra={"endpoint": "/api/v3/connected_accounts", "params": params},
            )
            response = await client.get(
                "/api/v3/connected_accounts",
                headers=_get_composio_headers(),
                params=params or None,
            )

        logger.debug(
            "Received response from Composio API",
            extra={
                "status_code": response.status_code,
                "response_length": len(response.text),
            },
        )

        if response.status_code != 200:
            logger.error(
                "Failed to fetch connected accounts from Composio",
                extra={
                    "status_code": response.status_code,
                    "response_text": response.text[:500],
                },
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching connected accounts from Composio: {response.text[:500]}",
            )

        data = response.json()
        items = data.get("items", [])

        logger.info(
            "Successfully fetched connected accounts",
            extra={"total_items": len(items)},
        )

        # Normalize into the minimal shape used by the frontend proxy client
        normalized_items: List[Dict[str, Any]] = []
        for item in items:
            status = item.get("status", "INACTIVE")
            # Map Composio's detailed statuses into a simpler set
            if status in ("INITIALIZING", "INITIATED"):
                normalized_status = "PENDING"
            elif status == "ACTIVE":
                normalized_status = "ACTIVE"
            else:
                normalized_status = "INACTIVE"

            normalized_items.append(
                {
                    "id": item.get("id"),
                    "status": normalized_status,
                    # Optional user_id for debugging / inspection in the UI
                    "user_id": item.get("connection", {}).get("user_id")
                    if isinstance(item.get("connection"), dict)
                    else None,
                    "toolkit": {
                        "slug": (
                            item.get("toolkit", {}).get("slug")
                            if isinstance(item.get("toolkit"), dict)
                            else None
                        )
                    },
                }
            )

        return {
            "items": normalized_items,
            "total": data.get("total_items", len(normalized_items)),
        }

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected error fetching connected accounts from Composio",
            extra={"error": str(exc)},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching connected accounts from Composio: {str(exc)}",
        ) from exc


async def initiate_connection(
    *,
    user_id: str,
    auth_config_id: str,
    callback_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Initiate a Composio OAuth / auth connection for a user.

    Wraps `POST /api/v3/connected_accounts` and returns only the fields
    needed by the frontend: redirectUrl and connectionId.
    """
    logger.info(
        "Initiating Composio connection",
        extra={
            "user_id": user_id,
            "auth_config_id": auth_config_id,
            "has_callback_url": callback_url is not None,
        },
    )

    body: Dict[str, Any] = {
        "auth_config": {"id": auth_config_id},
        "connection": {
            "user_id": user_id,
        },
    }
    if callback_url:
        body["connection"]["callback_url"] = callback_url

    try:
        async with httpx.AsyncClient(base_url=COMPOSIO_BASE_URL, timeout=30.0) as client:
            logger.debug(
                "Sending connection initiation request to Composio",
                extra={"endpoint": "/api/v3/connected_accounts", "user_id": user_id},
            )
            response = await client.post(
                "/api/v3/connected_accounts",
                headers=_get_composio_headers(),
                json=body,
            )

        logger.debug(
            "Received connection initiation response",
            extra={
                "status_code": response.status_code,
                "user_id": user_id,
            },
        )

        if response.status_code not in (200, 201):
            logger.error(
                "Failed to initiate Composio connection",
                extra={
                    "status_code": response.status_code,
                    "response_text": response.text[:500],
                    "user_id": user_id,
                },
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error initiating Composio connection: {response.text[:500]}",
            )

        data = response.json()
        connection_id = data.get("id")

        # connectionData is a tagged union; we only care about a possible redirectUrl
        redirect_url: Optional[str] = None
        connection_data = data.get("connectionData")
        if isinstance(connection_data, dict):
            val = connection_data.get("val")
            if isinstance(val, dict):
                redirect = val.get("redirectUrl") or val.get("redirect_url")
                if isinstance(redirect, str):
                    redirect_url = redirect

        if not redirect_url or not connection_id:
            logger.error(
                "Missing redirectUrl or connection_id in Composio response",
                extra={
                    "has_redirect_url": redirect_url is not None,
                    "has_connection_id": connection_id is not None,
                    "user_id": user_id,
                },
            )
            raise HTTPException(
                status_code=500,
                detail="Unexpected Composio response: missing redirectUrl or connection id",
            )

        logger.info(
            "Successfully initiated Composio connection",
            extra={
                "connection_id": connection_id,
                "user_id": user_id,
            },
        )

        return {
            "redirectUrl": redirect_url,
            "connectionId": connection_id,
        }

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected error initiating Composio connection",
            extra={"user_id": user_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error initiating Composio connection: {str(exc)}",
        ) from exc


async def wait_for_connection(
    *,
    connection_id: str,
    timeout_ms: int = 120_000,
) -> Dict[str, Any]:
    """
    Poll Composio for a connection to become ACTIVE.

    This mirrors the behavior of `connectedAccounts.waitForConnection` in
    the `@composio/core` SDK by repeatedly calling
    `GET /api/v3/connected_accounts/{id}` until the status is ACTIVE or a
    terminal error/timeout is reached.
    """
    logger.info(
        "Waiting for Composio connection to become ACTIVE",
        extra={"connection_id": connection_id, "timeout_ms": timeout_ms},
    )

    # Bounds to avoid very small or huge timeouts
    timeout_ms = max(1_000, min(timeout_ms, 10 * 60_000))
    poll_interval_ms = 1_000
    deadline = asyncio.get_event_loop().time() + timeout_ms / 1000.0

    try:
        async with httpx.AsyncClient(base_url=COMPOSIO_BASE_URL, timeout=30.0) as client:
            last_status: Optional[str] = None
            poll_count = 0

            while True:
                poll_count += 1

                if asyncio.get_event_loop().time() >= deadline:
                    logger.error(
                        "Timed out waiting for Composio connection",
                        extra={
                            "connection_id": connection_id,
                            "last_status": last_status,
                            "poll_count": poll_count,
                            "timeout_ms": timeout_ms,
                        },
                    )
                    raise HTTPException(
                        status_code=408,
                        detail=f"Timed out waiting for Composio connection {connection_id} "
                        f"to become ACTIVE (last status={last_status})",
                    )

                logger.debug(
                    "Polling Composio connection status",
                    extra={
                        "connection_id": connection_id,
                        "poll_count": poll_count,
                    },
                )

                response = await client.get(
                    f"/api/v3/connected_accounts/{connection_id}",
                    headers=_get_composio_headers(),
                )

                if response.status_code == 404:
                    logger.error(
                        "Composio connected account not found",
                        extra={"connection_id": connection_id},
                    )
                    raise HTTPException(
                        status_code=404,
                        detail=f"Composio connected account {connection_id} not found",
                    )

                if response.status_code != 200:
                    logger.error(
                        "Error fetching Composio connection status",
                        extra={
                            "connection_id": connection_id,
                            "status_code": response.status_code,
                            "response_text": response.text[:500],
                        },
                    )
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Error fetching Composio connection status: {response.text[:500]}",
                    )

                data = response.json()
                status = data.get("status")
                last_status = status

                logger.debug(
                    "Received connection status",
                    extra={
                        "connection_id": connection_id,
                        "status": status,
                        "poll_count": poll_count,
                    },
                )

                if status == "ACTIVE":
                    logger.info(
                        "Composio connection became ACTIVE",
                        extra={
                            "connection_id": connection_id,
                            "poll_count": poll_count,
                        },
                    )
                    return {
                        "status": status,
                        "connectedAccountId": data.get("id", connection_id),
                    }

                # Terminal error states from the JS SDK
                if status in {"FAILED", "EXPIRED"}:
                    logger.error(
                        "Composio connection failed with terminal status",
                        extra={
                            "connection_id": connection_id,
                            "status": status,
                            "poll_count": poll_count,
                        },
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Composio connection {connection_id} failed with status {status}",
                    )

                await asyncio.sleep(poll_interval_ms / 1000.0)

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected error waiting for Composio connection",
            extra={"connection_id": connection_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error waiting for Composio connection {connection_id}: {str(exc)}",
        ) from exc


async def delete_connected_account(*, account_id: str) -> None:
    """Soft delete a connected account in Composio."""
    logger.info(
        "Deleting Composio connected account",
        extra={"account_id": account_id},
    )

    try:
        async with httpx.AsyncClient(base_url=COMPOSIO_BASE_URL, timeout=30.0) as client:
            logger.debug(
                "Sending delete request to Composio",
                extra={"account_id": account_id},
            )
            response = await client.delete(
                f"/api/v3/connected_accounts/{account_id}",
                headers=_get_composio_headers(),
            )

        logger.debug(
            "Received delete response from Composio",
            extra={
                "account_id": account_id,
                "status_code": response.status_code,
            },
        )

        if response.status_code not in (200, 204):
            logger.error(
                "Failed to delete Composio connected account",
                extra={
                    "account_id": account_id,
                    "status_code": response.status_code,
                    "response_text": response.text[:500],
                },
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error deleting Composio connected account: {response.text[:500]}",
            )

        logger.info(
            "Successfully deleted Composio connected account",
            extra={"account_id": account_id},
        )

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected error deleting Composio connected account",
            extra={"account_id": account_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting Composio connected account: {str(exc)}",
        ) from exc


async def execute_tool(
    *,
    tool_slug: str,
    user_id: str,
    connected_account_id: Optional[str],
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a Composio tool.

    Wraps `POST /api/v3/tools/execute/{tool_slug}` and normalizes the
    response into `{ data, success }` for the frontend.
    """
    logger.info(
        "Executing Composio tool",
        extra={
            "tool_slug": tool_slug,
            "user_id": user_id,
            "connected_account_id": connected_account_id,
            "has_arguments": arguments is not None,
        },
    )

    body: Dict[str, Any] = {
        "user_id": user_id,
    }
    if connected_account_id:
        body["connected_account_id"] = connected_account_id
    if arguments:
        body["arguments"] = arguments

    try:
        async with httpx.AsyncClient(base_url=COMPOSIO_BASE_URL, timeout=60.0) as client:
            logger.debug(
                "Sending tool execution request to Composio",
                extra={
                    "tool_slug": tool_slug,
                    "user_id": user_id,
                },
            )
            response = await client.post(
                f"/api/v3/tools/execute/{tool_slug}",
                headers=_get_composio_headers(),
                json=body,
            )

        logger.debug(
            "Received tool execution response from Composio",
            extra={
                "tool_slug": tool_slug,
                "status_code": response.status_code,
                "user_id": user_id,
            },
        )

        if response.status_code != 200:
            logger.error(
                "Failed to execute Composio tool",
                extra={
                    "tool_slug": tool_slug,
                    "user_id": user_id,
                    "status_code": response.status_code,
                    "response_text": response.text[:500],
                },
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error executing Composio tool {tool_slug}: {response.text[:500]}",
            )

        data = response.json()

        # ToolExecuteResponse has shape { data, error, successful, ... }
        success = bool(data.get("successful", False))

        logger.info(
            "Successfully executed Composio tool",
            extra={
                "tool_slug": tool_slug,
                "user_id": user_id,
                "success": success,
            },
        )

        return {
            "data": data.get("data"),
            "success": success,
        }

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected error executing Composio tool",
            extra={
                "tool_slug": tool_slug,
                "user_id": user_id,
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error executing Composio tool {tool_slug}: {str(exc)}",
        ) from exc


__all__ = [
    "list_connected_accounts",
    "initiate_connection",
    "wait_for_connection",
    "delete_connected_account",
    "execute_tool",
]

