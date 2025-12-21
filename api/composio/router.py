from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel

from src.composio.services import (
    delete_connected_account,
    execute_tool,
    initiate_connection,
    list_connected_accounts,
    wait_for_connection,
)


router = APIRouter(prefix="/api/composio", tags=["composio"])


class ConnectRequest(BaseModel):
    user_id: str
    auth_config_id: str
    callback_url: Optional[str] = None


class ConnectResponse(BaseModel):
    redirectUrl: str
    connectionId: str


class WaitForConnectionRequest(BaseModel):
    connection_id: str
    timeout_ms: Optional[int] = None


class WaitForConnectionResponse(BaseModel):
    status: str
    connectedAccountId: str


class ExecuteToolRequest(BaseModel):
    user_id: str
    connected_account_id: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


class ExecuteToolResponse(BaseModel):
    data: Any
    success: bool


@router.get("/connected-accounts")
async def list_connected_accounts_endpoint(
    user_ids: Optional[List[str]] = Query(default=None),
    toolkit_slugs: Optional[List[str]] = Query(default=None),
    auth_config_ids: Optional[List[str]] = Query(default=None),
) -> Dict[str, Any]:
    """
    List connected accounts for a user/toolkit/auth config combination.

    This is a thin wrapper around Composio's `/api/v3/connected_accounts`
    with a simplified response shape for the frontend.
    """
    return await list_connected_accounts(
        user_ids=user_ids,
        toolkit_slugs=toolkit_slugs,
        auth_config_ids=auth_config_ids,
    )


@router.post("/connect", response_model=ConnectResponse)
async def initiate_connection_endpoint(payload: ConnectRequest = Body(...)) -> Dict[str, Any]:
    """Initiate a Composio connection and return a redirect URL + connection id."""
    return await initiate_connection(
        user_id=payload.user_id,
        auth_config_id=payload.auth_config_id,
        callback_url=payload.callback_url,
    )


@router.post("/wait-for-connection", response_model=WaitForConnectionResponse)
async def wait_for_connection_endpoint(
    payload: WaitForConnectionRequest = Body(...),
) -> Dict[str, Any]:
    """Wait for a Composio connection to transition to ACTIVE."""
    timeout_ms = payload.timeout_ms if payload.timeout_ms is not None else 120_000
    return await wait_for_connection(
        connection_id=payload.connection_id,
        timeout_ms=timeout_ms,
    )


@router.delete("/connected-accounts/{account_id}", status_code=204)
async def delete_connected_account_endpoint(account_id: str) -> None:
    """Delete (soft-delete) a Composio connected account."""
    await delete_connected_account(account_id=account_id)


@router.post("/tools/execute/{tool_slug}", response_model=ExecuteToolResponse)
async def execute_tool_endpoint(
    tool_slug: str,
    payload: ExecuteToolRequest = Body(...),
) -> Dict[str, Any]:
    """Execute a Composio tool on behalf of a user/connected account."""
    return await execute_tool(
        tool_slug=tool_slug,
        user_id=payload.user_id,
        connected_account_id=payload.connected_account_id,
        arguments=payload.arguments,
    )


__all__ = ["router"]


