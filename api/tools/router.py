"""
Tool API router for listing and executing tools.
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Query, HTTPException, Request
from pydantic import BaseModel
from shared.database.models import User
from api.tools.services import list_tools, execute_tool_service

router = APIRouter(prefix="/api/tools", tags=["tools"])


class ExecuteToolRequest(BaseModel):
    """Request body for tool execution."""
    connection_id: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


class ExecuteToolResponse(BaseModel):
    """Response for tool execution."""
    data: Any
    success: bool
    error: Optional[str] = None


@router.get("")
async def list_tools_endpoint(
    integration_type: Optional[str] = Query(None, description="Filter by integration type (e.g., gmail, github)")
) -> Dict[str, Any]:
    """
    List available tools.
    
    Returns tools with metadata including name, description, required_scopes, and parameters schema.
    """
    return await list_tools(integration_type=integration_type)


@router.post("/{tool_name}/execute", response_model=ExecuteToolResponse)
async def execute_tool_endpoint(
    request: Request,
    tool_name: str,
    payload: ExecuteToolRequest = Body(...),
) -> Dict[str, Any]:
    """
    Execute a tool.
    
    Args:
        tool_name: Name of the tool to execute
        payload: Execution request with  connection_id (optional), and arguments
    
    Returns:
        Tool execution result with data and success flag
    """
    user:User = request.state.db_user
    try:
        result = await execute_tool_service(
            tool_name=tool_name,
            user=user,
            connection_id=payload.connection_id,
            arguments=payload.arguments
        )
        return result
    except HTTPException as e:
        return {
            "data": None,
            "success": False,
            "error": e.detail
        }


__all__ = ["router"]

