"""
PR Sync Workflow

POC workflow to sync GitHub PR status to Google Sheets.
"""
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel

from shared.tools.executor import execute_tool
from shared.logger import get_logger

logger = get_logger("api.workflows.pr_sync")

router = APIRouter(prefix="/workflows/pr-sync", tags=["workflows"])


class PRSyncRequest(BaseModel):
    """Request model for PR sync workflow."""
    github_connection_id: str
    google_connection_id: str
    repo_owner: str
    repo_name: str
    spreadsheet_id: str
    sheet_range: str = "Sheet1!A1"
    state: str = "open"  # PR state: open, closed, or all


def _get_user_id(request: Request) -> Optional[str]:
    """
    Get user_id from request.state.db_user.
    
    The auth middleware handles authentication and sets db_user in both modes.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User ID string or None if not authenticated
    """
    db_user = getattr(request.state, 'db_user', None)
    return db_user.user_id if db_user else None


@router.post("")
async def sync_pr_status(
    request: Request,
    payload: PRSyncRequest,
    user_id: Optional[str] = Query(None),
):
    """
    Sync latest PR status from GitHub to Google Sheets.
    
    This is a POC workflow that:
    1. Fetches PRs from GitHub using GitHub tools
    2. Formats PR data
    3. Writes to Google Sheets using Google Sheets tool
    
    Args:
        request: FastAPI request
        payload: PR sync request parameters
        user_id: Optional user ID (extracted from request in cloud mode)
    
    Returns:
        Sync result with PR count
    """
    # Get user_id from request or query param
    if not user_id:
        user_id = _get_user_id(request)
    
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id is required"
        )
    
    try:
        logger.info(
            f"Starting PR sync: repo={payload.repo_owner}/{payload.repo_name}, "
            f"spreadsheet={payload.spreadsheet_id}"
        )
        
        # 1. Get PRs using GitHub tool
        logger.info("Fetching PRs from GitHub...")
        prs_result = await execute_tool(
            tool_name="github_list_pull_requests",
            user_id=user_id,
            connection_id=payload.github_connection_id,
            arguments={
                "owner": payload.repo_owner,
                "repo": payload.repo_name,
                "state": payload.state
            }
        )
        
        if not isinstance(prs_result, list):
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected response format from GitHub tool: {type(prs_result)}"
            )
        
        logger.info(f"Fetched {len(prs_result)} PRs from GitHub")
        
        # 2. Format PR data for Sheets
        # Header row
        values = [["PR Number", "Title", "State", "Author", "URL", "Created At", "Updated At"]]
        
        # Data rows
        for pr in prs_result:
            values.append([
                str(pr.get("number", "")),
                pr.get("title", ""),
                pr.get("state", ""),
                pr.get("user", {}).get("login", "") if isinstance(pr.get("user"), dict) else "",
                pr.get("html_url", ""),
                pr.get("created_at", ""),
                pr.get("updated_at", ""),
            ])
        
        logger.info(f"Formatted {len(values) - 1} PRs for Google Sheets")
        
        # 3. Write to Google Sheets
        logger.info("Writing to Google Sheets...")
        sheets_result = await execute_tool(
            tool_name="google_sheets_write",
            user_id=user_id,
            connection_id=payload.google_connection_id,
            arguments={
                "spreadsheet_id": payload.spreadsheet_id,
                "range": payload.sheet_range,
                "values": values,
                "value_input_option": "USER_ENTERED"
            }
        )
        
        updated_cells = sheets_result.get("updatedCells", 0) if isinstance(sheets_result, dict) else 0
        
        logger.info(f"Successfully synced {len(prs_result)} PRs to Google Sheets")
        
        return {
            "status": "success",
            "prs_synced": len(prs_result),
            "updated_cells": updated_cells,
            "spreadsheet_id": payload.spreadsheet_id,
            "range": payload.sheet_range
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"PR sync failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PR sync failed: {str(e)}"
        )


__all__ = ["router"]

