"""
Tool executor for executing tools with OAuth token management.

Handles:
- Loading OAuth tokens from database
- Validating scopes
- Token refresh if expired
- Tool execution
"""
from typing import Any, Dict, Optional
from datetime import datetime, timezone, timedelta
import httpx
from fastapi import HTTPException

from shared.database.models_oauth import OAuthConnection
from shared.database.models import User
from shared.tools.base import BaseTool, get_tool
from shared.tools.scope_validator import validate_scopes
from shared.logger import get_logger
from shared.config import config

logger = get_logger("shared.tools.executor")


async def refresh_oauth_token(connection: OAuthConnection) -> OAuthConnection:
    """
    Refresh an expired OAuth token using refresh_token.
    
    Args:
        connection: OAuthConnection with expired access_token
    
    Returns:
        Updated OAuthConnection with new access_token
    
    Raises:
        HTTPException: If token refresh fails
    """
    if not connection.refresh_token_enc:
        raise HTTPException(
            status_code=401,
            detail=f"No refresh token available for connection {connection.id}"
        )
    
    logger.info(f"Refreshing OAuth token for connection {connection.id} (provider: {connection.provider})")
    
    # Provider-specific refresh endpoints
    if connection.provider in ['google', 'googledrive', 'gmail']:
        if not config.GOOGLE_CLIENT_ID or not config.GOOGLE_CLIENT_SECRET:
            raise HTTPException(
                status_code=500,
                detail="Google OAuth client credentials not configured"
            )
        refresh_url = "https://oauth2.googleapis.com/token"
        refresh_data = {
            "client_id": config.GOOGLE_CLIENT_ID,
            "client_secret": config.GOOGLE_CLIENT_SECRET,
            "refresh_token": connection.refresh_token_enc,
            "grant_type": "refresh_token"
        }
    elif connection.provider == 'github':
        if not config.GITHUB_CLIENT_ID or not config.GITHUB_CLIENT_SECRET:
            raise HTTPException(
                status_code=500,
                detail="GitHub OAuth client credentials not configured"
            )
        refresh_url = "https://github.com/login/oauth/access_token"
        refresh_data = {
            "client_id": config.GITHUB_CLIENT_ID,
            "client_secret": config.GITHUB_CLIENT_SECRET,
            "refresh_token": connection.refresh_token_enc,
            "grant_type": "refresh_token"
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Token refresh not supported for provider: {connection.provider}"
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(refresh_url, data=refresh_data)
            response.raise_for_status()
            
            token_data = response.json()
            
            # Update connection with new token
            connection.access_token_enc = token_data.get('access_token')
            expires_in = token_data.get('expires_in', 3600)  # Default 1 hour
            connection.expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            
            # Update scopes if provided in refresh response
            if 'scope' in token_data:
                connection.scopes = token_data['scope']
            
            connection.updated_at = datetime.now(timezone.utc)
            await connection.save()
            
            logger.info(f"Successfully refreshed token for connection {connection.id}")
            return connection
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Token refresh failed: {e.response.text}")
        raise HTTPException(
            status_code=401,
            detail=f"Token refresh failed: {e.response.text[:200]}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error refreshing token: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Token refresh error: {str(e)}"
        )


async def get_oauth_token(
    user: User,
    connection_id: Optional[str] = None,
    provider: Optional[str] = None
) -> tuple[OAuthConnection, str]:
    """
    Get OAuth access token for a user connection.
    
    Args:
        user_id: User ID (None in self-hosted mode)
        connection_id: OAuthConnection ID (optional if provider is specified)
        provider: Provider name (optional if connection_id is specified)
    
    Returns:
        Tuple of (OAuthConnection, access_token)
    
    Raises:
        HTTPException: If connection not found or token invalid
    """
    
    if connection_id:
        # Parse connection_id (may be "provider:id" or just "id")
        if ":" in connection_id:
            _, db_id = connection_id.split(":", 1)
        else:
            db_id = connection_id
        
        try:
            connection = await OAuthConnection.get(id=int(db_id), user=user, status="active")
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"OAuth connection {connection_id} not found"
            )
    elif provider:
        connection = await OAuthConnection.get_or_none(
            user=user,
            provider=provider,
            status="active"
        )
        if not connection:
            raise HTTPException(
                status_code=404,
                detail=f"No active OAuth connection found for provider: {provider}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Either connection_id or provider must be provided"
        )
    
    # Check if token is expired and refresh if needed
    if connection.expires_at and connection.expires_at < datetime.now(timezone.utc):
        logger.info(f"Token expired for connection {connection.id}, refreshing...")
        connection = await refresh_oauth_token(connection)
    
    if not connection.access_token_enc:
        raise HTTPException(
            status_code=401,
            detail=f"No access token available for connection {connection.id}"
        )
    
    return connection, connection.access_token_enc


async def execute_tool(
    tool_name: str,
    user: User,
    connection_id: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Execute a tool with OAuth token management.
    
    Args:
        tool_name: Name of the tool to execute
        user: User
        connection_id: OAuth connection ID (if tool requires OAuth)
        arguments: Tool arguments
    
    Returns:
        Tool execution result
    
    Raises:
        HTTPException: If tool not found, scopes invalid, or execution fails
    """
    arguments = arguments or {}
    
    # Get tool from registry
    tool = get_tool(tool_name)
    if not tool:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found"
        )
    
    # Get OAuth token if tool requires scopes
    access_token = None
    connection = None
    
    if tool.required_scopes:
        if not user.user_id:
            raise HTTPException(
                status_code=401,
                detail=f"Tool '{tool_name}' requires OAuth authentication. User ID is required."
            )
        
        # If no connection_id provided, try to find connection by tool's integration_type
        provider = None
        if not connection_id:
            # Get integration_type from tool (e.g., "gmail", "googledrive", "github")
            integration_type = getattr(tool, 'provider', None)
            if integration_type:
                provider = integration_type
                logger.info(f"No connection_id provided, using tool integration_type '{provider}' to find connection")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tool '{tool_name}' requires OAuth connection. connection_id must be provided."
                )
        
        connection, access_token = await get_oauth_token(user, connection_id, provider=provider)
        
        # Validate scopes
        is_valid, missing_scope = validate_scopes(connection, tool.required_scopes)
        if not is_valid:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"OAuth connection {connection_id} missing required scope '{missing_scope}' "
                    f"for tool '{tool_name}'. Required scopes: {tool.required_scopes}"
                )
            )
    
    # Execute tool
    try:
        logger.info(f"Executing tool '{tool_name}' for user {user.user_id}")
        result = await tool.execute(access_token, arguments)
        logger.info(f"Tool '{tool_name}' executed successfully")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Tool execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution failed: {str(e)}"
        )

