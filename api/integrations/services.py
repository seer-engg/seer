from shared.database.models_oauth import OAuthConnection
from shared.database.models import User
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from shared.logger import get_logger
logger = get_logger("api.integrations.services")


def merge_scopes(existing_scopes: str, new_scopes: str) -> str:
    """
    Merge existing scopes with new scopes, removing duplicates.
    
    Args:
        existing_scopes: Space-separated string of existing scopes
        new_scopes: Space-separated string of new scopes to add
    
    Returns:
        Space-separated string of merged scopes
    """
    existing_set: Set[str] = set(existing_scopes.split()) if existing_scopes else set()
    new_set: Set[str] = set(new_scopes.split()) if new_scopes else set()
    merged = existing_set | new_set
    return " ".join(sorted(merged))


def has_required_scopes(granted_scopes: str, required_scopes: List[str]) -> bool:
    """
    Check if granted scopes include all required scopes.
    
    Args:
        granted_scopes: Space-separated string of granted scopes
        required_scopes: List of required scope strings
    
    Returns:
        True if all required scopes are granted
    """
    if not required_scopes:
        return True
    granted_set = set(granted_scopes.split()) if granted_scopes else set()
    return all(scope in granted_set for scope in required_scopes)


def get_oauth_provider(integration_type: str) -> str:
    """
    Map integration type to OAuth provider.
    Multiple integration types can share the same OAuth provider.
    
    Args:
        integration_type: Integration type (gmail, googlesheets, googledrive, etc.)
    
    Returns:
        OAuth provider name (google, github, etc.)
    """
    google_integrations = ['gmail', 'googlesheets', 'googledrive', 'google']
    if integration_type in google_integrations:
        return 'google'
    # For other providers, the integration type is the same as the provider
    return integration_type


async def store_oauth_connection(
    user_id: str,
    provider: str,
    token: Dict[str, Any],
    profile: Dict[str, Any],
    granted_scopes: str = "",
    integration_type: Optional[str] = None
):
    """
    Store OAuth connection with granted scopes.
    Connections are stored by OAuth provider (e.g., 'google') and scopes are merged
    when the same provider is connected again with different scopes.
    
    Args:
        user_id: User ID
        provider: OAuth provider name (google, github, etc.) - NOT integration type
        token: OAuth token response dict
        profile: User profile information
        granted_scopes: Space-separated string of granted OAuth scopes
        integration_type: Original integration type that triggered this connection (for logging)
    """
    # Normalize provider to OAuth provider
    oauth_provider = get_oauth_provider(provider)
    
    logger.info(f"Storing OAuth connection: user_id={user_id}, oauth_provider={oauth_provider}, "
                f"integration_type={integration_type}, scopes={granted_scopes[:100]}...")
    
    # Find user
    user = await User.get(user_id=user_id)
    
    # Extract provider account id
    if oauth_provider == 'google':
        provider_account_id = profile.get('sub') or profile.get('email')
    elif oauth_provider == 'github':
        provider_account_id = str(profile.get('id'))
    else:
        provider_account_id = str(profile.get('id', 'unknown'))
        
    provider_metadata = profile
        
    # Tokens
    access_token = token.get('access_token')
    refresh_token = token.get('refresh_token')
    expires_at_ts = token.get('expires_at')
    expires_at = datetime.fromtimestamp(expires_at_ts, tz=timezone.utc) if expires_at_ts else None
    
    # Extract token_type (usually 'Bearer')
    token_type = token.get('token_type', 'Bearer')
    
    # Update or Create - always use OAuth provider (not integration type)
    connection = await OAuthConnection.get_or_none(
        user=user,
        provider=oauth_provider,
        provider_account_id=provider_account_id
    )
    
    if connection:
        connection.access_token_enc = access_token
        if refresh_token:
            connection.refresh_token_enc = refresh_token
        connection.provider_metadata = provider_metadata
        connection.status = "active"
        connection.expires_at = expires_at
        # IMPORTANT: Merge scopes instead of replacing them
        connection.scopes = merge_scopes(connection.scopes or "", granted_scopes)
        connection.token_type = token_type
        connection.updated_at = datetime.now(timezone.utc)
        await connection.save()
        logger.info(f"Updated existing connection for {oauth_provider}, merged scopes: {connection.scopes[:100]}...")
    else:
        connection = await OAuthConnection.create(
            user=user,
            provider=oauth_provider,
            provider_account_id=provider_account_id,
            access_token_enc=access_token,
            refresh_token_enc=refresh_token,
            provider_metadata=provider_metadata,
            status="active",
            expires_at=expires_at,
            scopes=granted_scopes,
            token_type=token_type
        )
        logger.info(f"Created new connection for {oauth_provider}")
        
    return connection

async def list_connections(user: User):
    """
    List all active OAuth connections for a user.
    """
    try:
        logger.info(f"Listing connections for user {user.user_id}")
        connections = await OAuthConnection.filter(user=user, status="active").all()
        return connections
    except Exception as e:
        logger.error(f"Error listing connections for user {user.user_id}: {e}")
        return []


async def get_connection_for_provider(user: User, provider: str) -> Optional[OAuthConnection]:
    """
    Get active OAuth connection for a specific provider.
    
    Args:
        user: User model instance
        provider: OAuth provider name (google, github, etc.)
    
    Returns:
        OAuthConnection if found, None otherwise
    """
    oauth_provider = get_oauth_provider(provider)
    try:
        connection = await OAuthConnection.get_or_none(
            user=user,
            provider=oauth_provider,
            status="active"
        )
        return connection
    except Exception as e:
        logger.error(f"Error getting connection for provider {provider}: {e}")
        return None


async def get_tool_connection_status(user: User, tool_name: str, required_scopes: List[str], provider: str) -> Dict[str, Any]:
    """
    Check if user has a connection with required scopes for a specific tool.
    
    Args:
        user: User model instance
        tool_name: Name of the tool
        required_scopes: List of OAuth scopes required by the tool
        provider: OAuth provider for the tool (google, github, etc.)
    
    Returns:
        Dict with connection status information
    """
    oauth_provider = get_oauth_provider(provider)
    
    try:
        connection = await OAuthConnection.get_or_none(
            user=user,
            provider=oauth_provider,
            status="active"
        )
        
        if not connection:
            return {
                "tool_name": tool_name,
                "connected": False,
                "has_required_scopes": False,
                "missing_scopes": required_scopes,
                "provider": oauth_provider,
                "connection_id": None
            }
        
        # Check if connection has all required scopes
        granted_scopes = connection.scopes or ""
        has_scopes = has_required_scopes(granted_scopes, required_scopes)
        
        # Find missing scopes
        granted_set = set(granted_scopes.split()) if granted_scopes else set()
        missing = [s for s in required_scopes if s not in granted_set]
        
        return {
            "tool_name": tool_name,
            "connected": True,
            "has_required_scopes": has_scopes,
            "missing_scopes": missing,
            "provider": oauth_provider,
            "connection_id": f"{oauth_provider}:{connection.id}",
            "provider_account_id": connection.provider_account_id
        }
    except Exception as e:
        logger.error(f"Error checking tool connection status: {e}")
        return {
            "tool_name": tool_name,
            "connected": False,
            "has_required_scopes": False,
            "missing_scopes": required_scopes,
            "provider": oauth_provider,
            "connection_id": None,
            "error": str(e)
        }


async def disconnect_provider(user: User, provider: str):
    """Disconnect all connections for a provider."""
    oauth_provider = get_oauth_provider(provider)
    try:
        # Soft delete (revoke) all connections for this provider
        await OAuthConnection.filter(user=user, provider=oauth_provider).update(status="revoked")
    except Exception as e:
        logger.error(f"Error disconnecting provider {provider} for user {user.user_id}: {e}")
        raise


async def delete_connection_by_id(user: User, connection_id: str):
    """Delete a specific connection by ID."""
    try:
        # connection_id might be "provider:id" or just "id"
        if ":" in connection_id:
            _, db_id = connection_id.split(":", 1)
        else:
            db_id = connection_id
            
        await OAuthConnection.filter(id=int(db_id), user=user).update(status="revoked")
    except Exception as e:
        logger.error(f"Error deleting connection {connection_id} for user {user.user_id}: {e}")
        raise
