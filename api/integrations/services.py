from shared.database.models_oauth import OAuthConnection
from shared.database.models import User
from typing import Dict, Any, List
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

async def store_oauth_connection(
    user_id: str,
    provider: str,
    token: Dict[str, Any],
    profile: Dict[str, Any],
    granted_scopes: str = ""
):
    """
    Store OAuth connection with granted scopes.
    
    Args:
        user_id: User ID
        provider: Provider name
        token: OAuth token response dict
        profile: User profile information
        granted_scopes: Space-separated string of granted OAuth scopes
    """
    # Find user
    user = await User.get(email=user_id)
    
    # Extract provider account id
    if provider in ['google', 'googledrive', 'gmail']:
        provider_account_id = profile.get('sub') or profile.get('email')
    elif provider == 'github':
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
    
    # Update or Create
    connection = await OAuthConnection.get_or_none(
        user=user,
        provider=provider,
        provider_account_id=provider_account_id
    )
    
    if connection:
        connection.access_token_enc = access_token
        if refresh_token:
            connection.refresh_token_enc = refresh_token
        connection.provider_metadata = provider_metadata
        connection.status = "active"
        connection.expires_at = expires_at
        connection.scopes = granted_scopes  # Store granted scopes
        connection.token_type = token_type
        connection.updated_at = datetime.now(timezone.utc)
        await connection.save()
    else:
        connection = await OAuthConnection.create(
            user=user,
            provider=provider,
            provider_account_id=provider_account_id,
            access_token_enc=access_token,
            refresh_token_enc=refresh_token,
            provider_metadata=provider_metadata,
            status="active",
            expires_at=expires_at,
            scopes=granted_scopes,  # Store granted scopes
            token_type=token_type
        )
        
    return connection

async def list_connections(user_id: str):
    # Using user_id (string) to find User model
    try:
        user = await User.get(email=user_id)
        logger.info(f"Listing connections for user {user_id}")
        connections = await OAuthConnection.filter(user=user, status="active").all()
        return connections
    except Exception as e:
        logger.error(f"Error listing connections for user {user_id}: {e}")
        return []

async def disconnect_provider(user_id: str, provider: str):
    try:
        user = await User.get(user_id=user_id)
        # Soft delete (revoke) all connections for this provider
        await OAuthConnection.filter(user=user, provider=provider).update(status="revoked")
    except Exception as e:
        logger.error(f"Error disconnecting provider {provider} for user {user_id}: {e}")
        raise

async def delete_connection_by_id(user_id: str, connection_id: str):
    try:
        user = await User.get(user_id=user_id)
        # connection_id might be "provider:id" or just "id"
        if ":" in connection_id:
            _, db_id = connection_id.split(":")
        else:
            db_id = connection_id
            
        await OAuthConnection.filter(id=int(db_id), user=user).update(status="revoked")
    except Exception as e:
        logger.error(f"Error deleting connection {connection_id} for user {user_id}: {e}")
        raise
