from shared.database.models_oauth import OAuthConnection
from shared.database.models import User
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone, timedelta
from shared.logger import get_logger
logger = get_logger("api.integrations.services")


def parse_scopes(scopes_str: str) -> Set[str]:
    """
    Parse a scopes string into a set of individual scopes.
    Handles both whitespace-separated (Google) and comma-separated (GitHub) formats.
    
    Args:
        scopes_str: String containing scopes (either whitespace or comma separated)
    
    Returns:
        Set of individual scope strings
    """
    if not scopes_str:
        return set()
    
    # If scopes contain commas, split by comma; otherwise split by whitespace
    if ',' in scopes_str:
        return set(s.strip() for s in scopes_str.split(',') if s.strip())
    else:
        return set(scopes_str.split())


def merge_scopes(existing_scopes: str, new_scopes: str) -> str:
    """
    Merge existing scopes with new scopes, removing duplicates.
    Handles both whitespace-separated (Google) and comma-separated (GitHub) formats.
    
    Args:
        existing_scopes: String of existing scopes (whitespace or comma separated)
        new_scopes: String of new scopes to add (whitespace or comma separated)
    
    Returns:
        Space-separated string of merged scopes (normalized to whitespace-separated)
    """
    existing_set = parse_scopes(existing_scopes)
    new_set = parse_scopes(new_scopes)
    merged = existing_set | new_set
    return " ".join(sorted(merged))


def _extract_base_google_scope(scope: str) -> Optional[str]:
    """
    Extract base scope from a Google API scope by removing common suffixes.
    
    For Google APIs, broader scopes include narrower ones:
    - gmail includes gmail.readonly, gmail.modify, gmail.send, etc.
    - drive includes drive.readonly, drive.file, etc.
    - spreadsheets includes spreadsheets.readonly, etc.
    
    Args:
        scope: Full scope string (e.g., "https://www.googleapis.com/auth/gmail.readonly")
    
    Returns:
        Base scope string (e.g., "https://www.googleapis.com/auth/gmail") or None if not a Google scope
    """
    if "googleapis.com" not in scope:
        return None
    
    # Common Google scope suffixes to remove
    suffixes = [".readonly", ".modify", ".send", ".compose", ".labels", ".file", ".metadata"]
    
    base_scope = scope
    for suffix in suffixes:
        if scope.endswith(suffix):
            base_scope = scope[:-len(suffix)]
            break
    
    return base_scope if base_scope != scope else None


def _scope_satisfies_requirement(granted_scope: str, required_scope: str) -> bool:
    """
    Check if a granted scope satisfies a required scope, handling Google scope hierarchy.
    
    Hierarchy rules:
    - Base scope (e.g., "gmail") satisfies all narrower scopes (e.g., "gmail.readonly", "gmail.modify")
    - Narrower scopes do NOT satisfy broader scopes or other narrower scopes
    
    Args:
        granted_scope: Scope that user has (e.g., "https://www.googleapis.com/auth/gmail")
        required_scope: Scope that is required (e.g., "https://www.googleapis.com/auth/gmail.readonly")
    
    Returns:
        True if granted scope satisfies required scope
    """
    # Exact match always satisfies
    if granted_scope == required_scope:
        return True
    
    # For Google APIs, check hierarchy
    if "googleapis.com" in required_scope and "googleapis.com" in granted_scope:
        # Extract base scope from required scope
        base_required = _extract_base_google_scope(required_scope)
        if base_required:
            # Check if granted scope is the base scope (broader satisfies narrower)
            # This handles: granted="gmail", required="gmail.readonly" -> True
            if granted_scope == base_required:
                return True
        
        # Check if required scope is a base scope and granted scope is narrower
        # This handles: granted="gmail.readonly", required="gmail" -> False (narrower doesn't satisfy broader)
        base_granted = _extract_base_google_scope(granted_scope)
        if base_granted and not _extract_base_google_scope(required_scope):
            # Required is base scope, granted is narrower -> doesn't satisfy
            return False
    
    return False


def has_required_scopes(granted_scopes: str, required_scopes: List[str]) -> bool:
    """
    Check if granted scopes include all required scopes.
    Handles both whitespace-separated (Google) and comma-separated (GitHub) formats.
    For Google APIs, handles scope hierarchy where broader scopes satisfy narrower ones.
    
    Args:
        granted_scopes: String of granted scopes (whitespace or comma separated)
        required_scopes: List of required scope strings
    
    Returns:
        True if all required scopes are granted (or satisfied by broader scopes for Google APIs)
    
    Examples:
        - has_required_scopes("gmail", ["gmail.readonly"]) -> True (broader satisfies narrower)
        - has_required_scopes("gmail.readonly", ["gmail"]) -> False (narrower doesn't satisfy broader)
        - has_required_scopes("gmail.readonly", ["gmail.readonly"]) -> True (exact match)
    """
    if not required_scopes:
        return True
    
    granted_set = parse_scopes(granted_scopes)
    
    # Check each required scope
    for required_scope in required_scopes:
        # First check for exact match
        if required_scope in granted_set:
            continue
        
        # For Google APIs, check if any granted scope satisfies the requirement via hierarchy
        if "googleapis.com" in required_scope:
            satisfied = False
            for granted_scope in granted_set:
                if _scope_satisfies_requirement(granted_scope, required_scope):
                    satisfied = True
                    break
            if not satisfied:
                return False
        else:
            # For non-Google providers, require exact match
            return False
    
    return True


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


def extract_provider_account_id(oauth_provider: str, profile: Dict[str, Any]) -> str:
    """
    Extract provider_account_id from profile.
    Raises ValueError if required fields are missing.
    
    Args:
        oauth_provider: OAuth provider name (google, github, etc.)
        profile: User profile dictionary from OAuth provider
    
    Returns:
        provider_account_id string
    
    Raises:
        ValueError: If required fields are missing from profile
    """
    if oauth_provider == 'google':
        provider_account_id = profile.get('sub') or profile.get('email')
        if not provider_account_id:
            raise ValueError(
                f"Google profile missing required fields 'sub' or 'email'. "
                f"Profile keys: {list(profile.keys())}"
            )
        return provider_account_id
    elif oauth_provider == 'github':
        provider_id = profile.get('id')
        if provider_id is None:
            raise ValueError(
                f"GitHub profile missing required field 'id'. "
                f"Profile keys: {list(profile.keys())}"
            )
        return str(provider_id)
    else:
        provider_id = profile.get('id')
        if provider_id is None:
            raise ValueError(
                f"{oauth_provider} profile missing required field 'id'. "
                f"Profile keys: {list(profile.keys())}"
            )
        return str(provider_id)


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
    provider_account_id = extract_provider_account_id(oauth_provider, profile)
        
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
        
        # Find missing scopes (using parse_scopes to handle both comma and whitespace separators)
        granted_set = parse_scopes(granted_scopes)
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


async def get_valid_access_token(user: User, provider: str) -> Optional[str]:
    """
    Get a valid access token for a provider, refreshing if needed.
    
    Args:
        user: User model instance
        provider: OAuth provider name (google, github, etc.)
    
    Returns:
        Valid access token or None if no connection exists
    """
    from .oauth import oauth
    import httpx
    
    oauth_provider = get_oauth_provider(provider)
    connection = await get_connection_for_provider(user, oauth_provider)
    
    if not connection:
        return None
    
    # Check if token is expired
    if connection.expires_at:
        now = datetime.now(timezone.utc)
        # Add a 5-minute buffer to refresh before expiration
        if now >= connection.expires_at - timedelta(minutes=5):
            # Token is expired or about to expire, try to refresh
            if connection.refresh_token_enc:
                try:
                    # Refresh the token using authlib
                    client = oauth.create_client(oauth_provider)
                    
                    if oauth_provider == 'google':
                        # Google token refresh
                        async with httpx.AsyncClient() as http_client:
                            response = await http_client.post(
                                'https://oauth2.googleapis.com/token',
                                data={
                                    'client_id': client.client_id,
                                    'client_secret': client.client_secret,
                                    'refresh_token': connection.refresh_token_enc,
                                    'grant_type': 'refresh_token',
                                }
                            )
                            
                            if response.status_code == 200:
                                token_data = response.json()
                                connection.access_token_enc = token_data.get('access_token')
                                if 'expires_in' in token_data:
                                    connection.expires_at = datetime.now(timezone.utc) + timedelta(seconds=token_data['expires_in'])
                                connection.updated_at = datetime.now(timezone.utc)
                                await connection.save()
                                logger.info(f"Refreshed access token for {oauth_provider}")
                            else:
                                logger.error(f"Failed to refresh token: {response.status_code} - {response.text[:200]}")
                                return None
                    else:
                        # For other providers, implement as needed
                        logger.warning(f"Token refresh not implemented for provider: {oauth_provider}")
                        return connection.access_token_enc
                        
                except Exception as e:
                    logger.error(f"Error refreshing token for {oauth_provider}: {e}")
                    return None
            else:
                logger.warning(f"Token expired and no refresh token available for {oauth_provider}")
                return None
    
    return connection.access_token_enc
