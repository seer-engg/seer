from typing import Set, Optional, List
from datetime import datetime, timezone
from typing import Dict, Any

from seer.database import OAuthConnection, User
from seer.logger import get_logger
from seer.services.integrations.auth.oauth import get_oauth_provider


logger = get_logger(__name__)

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
