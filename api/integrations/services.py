import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException

from api.integrations.constants import (
    SUPABASE_OAUTH_PROVIDER,
    SUPABASE_RESOURCE_PROVIDER,
    SUPABASE_RESOURCE_TYPE_PROJECT,
)
from api.integrations.providers import ProviderContext, get_integration_provider
from shared.database.models import User
from shared.database.models_integrations import IntegrationResource, IntegrationSecret
from shared.database.models_oauth import OAuthConnection
from shared.logger import get_logger
from shared.tools.oauth_manager import get_oauth_token

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
    if integration_type in ['supabase', 'supabase_mgmt']:
        return SUPABASE_OAUTH_PROVIDER
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
    oauth_provider = get_oauth_provider(provider)
    try:
        _, access_token = await get_oauth_token(user, provider=oauth_provider)
        return access_token
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


# =============================================================================
# Integration Resource Helpers
# =============================================================================

def serialize_integration_resource(resource: IntegrationResource) -> Dict[str, Any]:
    return {
        "id": resource.id,
        "provider": resource.provider,
        "resource_type": resource.resource_type,
        "resource_id": resource.resource_id,
        "resource_key": resource.resource_key,
        "name": resource.name,
        "status": resource.status,
        "metadata": resource.resource_metadata or {},
        "oauth_connection_id": resource.oauth_connection_id,
        "created_at": resource.created_at.isoformat() if resource.created_at else None,
        "updated_at": resource.updated_at.isoformat() if resource.updated_at else None,
    }


def serialize_integration_secret(secret: IntegrationSecret) -> Dict[str, Any]:
    return {
        "id": secret.id,
        "provider": secret.provider,
        "name": secret.name,
        "secret_type": secret.secret_type,
        "resource_id": secret.resource_id,
        "oauth_connection_id": secret.oauth_connection_id,
        "value_fingerprint": secret.value_fingerprint,
        "metadata": secret.metadata or {},
        "status": secret.status,
        "expires_at": secret.expires_at.isoformat() if secret.expires_at else None,
        "created_at": secret.created_at.isoformat() if secret.created_at else None,
        "updated_at": secret.updated_at.isoformat() if secret.updated_at else None,
    }


async def list_integration_resources(
    user: User,
    *,
    provider: Optional[str] = None,
    resource_type: Optional[str] = None,
) -> List[IntegrationResource]:
    queryset = IntegrationResource.filter(user=user, status="active")
    if provider:
        queryset = queryset.filter(provider=provider)
    if resource_type:
        queryset = queryset.filter(resource_type=resource_type)
    return await queryset.order_by("-updated_at")


async def list_resource_secrets(user: User, resource_id: int) -> List[IntegrationSecret]:
    resource = await IntegrationResource.get_or_none(id=resource_id, user=user)
    if not resource:
        raise HTTPException(status_code=404, detail=f"Integration resource {resource_id} not found")
    return await IntegrationSecret.filter(user=user, resource=resource, status="active").order_by("-updated_at")


async def deactivate_integration_resource(user: User, resource_id: int) -> IntegrationResource:
    resource = await IntegrationResource.get_or_none(id=resource_id, user=user)
    if not resource:
        raise HTTPException(status_code=404, detail=f"Integration resource {resource_id} not found")
    resource.status = "revoked"
    await resource.save(update_fields=["status", "updated_at"])
    await IntegrationSecret.filter(resource=resource, user=user).update(status="revoked")
    return resource


async def _upsert_integration_resource(
    *,
    user: User,
    oauth_connection: Optional[OAuthConnection],
    provider: str,
    resource_type: str,
    resource_id: str,
    resource_key: Optional[str],
    name: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> IntegrationResource:
    defaults = {
        "resource_key": resource_key,
        "name": name,
        "resource_metadata": metadata or {},
        "status": "active",
    }
    lookup_filters = {
        "user": user,
        "provider": provider,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "oauth_connection": oauth_connection,
    }
    resource = await IntegrationResource.get_or_none(**lookup_filters)
    if resource:
        update_fields: List[str] = []
        for field, value in defaults.items():
            if getattr(resource, field) != value:
                setattr(resource, field, value)
                update_fields.append(field)
        if update_fields:
            update_fields.append("updated_at")
            await resource.save(update_fields=update_fields)
        return resource

    return await IntegrationResource.create(
        user=user,
        oauth_connection=oauth_connection,
        provider=provider,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_key=resource_key,
        name=name,
        resource_metadata=metadata or {},
        status="active",
    )


def _fingerprint_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _format_supabase_secret_name(raw_name: str) -> str:
    mapping = {
        "service_role": "supabase_service_role_key",
        "service-role": "supabase_service_role_key",
        "service": "supabase_service_role_key",
        "anon": "supabase_anon_key",
        "anon_key": "supabase_anon_key",
    }
    normalized = (raw_name or "").strip().lower()
    if not normalized:
        return "supabase_custom_key"
    return mapping.get(normalized, f"supabase_{normalized}_key")


def _build_manual_supabase_metadata(
    *,
    project_ref: str,
    project_name: Optional[str],
) -> Dict[str, Any]:
    base_url = f"https://{project_ref}.supabase.co"
    metadata: Dict[str, Any] = {
        "project_ref": project_ref,
        "binding_mode": "manual",
        "name": project_name or project_ref,
        "rest_url": f"{base_url}/rest/v1",
        "auth_url": f"{base_url}/auth/v1",
        "storage_url": f"{base_url}/storage/v1",
        "functions_url": f"{base_url}/functions/v1",
    }
    return metadata


async def _upsert_integration_secret(
    *,
    user: User,
    provider: str,
    name: str,
    secret_type: str,
    value_enc: str,
    resource: Optional[IntegrationResource] = None,
    oauth_connection: Optional[OAuthConnection] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> IntegrationSecret:
    defaults = {
        "secret_type": secret_type,
        "value_enc": value_enc,
        "value_fingerprint": _fingerprint_secret(value_enc),
        "metadata": metadata or {},
        "status": "active",
    }
    secret, created = await IntegrationSecret.get_or_create(
        user=user,
        provider=provider,
        name=name,
        resource=resource,
        oauth_connection=oauth_connection,
        defaults=defaults,
    )
    if created:
        return secret

    update_fields: List[str] = []
    for field, value in defaults.items():
        if getattr(secret, field) != value:
            setattr(secret, field, value)
            update_fields.append(field)
    if update_fields:
        update_fields.append("updated_at")
        await secret.save(update_fields=update_fields)
    return secret


# =============================================================================
# Provider Dispatch Helpers
# =============================================================================


def _build_provider_context() -> ProviderContext:
    return ProviderContext(
        upsert_resource=_upsert_integration_resource,
        upsert_secret=_upsert_integration_secret,
    )


def _require_provider(provider_name: str):
    provider = get_integration_provider(provider_name)
    if not provider:
        raise HTTPException(status_code=500, detail=f"Integration provider '{provider_name}' is not configured")
    return provider


async def bind_supabase_project(
    user: User,
    project_ref: str,
    connection_id: Optional[str] = None,
) -> IntegrationResource:
    provider = _require_provider(SUPABASE_RESOURCE_PROVIDER)
    return await provider.bind_resource(
        context=_build_provider_context(),
        user=user,
        resource_type=SUPABASE_RESOURCE_TYPE_PROJECT,
        project_ref=project_ref,
        connection_id=connection_id,
    )


async def bind_supabase_project_manual(
    user: User,
    *,
    project_ref: str,
    service_role_key: str,
    project_name: Optional[str] = None,
    anon_key: Optional[str] = None,
) -> IntegrationResource:
    normalized_ref = (project_ref or "").strip()
    if not normalized_ref:
        raise HTTPException(status_code=400, detail="project_ref is required")
    if not service_role_key:
        raise HTTPException(status_code=400, detail="service_role_key is required for manual binding")

    resource_metadata = _build_manual_supabase_metadata(
        project_ref=normalized_ref,
        project_name=project_name,
    )

    resource = await _upsert_integration_resource(
        user=user,
        oauth_connection=None,
        provider=SUPABASE_RESOURCE_PROVIDER,
        resource_type=SUPABASE_RESOURCE_TYPE_PROJECT,
        resource_id=normalized_ref,
        resource_key=normalized_ref,
        name=project_name or resource_metadata.get("name") or normalized_ref,
        metadata=resource_metadata,
    )

    await _upsert_integration_secret(
        user=user,
        provider=SUPABASE_RESOURCE_PROVIDER,
        name="supabase_service_role_key",
        secret_type="api_key",
        value_enc=service_role_key,
        resource=resource,
        metadata={"binding_mode": "manual"},
    )

    if anon_key:
        await _upsert_integration_secret(
            user=user,
            provider=SUPABASE_RESOURCE_PROVIDER,
            name="supabase_anon_key",
            secret_type="api_key",
            value_enc=anon_key,
            resource=resource,
            metadata={"binding_mode": "manual"},
        )

    return resource
