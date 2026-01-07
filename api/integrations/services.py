import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException
from sqlmodel import select

from api.integrations.constants import (
    SUPABASE_OAUTH_PROVIDER,
    SUPABASE_RESOURCE_PROVIDER,
    SUPABASE_RESOURCE_TYPE_PROJECT,
)
from api.integrations.providers import ProviderContext, get_integration_provider
from shared.database.models import User, IntegrationResource, IntegrationSecret, OAuthConnection
from shared.database.base import async_session_maker
from shared.logger import get_logger
from shared.tools.oauth_manager import get_oauth_token
logger = get_logger("api.integrations.services")


def build_provider_connection_info(conn) -> dict:
    """Build provider connection info dict with token validity checks."""
    has_access_token = bool(conn.access_token_enc)
    has_refresh_token = bool(conn.refresh_token_enc)
    is_token_expired = conn.expires_at < datetime.now(timezone.utc) if conn.expires_at else False
    access_token_valid = (has_access_token and not is_token_expired) or has_refresh_token
    return {
        "scopes": conn.scopes or "",
        "connection_id": f"{conn.provider}:{conn.id}",
        "provider_account_id": conn.provider_account_id,
        "has_refresh_token": has_refresh_token,
        "access_token_valid": access_token_valid,
        "connection": conn
    }


def determine_auth_mode(required_scopes: list, required_secrets: list) -> tuple:
    """Determine auth mode for a tool. Returns (auth_mode, requires_oauth, requires_secrets, supports_tokenless_auth)."""
    requires_oauth = bool(required_scopes)
    requires_secrets = bool(required_secrets)
    supports_tokenless_auth = not requires_oauth
    if requires_oauth and requires_secrets:
        auth_mode = "oauth_and_secrets"
    elif requires_oauth:
        auth_mode = "oauth"
    elif requires_secrets:
        auth_mode = "secrets"
    else:
        auth_mode = "none"
    return auth_mode, requires_oauth, requires_secrets, supports_tokenless_auth


def build_tool_result_base(tool, auth_mode: str, requires_oauth: bool, requires_secrets: bool, supports_tokenless_auth: bool) -> dict:
    """Build base tool result dict."""
    return {
        "tool_name": tool.name,
        "integration_type": tool.integration_type,
        "requires_oauth_connection": requires_oauth,
        "requires_secrets": requires_secrets,
        "supports_tokenless_auth": supports_tokenless_auth,
        "auth_mode": auth_mode,
    }


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

    async with async_session_maker() as session:
        # Find user
        result = await session.execute(
            select(User).where(User.user_id == user_id)
        )
        user = result.scalar_one()

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
        result = await session.execute(
            select(OAuthConnection).where(
                OAuthConnection.user_id == user.id,
                OAuthConnection.provider == oauth_provider,
                OAuthConnection.provider_account_id == provider_account_id
            )
        )
        connection = result.scalar_one_or_none()

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
            session.add(connection)
            await session.commit()
            await session.refresh(connection)
            logger.info(f"Updated existing connection for {oauth_provider}, merged scopes: {connection.scopes[:100]}...")
        else:
            connection = OAuthConnection(
                user_id=user.id,
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
            session.add(connection)
            await session.commit()
            await session.refresh(connection)
            logger.info(f"Created new connection for {oauth_provider}")

        return connection

async def list_connections(user: User):
    """
    List all active OAuth connections for a user.
    """
    try:
        logger.info(f"Listing connections for user {user.user_id}")
        async with async_session_maker() as session:
            result = await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.user_id == user.id,
                    OAuthConnection.status == "active"
                )
            )
            connections = list(result.scalars().all())
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
        async with async_session_maker() as session:
            result = await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.user_id == user.id,
                    OAuthConnection.provider == oauth_provider,
                    OAuthConnection.status == "active"
                )
            )
            connection = result.scalar_one_or_none()
            return connection
    except Exception as e:
        logger.error(f"Error getting connection for provider {provider}: {e}")
        return None


async def disconnect_provider(user: User, provider: str):
    """Disconnect all connections for a provider."""
    oauth_provider = get_oauth_provider(provider)
    try:
        async with async_session_maker() as session:
            # Soft delete (revoke) all connections for this provider
            result = await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.user_id == user.id,
                    OAuthConnection.provider == oauth_provider
                )
            )
            connections = result.scalars().all()
            for connection in connections:
                connection.status = "revoked"
                session.add(connection)
            await session.commit()
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

        async with async_session_maker() as session:
            result = await session.execute(
                select(OAuthConnection).where(
                    OAuthConnection.id == int(db_id),
                    OAuthConnection.user_id == user.id
                )
            )
            connection = result.scalar_one_or_none()
            if connection:
                connection.status = "revoked"
                session.add(connection)
                await session.commit()
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
    async with async_session_maker() as session:
        stmt = select(IntegrationResource).where(
            IntegrationResource.user_id == user.id,
            IntegrationResource.status == "active"
        )
        if provider:
            stmt = stmt.where(IntegrationResource.provider == provider)
        if resource_type:
            stmt = stmt.where(IntegrationResource.resource_type == resource_type)
        stmt = stmt.order_by(IntegrationResource.updated_at.desc())

        result = await session.execute(stmt)
        return list(result.scalars().all())


async def list_resource_secrets(user: User, resource_id: int) -> List[IntegrationSecret]:
    async with async_session_maker() as session:
        # Check resource exists
        result = await session.execute(
            select(IntegrationResource).where(
                IntegrationResource.id == resource_id,
                IntegrationResource.user_id == user.id
            )
        )
        resource = result.scalar_one_or_none()
        if not resource:
            raise HTTPException(status_code=404, detail=f"Integration resource {resource_id} not found")

        # Get secrets
        result = await session.execute(
            select(IntegrationSecret).where(
                IntegrationSecret.user_id == user.id,
                IntegrationSecret.resource_id == resource_id,
                IntegrationSecret.status == "active"
            ).order_by(IntegrationSecret.updated_at.desc())
        )
        return list(result.scalars().all())


async def deactivate_integration_resource(user: User, resource_id: int) -> IntegrationResource:
    async with async_session_maker() as session:
        result = await session.execute(
            select(IntegrationResource).where(
                IntegrationResource.id == resource_id,
                IntegrationResource.user_id == user.id
            )
        )
        resource = result.scalar_one_or_none()
        if not resource:
            raise HTTPException(status_code=404, detail=f"Integration resource {resource_id} not found")

        resource.status = "revoked"
        resource.updated_at = datetime.now(timezone.utc)
        session.add(resource)

        # Update related secrets
        result = await session.execute(
            select(IntegrationSecret).where(
                IntegrationSecret.resource_id == resource_id,
                IntegrationSecret.user_id == user.id
            )
        )
        secrets = result.scalars().all()
        for secret in secrets:
            secret.status = "revoked"
            session.add(secret)

        await session.commit()
        await session.refresh(resource)
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
    async with async_session_maker() as session:
        # Try to find existing resource
        stmt = select(IntegrationResource).where(
            IntegrationResource.user_id == user.id,
            IntegrationResource.provider == provider,
            IntegrationResource.resource_type == resource_type,
            IntegrationResource.resource_id == resource_id
        )
        if oauth_connection:
            stmt = stmt.where(IntegrationResource.oauth_connection_id == oauth_connection.id)
        else:
            stmt = stmt.where(IntegrationResource.oauth_connection_id.is_(None))

        result = await session.execute(stmt)
        resource = result.scalar_one_or_none()

        if resource:
            # Update existing resource
            needs_update = False
            if resource.resource_key != resource_key:
                resource.resource_key = resource_key
                needs_update = True
            if resource.name != name:
                resource.name = name
                needs_update = True
            if resource.resource_metadata != (metadata or {}):
                resource.resource_metadata = metadata or {}
                needs_update = True
            if resource.status != "active":
                resource.status = "active"
                needs_update = True

            if needs_update:
                resource.updated_at = datetime.now(timezone.utc)
                session.add(resource)
                await session.commit()
                await session.refresh(resource)
            return resource

        # Create new resource
        resource = IntegrationResource(
            user_id=user.id,
            oauth_connection_id=oauth_connection.id if oauth_connection else None,
            provider=provider,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_key=resource_key,
            name=name,
            resource_metadata=metadata or {},
            status="active",
        )
        session.add(resource)
        await session.commit()
        await session.refresh(resource)
        return resource


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
    async with async_session_maker() as session:
        # Try to find existing secret
        stmt = select(IntegrationSecret).where(
            IntegrationSecret.user_id == user.id,
            IntegrationSecret.provider == provider,
            IntegrationSecret.name == name
        )
        if resource:
            stmt = stmt.where(IntegrationSecret.resource_id == resource.id)
        else:
            stmt = stmt.where(IntegrationSecret.resource_id.is_(None))
        if oauth_connection:
            stmt = stmt.where(IntegrationSecret.oauth_connection_id == oauth_connection.id)
        else:
            stmt = stmt.where(IntegrationSecret.oauth_connection_id.is_(None))

        result = await session.execute(stmt)
        secret = result.scalar_one_or_none()

        fingerprint = _fingerprint_secret(value_enc)

        if secret:
            # Update existing secret
            needs_update = False
            if secret.secret_type != secret_type:
                secret.secret_type = secret_type
                needs_update = True
            if secret.value_enc != value_enc:
                secret.value_enc = value_enc
                needs_update = True
            if secret.value_fingerprint != fingerprint:
                secret.value_fingerprint = fingerprint
                needs_update = True
            if secret.metadata != (metadata or {}):
                secret.metadata = metadata or {}
                needs_update = True
            if secret.status != "active":
                secret.status = "active"
                needs_update = True

            if needs_update:
                secret.updated_at = datetime.now(timezone.utc)
                session.add(secret)
                await session.commit()
                await session.refresh(secret)
            return secret

        # Create new secret
        secret = IntegrationSecret(
            user_id=user.id,
            provider=provider,
            name=name,
            secret_type=secret_type,
            value_enc=value_enc,
            value_fingerprint=fingerprint,
            metadata=metadata or {},
            status="active",
            resource_id=resource.id if resource else None,
            oauth_connection_id=oauth_connection.id if oauth_connection else None,
        )
        session.add(secret)
        await session.commit()
        await session.refresh(secret)
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
