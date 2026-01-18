# pylint: disable=too-many-lines,too-many-positional-arguments,too-many-arguments
# Reason: Integration services handles OAuth, resource/secret management, and provider operations.
# High argument counts are necessary for resource/secret creation with metadata and encryption.
# TODO: Split into separate service modules (oauth_service.py, resource_service.py) in future refactor.

# pylint: disable=no-else-return,broad-exception-caught
# Reason: Else-after-return pattern used for clarity in OAuth provider mapping logic.
# Broad exception catching is intentional for logging and graceful degradation.

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException

from seer.services.integrations.constants import (
    SUPABASE_OAUTH_PROVIDER,
    SUPABASE_RESOURCE_PROVIDER,
    SUPABASE_RESOURCE_TYPE_PROJECT,
)
from seer.services.integrations.providers import ProviderContext, get_integration_provider
from seer.database import (
    User,
    IntegrationResource, IntegrationSecret, OAuthConnection
)
from seer.logger import get_logger
from seer.tools.oauth_manager import get_oauth_token
from seer.services.integrations.auth.oauth import get_oauth_provider

logger = get_logger("api.integrations.services")






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
    """Disconnect all connections for a provider and cascade to related resources."""
    oauth_provider = get_oauth_provider(provider)
    try:
        # Get all active connections for this provider before revoking
        connections = await OAuthConnection.filter(user=user, provider=oauth_provider, status="active")

        # Soft delete (revoke) all connections for this provider
        await OAuthConnection.filter(user=user, provider=oauth_provider).update(status="revoked")

        # For each revoked connection, cascade to resources and secrets
        for connection in connections:
            # Find and revoke all resources linked to this connection
            linked_resources = await IntegrationResource.filter(
                user=user,
                oauth_connection=connection,
                status="active"
            )

            for resource in linked_resources:
                # Use existing deactivate logic (cascades to secrets)
                await deactivate_integration_resource(user, resource.id)

            # Revoke secrets directly tied to connection (not via resource)
            await IntegrationSecret.filter(
                user=user,
                oauth_connection=connection,
                resource_id__isnull=True,
                status="active"
            ).update(status="revoked")

        logger.info(
            f"Revoked {len(connections)} connections for provider {oauth_provider} (user {user.user_id})"
        )
    except Exception as e:
        logger.error(f"Error disconnecting provider {provider} for user {user.user_id}: {e}")
        raise


async def delete_connection_by_id(user: User, connection_id: str):
    """Delete a specific connection by ID and cascade to related resources."""
    try:
        # connection_id might be "provider:id" or just "id"
        if ":" in connection_id:
            _, db_id = connection_id.split(":", 1)
        else:
            db_id = connection_id

        connection = await OAuthConnection.get_or_none(id=int(db_id), user=user)
        if not connection:
            raise HTTPException(status_code=404, detail="Connection not found")

        # 1. Revoke the OAuth connection
        await OAuthConnection.filter(id=int(db_id), user=user).update(status="revoked")

        # 2. Find and revoke all resources linked to this connection
        linked_resources = await IntegrationResource.filter(
            user=user,
            oauth_connection=connection,
            status="active"
        )

        for resource in linked_resources:
            # Use existing deactivate logic (cascades to secrets)
            await deactivate_integration_resource(user, resource.id)

        # 3. Revoke secrets directly tied to connection (not via resource)
        await IntegrationSecret.filter(
            user=user,
            oauth_connection=connection,
            resource_id__isnull=True,  # Only secrets directly on connection
            status="active"
        ).update(status="revoked")

        logger.info(
            f"Revoked connection {db_id} with {len(linked_resources)} resources for user {user.user_id}"
        )

    except HTTPException:
        raise
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
