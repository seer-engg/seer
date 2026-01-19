from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

from seer.database import IntegrationResource, IntegrationSecret, OAuthConnection, User
from seer.logger import get_logger
from seer.tools.base import BaseTool
from seer.tools.oauth_manager import get_oauth_token
from seer.tools.scope_validator import validate_scopes

logger = get_logger("shared.tools.credential_resolver")

RESOURCE_ARGUMENT_KEYS = ("integration_resource_id", "resource_id", "resource_binding_id")


@dataclass
class ResolvedCredentials:
    connection: Optional[OAuthConnection] = None
    access_token: Optional[str] = None
    resource: Optional[IntegrationResource] = None
    secrets: Dict[str, str] = field(default_factory=dict)
    secret_records: Dict[str, IntegrationSecret] = field(default_factory=dict)


class CredentialResolver:
    """
    Resolves all runtime credentials (OAuth, resources, secrets) for tool execution.
    """

    def __init__(
        self,
        *,
        user: User,
        tool: BaseTool,
        connection_id: Optional[str] = None,
    ) -> None:
        self.user = user
        self.tool = tool
        self.connection_id = connection_id

    async def resolve(self, arguments: Dict[str, Any]) -> ResolvedCredentials:
        args = arguments or {}
        connection, access_token = await self._resolve_connection()
        resource = await self._resolve_resource(args, connection)
        secrets, secret_records = await self._resolve_secrets(resource, connection)
        return ResolvedCredentials(
            connection=connection,
            access_token=access_token,
            resource=resource,
            secrets=secrets,
            secret_records=secret_records,
        )

    async def _resolve_connection(self) -> Tuple[Optional[OAuthConnection], Optional[str]]:
        if not self.tool.required_scopes:
            return None, None

        if not self.user.user_id:
            raise HTTPException(
                status_code=401,
                detail=f"Tool '{self.tool.name}' requires OAuth authentication. User ID is required.",
            )

        provider = self._infer_provider()
        if not provider and not self.connection_id:
            raise HTTPException(
                status_code=400,
                detail=f"Tool '{self.tool.name}' requires OAuth connection. connection_id must be provided.",
            )

        connection, access_token = await get_oauth_token(
            self.user,
            connection_id=self.connection_id,
            provider=provider,
        )

        is_valid, missing_scope = validate_scopes(connection, self.tool.required_scopes)
        if not is_valid:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"OAuth connection {connection.id} missing required scope '{missing_scope}' "
                    f"for tool '{self.tool.name}'. Required scopes: {self.tool.required_scopes}"
                ),
            )

        return connection, access_token

    async def _resolve_resource(
        self,
        arguments: Dict[str, Any],
        connection: Optional[OAuthConnection],
    ) -> Optional[IntegrationResource]:
        resource_id = self._extract_resource_id(arguments)
        provider = self._infer_provider(connection)
        resource: Optional[IntegrationResource] = None

        if resource_id:
            resource = await IntegrationResource.get_or_none(
                id=resource_id,
                user=self.user,
                status="active",
            )
            if not resource:
                raise HTTPException(status_code=404, detail=f"Integration resource {resource_id} not found")
            if provider and resource.provider != provider:
                raise HTTPException(
                    status_code=400,
                    detail=f"Resource {resource_id} provider mismatch. Expected {provider}, got {resource.provider}",
                )
            if connection and resource.oauth_connection_id != connection.id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Resource {resource_id} does not belong to the selected OAuth connection",
                )
        elif self.tool.default_resource:
            resource = await self._find_default_resource(provider, connection)
            if self.tool.default_resource.get("required") and not resource:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Tool '{self.tool.name}' requires a persisted "
                        f"{self.tool.default_resource['resource_type']} resource, but none exist."
                    ),
                )

        return resource

    async def _resolve_secrets(
        self,
        resource: Optional[IntegrationResource],
        connection: Optional[OAuthConnection],
    ) -> Tuple[Dict[str, str], Dict[str, IntegrationSecret]]:
        if not self.tool.required_secrets:
            return {}, {}

        provider = self._infer_provider(connection, resource)
        if not provider:
            raise HTTPException(
                status_code=400,
                detail=f"Tool '{self.tool.name}' declares required secrets but no provider could be inferred.",
            )

        secrets: Dict[str, str] = {}
        secret_records: Dict[str, IntegrationSecret] = {}

        for name in self.tool.required_secrets:
            secret = await self._find_secret(provider, name, resource, connection)
            if not secret:
                raise HTTPException(
                    status_code=404,
                    detail=f"Missing required secret '{name}' for provider '{provider}'.",
                )
            secrets[name] = secret.value_enc
            secret_records[name] = secret

        return secrets, secret_records

    async def _find_secret(
        self,
        provider: str,
        name: str,
        resource: Optional[IntegrationResource],
        connection: Optional[OAuthConnection],
    ) -> Optional[IntegrationSecret]:
        filters = {
            "user": self.user,
            "provider": provider,
            "name": name,
            "status": "active",
        }

        if resource:
            secret = await IntegrationSecret.get_or_none(**filters, resource=resource)
            if secret:
                return secret

        if connection:
            secret = await IntegrationSecret.get_or_none(**filters, oauth_connection=connection)
            if secret:
                return secret

        return await IntegrationSecret.get_or_none(**filters, resource=None, oauth_connection=None)

    async def _find_default_resource(
        self,
        provider: Optional[str],
        connection: Optional[OAuthConnection],
    ) -> Optional[IntegrationResource]:
        config = self.tool.default_resource or {}
        resource_type = config.get("resource_type")
        if not resource_type:
            return None

        provider_name = config.get("provider") or provider
        if not provider_name:
            return None

        filters = {
            "user": self.user,
            "provider": provider_name,
            "resource_type": resource_type,
            "status": "active",
        }

        queryset = IntegrationResource.filter(**filters)
        if connection:
            queryset = queryset.filter(oauth_connection=connection)

        resource = await queryset.order_by("-updated_at").first()
        if resource:
            logger.info(
                "Resolved default resource",
                extra={"resource_id": resource.id, "provider": provider_name, "resource_type": resource_type},
            )
        return resource

    @staticmethod
    def _extract_resource_id(arguments: Dict[str, Any]) -> Optional[int]:
        for key in RESOURCE_ARGUMENT_KEYS:
            if key in arguments and arguments[key] is not None:
                try:
                    return int(arguments[key])
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid resource identifier '{arguments[key]}' for parameter '{key}'",
                    )
        return None

    def _infer_provider(
        self,
        connection: Optional[OAuthConnection] = None,
        resource: Optional[IntegrationResource] = None,
    ) -> Optional[str]:
        if resource and resource.provider:
            return resource.provider
        if connection and connection.provider:
            return connection.provider
        provider = getattr(self.tool, "provider", None) or getattr(self.tool, "integration_type", None)
        return provider


__all__ = ["CredentialResolver", "ResolvedCredentials"]
