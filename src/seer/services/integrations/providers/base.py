from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
)

from fastapi import HTTPException

if TYPE_CHECKING:
    from seer.database import (
        User,
        IntegrationResource,
        IntegrationSecret,
        OAuthConnection
    )


class _ResourceUpserter(Protocol):
    # pylint: disable=too-many-arguments
    # Reason: Resource upsert requires all metadata fields to persist integration resources
    def __call__(
        self,
        *,
        user: "User",
        oauth_connection: Optional["OAuthConnection"],
        provider: str,
        resource_type: str,
        resource_id: str,
        resource_key: Optional[str],
        name: Optional[str],
        metadata: Optional[Dict[str, object]],
    ) -> Awaitable["IntegrationResource"]:
        ...


class _SecretUpserter(Protocol):
    # pylint: disable=too-many-arguments
    # Reason: Secret upsert requires full context for encryption, linking, and metadata
    def __call__(
        self,
        *,
        user: "User",
        provider: str,
        name: str,
        secret_type: str,
        value_enc: str,
        resource: Optional["IntegrationResource"],
        oauth_connection,
        metadata: Optional[Dict[str, object]],
    ) -> Awaitable["IntegrationSecret"]:
        ...


@dataclass
class ProviderContext:
    """Holds callbacks the providers need for persisting resources/secrets."""

    upsert_resource: _ResourceUpserter
    upsert_secret: _SecretUpserter


@dataclass
class OAuthHelpers:
    """Helper callbacks providers can use for OAuth logic."""

    has_required_scopes: Callable[[str, List[str]], bool]


@dataclass
class OAuthAuthorizeContext:
    """Context shared with providers during /connect handling."""

    user: "User"
    oauth_provider: str
    integration_type: Optional[str]
    requested_scopes: List[str]
    existing_connection: Optional["OAuthConnection"]
    helpers: OAuthHelpers


class IntegrationProvider:
    """Base provider with optional overrides for supported operations."""

    provider: str
    resource_types: Set[str] = set()
    aliases: Set[str] = set()

    async def bind_resource(
        self,
        context: ProviderContext,
        user: "User",
        resource_type: str,
        **kwargs,
    ):
        raise HTTPException(status_code=400, detail=f"{self.provider} does not support binding {resource_type}")

    async def list_remote_resources(
        self,
        access_token: str,
        resource_type: str,
        **kwargs,
    ) -> List[Dict[str, object]]:
        raise HTTPException(status_code=400, detail=f"{self.provider} does not expose remote listing for {resource_type}")

    def supports_provider(self, provider_name: str) -> bool:
        return provider_name == self.provider or provider_name in self.aliases

    def get_supported_resource_types(self) -> Set[str]:
        return self.resource_types

    # -------------------------------------------------------------------------
    # OAuth lifecycle hooks (optional)
    # -------------------------------------------------------------------------

    def get_oauth_scope(self, context: OAuthAuthorizeContext) -> str:
        """Return the scope string that should be sent to the provider."""
        return " ".join(context.requested_scopes)

    def build_authorize_kwargs(
        self,
        context: OAuthAuthorizeContext,
        *,
        state: str,
        scope: str,
    ) -> Dict[str, Any]:
        """Customize authorize_redirect kwargs."""
        _ = context  # Context available for overrides; unused in base implementation
        return {"state": state, "scope": scope}

    # pylint: disable=unused-argument
    # Reason: Base class stub - parameters are used by provider-specific overrides
    async def introspect_token(
        self,
        *,
        access_token: str,
        client_id: str,
        client_secret: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Call provider's token introspection endpoint to get actual granted scopes.

        Override in providers that support token introspection (e.g., LinkedIn, Google).
        Returns the introspection response dict on success, or None on failure.

        This method should handle all errors gracefully and return None rather than
        raising exceptions, allowing the OAuth flow to fall back to other scope sources.
        """
        return None

    async def resolve_granted_scopes(
        self,
        *,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> str:
        """
        Determine which scopes should be persisted.

        Default behavior: prefer token response scope, fall back to requested scope.
        Providers can override to use token introspection for more accurate scope data.
        """
        return token.get("scope") or state_data.get("requested_scope") or ""

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return profile details for the connected account."""
        _ = client  # client/state_data reserved for provider-specific overrides
        _ = state_data
        return token.get("userinfo") or {}


class ProviderRegistry:
    """Registry to resolve integration providers by name or alias."""

    def __init__(self) -> None:
        self._providers: Dict[str, IntegrationProvider] = {}

    def register(self, provider: IntegrationProvider) -> None:
        keys = {provider.provider, *provider.aliases}
        for key in filter(None, keys):
            self._providers[key] = provider

    def get(self, provider_name: str) -> Optional[IntegrationProvider]:
        return self._providers.get(provider_name)

    def all_providers(self) -> List[IntegrationProvider]:
        seen = set()
        providers: List[IntegrationProvider] = []
        for provider in self._providers.values():
            if provider.provider in seen:
                continue
            seen.add(provider.provider)
            providers.append(provider)
        return providers
