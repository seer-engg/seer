from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException


@dataclass(frozen=True)
class ResourceContext:
    """
    Authentication and user context for resource browsing.

    Supports multiple authentication types:
    - OAuth tokens (access_token)
    - Bot tokens (bot_token) - e.g., Discord
    - API keys (api_key)
    - Service role keys (service_key) - e.g., Supabase
    - Custom provider-specific auth (custom_auth)

    Also provides user context for database-backed resources.
    """
    user: Any  # User object (avoiding circular import)
    access_token: Optional[str] = None  # OAuth access token
    bot_token: Optional[str] = None  # Bot token (Discord)
    api_key: Optional[str] = None  # API key
    service_key: Optional[str] = None  # Service role key (Supabase)
    custom_auth: Optional[Dict[str, Any]] = None  # Provider-specific auth


class ResourceProvider:
    """Base class for provider-specific resource browsers."""

    provider: str
    aliases: Set[str] = set()
    resource_configs: Dict[str, Dict[str, Any]] = {}

    def matches_provider(self, provider_name: str) -> bool:
        return provider_name == self.provider or provider_name in self.aliases

    def supports_resource_type(self, resource_type: str) -> bool:
        return resource_type in self.resource_configs

    def get_supported_resource_types(self) -> List[str]:
        return list(self.resource_configs.keys())

    def get_resource_config(self, resource_type: str) -> Optional[Dict[str, Any]]:
        return self.resource_configs.get(resource_type)

    async def list_resources(  # pylint: disable=too-many-arguments  # Provider interface requires full query context
        self,
        *,
        access_token: Optional[str] = None,  # Deprecated: use context instead
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
        context: Optional[ResourceContext] = None,  # Preferred: use this for all auth types
    ) -> Dict[str, Any]:
        """
        List resources for this provider.

        Args:
            access_token: (Deprecated) OAuth access token - use context instead
            resource_type: Type of resource to list
            query: Optional search query
            parent_id: Optional parent ID for hierarchical resources
            page_token: Pagination token
            page_size: Number of results per page
            filter_params: Optional filter parameters
            depends_on_values: Optional dependent parameter values
            context: (Preferred) ResourceContext with user and auth information

        Returns:
            Dictionary with items, next_page_token, and metadata

        Raises:
            HTTPException: If resource type is not supported
        """
        raise HTTPException(status_code=400, detail=f"{self.provider} does not support {resource_type}")


class ResourceProviderRegistry:
    """Registry for resolving resource providers."""

    def __init__(self) -> None:
        self._providers: Dict[str, ResourceProvider] = {}

    def register(self, provider: ResourceProvider) -> None:
        keys = {provider.provider, *provider.aliases}
        for key in filter(None, keys):
            self._providers[key] = provider

    def get(self, provider_name: str) -> Optional[ResourceProvider]:
        return self._providers.get(provider_name)

    def find_by_resource_type(self, resource_type: str) -> Optional[ResourceProvider]:
        for provider in self._providers.values():
            if provider.supports_resource_type(resource_type):
                return provider
        return None

    def resource_configs(self) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        seen_providers: Set[str] = set()
        for provider in self._providers.values():
            if provider.provider in seen_providers:
                continue
            seen_providers.add(provider.provider)
            merged.update(provider.resource_configs)
        return merged

    def supports(self, provider: str, resource_type: str) -> bool:
        """
        Check if a provider supports a specific resource type.

        Args:
            provider: Provider name
            resource_type: Resource type to check

        Returns:
            True if provider supports the resource type, False otherwise
        """
        provider_impl = self.get(provider)
        return provider_impl.supports_resource_type(resource_type) if provider_impl else False

    def get_config(self, provider: str, resource_type: str) -> Optional[Dict[str, Any]]:
        """
        Get resource configuration for a provider + resource type.

        Args:
            provider: Provider name
            resource_type: Resource type

        Returns:
            Resource config dict, or None if not found
        """
        provider_impl = self.get(provider)
        return provider_impl.get_resource_config(resource_type) if provider_impl else None
