from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException


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

    async def list_resources(
        self,
        *,
        access_token: str,
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
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
