from __future__ import annotations

from typing import Dict, Optional

from .base import ResourceProvider, ResourceProviderRegistry
from .github import GitHubResourceProvider
from .google import GoogleResourceProvider
from .supabase import SupabaseResourceProvider

_registry = ResourceProviderRegistry()
_registry.register(GoogleResourceProvider())
_registry.register(GitHubResourceProvider())
_registry.register(SupabaseResourceProvider())


def register_provider(provider: ResourceProvider) -> None:
    _registry.register(provider)


def get_resource_provider(provider_name: str) -> Optional[ResourceProvider]:
    return _registry.get(provider_name)


def get_resource_provider_for_type(resource_type: str) -> Optional[ResourceProvider]:
    return _registry.find_by_resource_type(resource_type)


def get_all_resource_configs() -> Dict[str, Dict[str, object]]:
    return _registry.resource_configs()
