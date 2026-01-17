from __future__ import annotations

from typing import Optional

from .base import IntegrationProvider, ProviderContext, ProviderRegistry
from .github import GitHubProvider
from .google import GoogleProvider
from .supabase import SupabaseProvider

_registry = ProviderRegistry()
_registry.register(GoogleProvider())
_registry.register(GitHubProvider())
_registry.register(SupabaseProvider())


def get_integration_provider(provider_name: str) -> Optional[IntegrationProvider]:
    return _registry.get(provider_name)


__all__ = [
    "IntegrationProvider",
    "ProviderContext",
    "get_integration_provider",
]
