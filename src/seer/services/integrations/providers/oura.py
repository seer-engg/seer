from __future__ import annotations

from seer.services.integrations.providers.base import IntegrationProvider


class OuraProvider(IntegrationProvider):
    """Oura Ring OAuth integration provider."""

    provider = "oura"
    resource_types: set[str] = set()
