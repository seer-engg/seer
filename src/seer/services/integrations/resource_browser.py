"""
Resource Browser facade that delegates to provider-specific implementations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from seer.services.integrations.resource_providers import (
    get_all_resource_configs,
    get_resource_provider,
)
from seer.logger import get_logger

logger = get_logger(__name__)


class ResourceBrowser:
    """
    Unified resource browser facade.

    Responsibilities:
    - Validate resource types and provider support.
    - Pass call context (paging/search params) to provider implementations.
    """

    def __init__(self, access_token: str, provider: str):
        self.access_token = access_token
        self.provider = provider

    def _get_provider(self):
        provider_impl = get_resource_provider(self.provider)
        if not provider_impl:
            raise HTTPException(status_code=400, detail=f"Provider '{self.provider}' is not configured")
        return provider_impl

    async def list_resources(
        self,
        resource_type: str,
        query: Optional[str] = None,
        parent_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 50,
        filter_params: Optional[Dict[str, Any]] = None,
        depends_on_values: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        configs = get_all_resource_configs()
        if resource_type not in configs:
            raise ValueError(f"Unknown resource type: {resource_type}")

        provider_impl = self._get_provider()
        logger.info(f"Listing resources for provider: {provider_impl.provider}")
        if not provider_impl.supports_resource_type(resource_type):
            raise HTTPException(
                status_code=400,
                detail=f"Resource type '{resource_type}' not supported by provider '{self.provider}'",
            )

        return await provider_impl.list_resources(
            access_token=self.access_token,
            resource_type=resource_type,
            query=query,
            parent_id=parent_id,
            page_token=page_token,
            page_size=page_size,
            filter_params=filter_params,
            depends_on_values=depends_on_values,
        )

    @classmethod
    def get_supported_resource_types(cls, provider: str) -> List[str]:
        provider_impl = get_resource_provider(provider)
        if not provider_impl:
            return []
        return provider_impl.get_supported_resource_types()

    @classmethod
    def get_resource_type_info(cls, resource_type: str) -> Optional[Dict[str, Any]]:
        configs = get_all_resource_configs()
        if resource_type not in configs:
            return None

        config = configs[resource_type]
        return {
            "resource_type": resource_type,
            "display_field": config.get("display_field", "name"),
            "value_field": config.get("value_field", "id"),
            "supports_hierarchy": config.get("supports_hierarchy", False),
            "supports_search": config.get("supports_search", False),
            "depends_on": config.get("depends_on"),
        }
