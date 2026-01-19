from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.services.integrations.constants import (
    SUPABASE_OAUTH_PROVIDER,
    SUPABASE_RESOURCE_PROVIDER,
    SUPABASE_RESOURCE_TYPE_PROJECT,
)
from seer.services.integrations.providers.base import IntegrationProvider, ProviderContext
from seer.config import config
from seer.logger import get_logger
from seer.tools.oauth_manager import get_oauth_token

logger = get_logger(__name__)


class SupabaseProvider(IntegrationProvider):
    provider = SUPABASE_RESOURCE_PROVIDER
    resource_types = {SUPABASE_RESOURCE_TYPE_PROJECT}
    aliases = {SUPABASE_OAUTH_PROVIDER}

    async def bind_resource(
        self,
        context: ProviderContext,
        user,
        resource_type: str,
        *,
        project_ref: Optional[str] = None,
        connection_id: Optional[str] = None,
    ):
        if resource_type != SUPABASE_RESOURCE_TYPE_PROJECT:
            raise HTTPException(status_code=400, detail=f"Unsupported Supabase resource type '{resource_type}'")
        if not project_ref:
            raise HTTPException(status_code=400, detail="project_ref is required")

        connection, access_token = await get_oauth_token(
            user,
            connection_id=connection_id,
            provider=SUPABASE_OAUTH_PROVIDER,
        )

        project = await self._fetch_project(access_token, project_ref)
        project_id = str(project.get("id") or project.get("project_id") or project_ref)
        resource = await context.upsert_resource(
            user=user,
            oauth_connection=connection,
            provider=SUPABASE_RESOURCE_PROVIDER,
            resource_type=SUPABASE_RESOURCE_TYPE_PROJECT,
            resource_id=project_id,
            resource_key=project.get("ref") or project_ref,
            name=project.get("name"),
            metadata=project,
        )

        api_keys = await self._fetch_api_keys(access_token, project_ref)
        await self._sync_project_secrets(context, user, resource, api_keys)
        return resource

    async def list_remote_resources(
        self,
        access_token: str,
        resource_type: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if resource_type != SUPABASE_RESOURCE_TYPE_PROJECT:
            raise HTTPException(status_code=400, detail=f"Unsupported Supabase resource type '{resource_type}'")
        return await self._fetch_projects(access_token)

    async def _request(self, method: str, path: str, access_token: str) -> Any:
        url = f"{self._api_base()}{path}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(method, url, headers=headers)
                response.raise_for_status()
                if response.text:
                    return response.json()
                return None
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Supabase API error",
                extra={"url": url, "status_code": exc.response.status_code, "body": exc.response.text[:200]},
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Supabase API error: {exc.response.text[:200]}",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected Supabase API error", extra={"url": url})
            raise HTTPException(status_code=500, detail=f"Supabase API error: {str(exc)}") from exc

    def _api_base(self) -> str:
        base = config.supabase_management_api_base or "https://api.supabase.com"
        return base.rstrip("/")

    async def _fetch_projects(self, access_token: str) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/v1/projects", access_token)
        if isinstance(data, list):
            return data
        return []

    async def _fetch_project(self, access_token: str, project_ref: str) -> Dict[str, Any]:
        project = await self._request("GET", f"/v1/projects/{project_ref}", access_token)
        if not project:
            raise HTTPException(status_code=404, detail=f"Supabase project '{project_ref}' not found")
        return project

    async def _fetch_api_keys(self, access_token: str, project_ref: str) -> List[Dict[str, Any]]:
        data = await self._request("GET", f"/v1/projects/{project_ref}/api-keys", access_token)
        if isinstance(data, list):
            return data
        return []

    async def _sync_project_secrets(
        self,
        context: ProviderContext,
        user,
        resource,
        api_keys: List[Dict[str, Any]],
    ) -> None:
        for entry in api_keys:
            api_key = entry.get("api_key") or entry.get("key")
            key_name = entry.get("name") or entry.get("key_name")
            if not api_key or not key_name:
                continue
            await context.upsert_secret(
                user=user,
                provider=SUPABASE_RESOURCE_PROVIDER,
                name=self._format_secret_name(key_name),
                secret_type="api_key",
                value_enc=api_key,
                resource=resource,
                oauth_connection=None,
                metadata={
                    "project_ref": resource.resource_key,
                    "supabase_key_name": key_name,
                },
            )

    def _format_secret_name(self, raw_name: str) -> str:
        mapping = {
            "service_role": "supabase_service_role_key",
            "service-role": "supabase_service_role_key",
            "service": "supabase_service_role_key",
            "anon": "supabase_anon_key",
            "anon_key": "supabase_anon_key",
        }
        normalized = raw_name.lower()
        return mapping.get(normalized, f"supabase_{normalized}_key")

    async def fetch_user_profile(
        self,
        *,
        client: Any,
        token: Dict[str, Any],
        state_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not token.get("access_token"):
            logger.error("Supabase token missing access_token. keys={keys}", extra={"keys": list(token.keys())})
            raise HTTPException(
                status_code=500,
                detail="No access token in OAuth response. This may indicate an OAuth configuration issue.",
            )
        user_id = state_data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="Missing user_id in OAuth state for Supabase")
        return {
            "id": user_id,
            "integration_type": state_data.get("integration_type"),
        }
