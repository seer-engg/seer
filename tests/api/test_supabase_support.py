from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from api.integrations.services import (
    _build_manual_supabase_metadata,
    _format_supabase_secret_name,
    bind_supabase_project_manual,
)
from shared.tools.supabase import _resolve_rest_url


def test_format_supabase_secret_name_aliases() -> None:
    assert _format_supabase_secret_name("service_role") == "supabase_service_role_key"
    assert _format_supabase_secret_name("anon") == "supabase_anon_key"
    assert _format_supabase_secret_name("custom") == "supabase_custom_key"


def test_resolve_rest_url_falls_back_to_project_ref() -> None:
    resource = SimpleNamespace(resource_metadata={}, resource_key="demo-ref")
    assert _resolve_rest_url(resource) == "https://demo-ref.supabase.co/rest/v1"

    resource_with_metadata = SimpleNamespace(
        resource_metadata={"rest_url": "https://example.supabase.co/rest/v1"},
        resource_key="ignored",
    )
    assert _resolve_rest_url(resource_with_metadata) == "https://example.supabase.co/rest/v1"


def test_build_manual_supabase_metadata_defaults() -> None:
    metadata = _build_manual_supabase_metadata(
        project_ref="demo-ref",
        project_name="Demo",
    )
    assert metadata["binding_mode"] == "manual"
    assert metadata["rest_url"] == "https://demo-ref.supabase.co/rest/v1"
    assert metadata["project_ref"] == "demo-ref"
    assert metadata["name"] == "Demo"


@pytest.mark.asyncio
async def test_bind_supabase_project_manual_persists_keys(monkeypatch) -> None:
    user = SimpleNamespace(user_id="user-123")
    resource = SimpleNamespace(id=42)

    upsert_resource = AsyncMock(return_value=resource)
    upsert_secret = AsyncMock()

    monkeypatch.setattr("api.integrations.services._upsert_integration_resource", upsert_resource)
    monkeypatch.setattr("api.integrations.services._upsert_integration_secret", upsert_secret)

    result = await bind_supabase_project_manual(
        user,
        project_ref="demo-ref",
        service_role_key="srv-key",
        anon_key="anon-key",
    )

    assert result is resource
    upsert_resource.assert_awaited_once()
    assert upsert_secret.await_count == 2


@pytest.mark.asyncio
async def test_bind_supabase_project_manual_requires_service_key(monkeypatch) -> None:
    user = SimpleNamespace(user_id="user-123")

    with pytest.raises(HTTPException):
        await bind_supabase_project_manual(
            user,
            project_ref="demo-ref",
            service_role_key="",
        )

