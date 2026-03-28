"""BYOK API key management routes."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from seer.api.core.errors import AUTH_PROBLEM, raise_problem
from seer.database.models import User
from seer.database.byok_models import LLMApiKey
from seer.database.organization_models import Organization
from seer.logger import get_logger
from seer.services.byok.key_vault import get_key_vault

logger = get_logger("api.byok")
router = APIRouter(prefix="/api/byok", tags=["byok"])


# --- Request/Response models ---

class AddKeyRequest(BaseModel):
    label: str = Field(max_length=100)
    api_key: str = Field(min_length=8)
    base_url: str = Field(default="https://openrouter.ai/api/v1", max_length=500)


class KeyResponse(BaseModel):
    id: int
    label: str
    base_url: str
    key_masked: str
    is_active: bool
    status: str
    last_used_at: Optional[str] = None
    created_at: str


class TestConnectionRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class TestConnectionResponse(BaseModel):
    success: bool
    models: List[str] = []
    error: Optional[str] = None


# --- Helpers ---

def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise_problem(type_uri=AUTH_PROBLEM, title="Unauthorized", detail="Authentication required", status=401)
    return user


async def _get_org(user: User) -> Organization:
    org = await Organization.get_or_none(id=user.active_organization_id)
    if org is None:
        raise_problem(type_uri=AUTH_PROBLEM, title="No organization", detail="No active organization", status=400)
    return org


def _key_to_response(key: LLMApiKey) -> KeyResponse:
    vault = get_key_vault()
    decrypted = vault.decrypt(key.key_enc)
    masked = vault.mask(decrypted) if decrypted else f"...{key.key_suffix}"
    return KeyResponse(
        id=key.id,
        label=key.label,
        base_url=key.base_url,
        key_masked=masked,
        is_active=key.is_active,
        status=key.status,
        last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
        created_at=key.created_at.isoformat(),
    )


# --- Routes ---

@router.get("/keys", response_model=List[KeyResponse])
async def list_keys(request: Request):
    """List all LLM API keys for the user's active organization."""
    user = _require_user(request)
    org = await _get_org(user)
    keys = await LLMApiKey.filter(organization=org, status="active").order_by("-created_at")
    return [_key_to_response(k) for k in keys]


@router.post("/keys", response_model=KeyResponse, status_code=201)
async def add_key(request: Request, body: AddKeyRequest):
    """Add a new LLM API key. Deactivates any existing active key."""
    user = _require_user(request)
    org = await _get_org(user)
    vault = get_key_vault()

    # Deactivate existing active keys for this org
    await LLMApiKey.filter(organization=org, is_active=True).update(is_active=False)

    key = await LLMApiKey.create(
        organization=org,
        created_by=user,
        label=body.label,
        base_url=body.base_url,
        key_enc=vault.encrypt(body.api_key),
        key_suffix=vault.suffix(body.api_key),
        is_active=True,
    )
    logger.info("BYOK key added for org %s by user %s", org.id, user.user_id)
    return _key_to_response(key)


@router.delete("/keys/{key_id}", status_code=204)
async def delete_key(request: Request, key_id: int):
    """Revoke (soft-delete) a BYOK key."""
    user = _require_user(request)
    org = await _get_org(user)
    updated = await LLMApiKey.filter(id=key_id, organization=org).update(status="revoked", is_active=False)
    if not updated:
        raise_problem(type_uri="not_found", title="Key not found", detail="Key not found", status=404)
    logger.info("BYOK key %s revoked for org %s", key_id, org.id)


@router.post("/keys/test", response_model=TestConnectionResponse)
async def test_connection(request: Request, body: TestConnectionRequest):
    """Test an LLM API key by listing models from the provider."""
    import httpx  # pylint: disable=import-outside-toplevel  # Reason: Only needed for this endpoint

    user = _require_user(request)
    org = await _get_org(user)

    api_key = body.api_key
    base_url = body.base_url

    if not api_key:
        active_key = await LLMApiKey.get_or_none(organization=org, is_active=True, status="active")
        if not active_key:
            return TestConnectionResponse(success=False, error="No active API key found")
        vault = get_key_vault()
        api_key = vault.decrypt(active_key.key_enc)
        base_url = base_url or active_key.base_url
        if not api_key:
            return TestConnectionResponse(success=False, error="Failed to decrypt stored key")

    base_url = base_url or "https://openrouter.ai/api/v1"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            model_ids = [m.get("id", "") for m in data.get("data", [])][:50]
            return TestConnectionResponse(success=True, models=model_ids)
    except httpx.HTTPStatusError as e:
        return TestConnectionResponse(success=False, error=f"Provider returned {e.response.status_code}")
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Test endpoint should return errors, not crash
        return TestConnectionResponse(success=False, error=str(e)[:200])
