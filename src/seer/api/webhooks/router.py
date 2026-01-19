from __future__ import annotations

from fastapi import APIRouter, Header, Request, status

from seer.api.webhooks import services as webhook_services

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])


@router.post("/generic/{subscription_id}", status_code=status.HTTP_202_ACCEPTED)
async def generic_webhook(
    subscription_id: int,
    request: Request,
    seer_secret: str | None = Header(default=None, alias="X-Seer-Webhook-Secret"),
    provider_event_id: str | None = Header(default=None, alias="X-Provider-Event-Id"),
):
    payload = await request.json()
    event = await webhook_services.handle_generic_webhook(
        subscription_id,
        payload=payload,
        headers=request.headers,
        secret=seer_secret,
        provider_event_id=provider_event_id,
    )
    return {"ok": True, "event_id": event.id}


__all__ = ["router"]
