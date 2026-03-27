from __future__ import annotations

from fastapi import APIRouter, Header, Request, status

from seer.api.webhooks import services as webhook_services
from seer.api.webhooks.twilio_whatsapp import router as twilio_whatsapp_router

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])


@router.post("/generic/{webhook_slug}", status_code=status.HTTP_202_ACCEPTED)
async def generic_webhook(
    webhook_slug: str,
    request: Request,
    provider_event_id: str | None = Header(default=None, alias="X-Provider-Event-Id"),
):
    """Handle incoming webhook events using slug-based URL security."""
    payload = await request.json()
    event = await webhook_services.handle_generic_webhook_by_slug(
        webhook_slug,
        payload=payload,
        headers=request.headers,
        provider_event_id=provider_event_id,
    )
    return {"ok": True, "event_id": event.id}


router.include_router(twilio_whatsapp_router)

__all__ = ["router"]
