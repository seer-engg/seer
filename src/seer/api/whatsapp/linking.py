"""WhatsApp phone number linking endpoints for user settings."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel

from seer.api.core.errors import AUTH_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.database import User
from seer.database.whatsapp_models import WhatsAppUserLink
from seer.logger import get_logger
from seer.services.whatsapp.client import send_text_message

router = APIRouter(prefix="/users/me/whatsapp", tags=["whatsapp-linking"])
logger = get_logger(__name__)


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise_problem(type_uri=AUTH_PROBLEM, title="Unauthorized", detail="Authentication required", status=401)
    return user


class LinkRequest(BaseModel):
    phone_number: str


class VerifyRequest(BaseModel):
    phone_number: str
    code: str


@router.post("/link")
async def link_phone(request: Request, body: LinkRequest):
    """Send a verification code to the given WhatsApp number."""
    user = _require_user(request)
    phone = body.phone_number.strip().lstrip("+")

    if not phone.isdigit() or len(phone) < 10:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Invalid phone", detail="Provide a valid phone number with country code", status=400)

    # Check if already linked to another user
    existing = await WhatsAppUserLink.filter(phone_number=phone, verified=True).first()
    if existing:
        await existing.fetch_related("user")
        if existing.user.id != user.id:
            raise_problem(type_uri=VALIDATION_PROBLEM, title="Phone already linked", detail="This number is linked to another account", status=409)

    code = secrets.token_hex(3).upper()  # 6-char hex code
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)

    link, _ = await WhatsAppUserLink.get_or_create(
        user=user,
        phone_number=phone,
        defaults={"verification_code": code, "verification_expires_at": expires},
    )
    if link.verified:
        return {"status": "already_verified", "phone_number": phone}

    link.verification_code = code
    link.verification_expires_at = expires
    await link.save(update_fields=["verification_code", "verification_expires_at"])

    try:
        await send_text_message(phone, f"Your Seer verification code: {code}")
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: Must inform user even if send fails
        logger.exception("Failed to send verification code via WhatsApp")
        raise_problem(
            type_uri=VALIDATION_PROBLEM, title="Send failed",
            detail="Could not send verification code. Check the phone number.", status=502,
        )

    return {"status": "code_sent", "phone_number": phone}


@router.post("/verify")
async def verify_phone(request: Request, body: VerifyRequest):
    """Confirm verification code to link the phone number."""
    user = _require_user(request)
    phone = body.phone_number.strip().lstrip("+")

    link = await WhatsAppUserLink.get_or_none(user=user, phone_number=phone)
    if not link:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Not found", detail="No pending link for this number", status=404)

    if link.verified:
        return {"status": "already_verified", "phone_number": phone}

    if link.verification_expires_at and link.verification_expires_at < datetime.now(timezone.utc):
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Code expired", detail="Verification code has expired. Request a new one.", status=400)

    if link.verification_code != body.code.strip().upper():
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Invalid code", detail="Verification code does not match", status=400)

    link.verified = True
    link.verification_code = None
    link.verification_expires_at = None
    await link.save(update_fields=["verified", "verification_code", "verification_expires_at"])

    return {"status": "verified", "phone_number": phone}


@router.delete("/link")
async def unlink_phone(request: Request):
    """Unlink WhatsApp phone number from user account."""
    user = _require_user(request)
    deleted = await WhatsAppUserLink.filter(user=user).delete()
    if not deleted:
        raise_problem(type_uri=VALIDATION_PROBLEM, title="Not found", detail="No WhatsApp link found", status=404)
    return {"status": "unlinked"}


@router.get("/link")
async def get_link_status(request: Request):
    """Get current WhatsApp link status."""
    user = _require_user(request)
    link = await WhatsAppUserLink.filter(user=user).first()
    if not link:
        return {"linked": False}
    return {"linked": link.verified, "phone_number": link.phone_number}
