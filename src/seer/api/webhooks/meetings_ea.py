"""
meetingsEA webhook — receives bot state change and transcript events from the Attendee service.

Two responsibilities:
1. Routes events to active trigger subscriptions (for standalone trigger workflows)
2. Auto-resumes interrupted EA Bot nodes when a meeting ends (for ea_bot node workflows)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Request, status

from seer.api.webhooks import services as webhook_services
from seer.config import config
from seer.database import TriggerSubscription
from seer.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["webhooks"])

_REQUEST_TIMEOUT_SECONDS = 30


async def _get_meetings_ea_subscription(org_id: Optional[str] = None) -> TriggerSubscription:
    """Find an active subscription for webhook.meetings_ea.bot_event."""
    query = TriggerSubscription.filter(
        trigger_key="webhook.meetings_ea.bot_event",
        enabled=True,
    )

    subscriptions = await query.all()

    if not subscriptions:
        raise LookupError("No active webhook.meetings_ea.bot_event subscription")

    if len(subscriptions) == 1:
        return subscriptions[0]

    # Multiple subscriptions: match by org_id in provider_config if available
    if org_id:
        for sub in subscriptions:
            cfg_org_id = (sub.provider_config or {}).get("seer_org_id", "")
            if str(cfg_org_id) == str(org_id):
                return sub

    # Fallback to first
    return subscriptions[0]


def _extract_event_id(payload: dict) -> Optional[str]:
    """Extract a unique event ID from the Attendee webhook payload for deduplication."""
    # Attendee uses idempotency_key as the unique event identifier
    idempotency_key = payload.get("idempotency_key")
    if idempotency_key:
        return str(idempotency_key)
    # Fallback: construct from trigger + bot_id + state
    trigger = payload.get("trigger", "")
    bot_id = payload.get("bot_id", "")
    data = payload.get("data") or {}
    new_state = data.get("new_state", "") if isinstance(data, dict) else ""
    if trigger and bot_id:
        return f"{trigger}:{bot_id}:{new_state}"
    return None


async def _fetch_attendee_data(bot_id: str) -> Dict[str, Any]:
    """Fetch transcript and recording from Attendee for a completed bot."""
    api_key = config.meetings_ea_api_key
    base_url = config.meetings_ea_base_url
    if not api_key or not base_url:
        logger.warning("Cannot fetch attendee data: meetingsEA not configured")
        return {"transcript": [], "recording_url": None}

    base = base_url.rstrip("/")  # pylint: disable=no-member  # Reason: Pydantic Field resolves to str at runtime
    headers = {"Authorization": f"Token {api_key}"}

    transcript = []
    recording_url = None

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        # Fetch transcript
        try:
            resp = await client.get(f"{base}/api/v1/bots/{bot_id}/transcript", headers=headers)
            if resp.status_code == 200:
                raw = resp.json()
                raw_utterances = raw if isinstance(raw, list) else raw.get("utterances", [])
                for u in raw_utterances:
                    transcription = u.get("transcription") or {}
                    text = transcription.get("transcript", "") if isinstance(transcription, dict) else str(transcription)
                    transcript.append({
                        "speaker": u.get("speaker_name", "Unknown"),
                        "text": text,
                        "start_ms": u.get("timestamp_ms", 0),
                        "duration_ms": u.get("duration_ms", 0),
                    })
        except (httpx.HTTPError, ValueError, KeyError):
            logger.exception("Failed to fetch transcript for bot %s", bot_id)

        # Fetch recording
        try:
            resp = await client.get(f"{base}/api/v1/bots/{bot_id}/recording", headers=headers)
            if resp.status_code == 200:
                recording_url = resp.json().get("url")
        except (httpx.HTTPError, ValueError, KeyError):
            logger.exception("Failed to fetch recording for bot %s", bot_id)

    return {"transcript": transcript, "recording_url": recording_url}


async def _try_auto_resume(bot_id: str, bot_state: str, metadata: Dict[str, Any]) -> bool:
    """
    If an EA Bot node is waiting for this bot, auto-resume the workflow.

    Returns True if a workflow was resumed, False otherwise.
    """
    run_id = metadata.get("seer_run_id")
    if not run_id:
        return False

    if bot_state != "ended":
        return False

    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.database.workflow_models import WorkflowRun, WorkflowRunStatus, parse_run_public_id
    from seer.services.workflows.execution import resume_workflow_run

    try:
        run_pk = parse_run_public_id(run_id)
        run = await WorkflowRun.get_or_none(id=run_pk).select_related("user")
    except (ValueError, LookupError):
        logger.debug("Could not look up run %s for auto-resume", run_id)
        return False

    if not run or run.status != WorkflowRunStatus.INTERRUPTED:
        logger.debug("Run %s not in INTERRUPTED state, skipping auto-resume", run_id)
        return False

    # Fetch transcript + recording from Attendee
    attendee_data = await _fetch_attendee_data(bot_id)

    resume_payload = {
        "bot_id": bot_id,
        "state": bot_state,
        "transcript": attendee_data["transcript"],
        "recording_url": attendee_data["recording_url"],
    }

    logger.info(
        "Auto-resuming workflow run from meetingsEA webhook",
        extra={"run_id": run_id, "bot_id": bot_id, "transcript_count": len(attendee_data["transcript"])},
    )

    try:
        await resume_workflow_run(run.user, run_id, resume_payload)
        return True
    except (RuntimeError, LookupError, ValueError):
        logger.exception("Failed to auto-resume run %s", run_id)
        return False


@router.post("/meetings_ea", status_code=status.HTTP_200_OK)
async def meetings_ea_webhook(request: Request) -> dict:
    """Receive bot state change and transcript events from the meetingsEA service."""
    payload = await request.json()

    # Attendee webhook payload format:
    # {idempotency_key, bot_id, bot_metadata, trigger, data: {event_type, new_state, old_state, ...}}
    event_type = payload.get("trigger", "unknown")
    bot_id = payload.get("bot_id", "")
    bot_metadata: Dict[str, Any] = payload.get("bot_metadata") or {}
    org_id = bot_metadata.get("seer_org_id")
    data = payload.get("data") or {}
    bot_state = data.get("new_state", "") if isinstance(data, dict) else ""

    logger.info(
        "meetingsEA webhook received",
        extra={"event_type": event_type, "bot_id": bot_id, "bot_state": bot_state, "org_id": org_id},
    )

    # Try auto-resume for EA Bot nodes (bot ended → resume interrupted workflow)
    if event_type == "bot.state_change" and bot_state == "ended":
        resumed = await _try_auto_resume(bot_id, bot_state, bot_metadata)
        if resumed:
            return {"ok": True, "resumed": True}

    # Also dispatch to trigger subscriptions (for standalone trigger workflows)
    try:
        subscription = await _get_meetings_ea_subscription(org_id=org_id)
        provider_event_id = _extract_event_id(payload)
        await webhook_services.handle_webhook_for_subscription(
            subscription,
            payload=payload,
            headers=request.headers,
            provider_event_id=provider_event_id,
        )
    except LookupError:
        logger.debug("No active meetingsEA subscription found, ignoring event %s", event_type)

    return {"ok": True}
