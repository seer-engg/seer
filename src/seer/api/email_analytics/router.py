"""
Email analytics API — paginated event list and summary stats.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from tortoise.expressions import Q

from seer.database.email_models import EmailEvent

router = APIRouter(prefix="/email-analytics", tags=["email-analytics"])


# =============================================================================
# Response models
# =============================================================================


class EmailEventItem(BaseModel):
    id: str
    tracking_id: str
    provider: str
    email_type: str
    event_type: str
    recipient: str
    subject: Optional[str]
    link_url: Optional[str]
    workflow_run_id: Optional[str]
    created_at: datetime


class EmailEventsResponse(BaseModel):
    events: list[EmailEventItem]
    total: int


class EmailSummaryResponse(BaseModel):
    total_sent: int
    total_opened: int
    total_clicked: int
    open_rate: float
    click_rate: float


# =============================================================================
# Helpers
# =============================================================================


def _require_user(request: Request):
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _get_org_id(request: Request) -> Optional[int]:
    org = getattr(request.state, "organization", None)
    return org.id if org else None


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/events", response_model=EmailEventsResponse)
async def list_email_events(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # Reason: query parameters for filtering
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    email_type: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    start: Optional[str] = Query(default=None, description="ISO date string"),
    end: Optional[str] = Query(default=None, description="ISO date string"),
) -> EmailEventsResponse:
    """List email tracking events for the current organization."""
    user = _require_user(request)
    org_id = _get_org_id(request)

    filters = Q()
    if org_id:
        filters &= Q(organization_id=org_id)
    else:
        filters &= Q(user_id=user.user_id)

    # Exclude pending (not yet sent)
    filters &= ~Q(event_type="pending")

    if email_type:
        filters &= Q(email_type=email_type)
    if event_type:
        filters &= Q(event_type=event_type)
    if start:
        filters &= Q(created_at__gte=datetime.fromisoformat(start))
    if end:
        filters &= Q(created_at__lte=datetime.fromisoformat(end))

    total = await EmailEvent.filter(filters).count()
    rows = await EmailEvent.filter(filters).order_by("-created_at").offset(offset).limit(limit)

    events = [
        EmailEventItem(
            id=str(row.id),
            tracking_id=str(row.tracking_id),
            provider=row.provider,
            email_type=row.email_type,
            event_type=row.event_type,
            recipient=row.recipient,
            subject=row.subject,
            link_url=row.link_url,
            workflow_run_id=row.workflow_run_id,
            created_at=row.created_at,
        )
        for row in rows
    ]

    return EmailEventsResponse(events=events, total=total)


@router.get("/summary", response_model=EmailSummaryResponse)
async def email_summary(
    request: Request,
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
) -> EmailSummaryResponse:
    """Summary stats: sent/opened/clicked counts and rates."""
    user = _require_user(request)
    org_id = _get_org_id(request)

    filters = Q()
    if org_id:
        filters &= Q(organization_id=org_id)
    else:
        filters &= Q(user_id=user.user_id)

    if start:
        filters &= Q(created_at__gte=datetime.fromisoformat(start))
    if end:
        filters &= Q(created_at__lte=datetime.fromisoformat(end))

    total_sent = await EmailEvent.filter(filters & Q(event_type="sent")).count()
    total_opened = await EmailEvent.filter(filters & Q(event_type="opened")).count()
    total_clicked = await EmailEvent.filter(filters & Q(event_type="clicked")).count()

    open_rate = (total_opened / total_sent * 100) if total_sent > 0 else 0.0
    click_rate = (total_clicked / total_sent * 100) if total_sent > 0 else 0.0

    return EmailSummaryResponse(
        total_sent=total_sent,
        total_opened=total_opened,
        total_clicked=total_clicked,
        open_rate=round(open_rate, 1),
        click_rate=round(click_rate, 1),
    )


__all__ = ["router"]
