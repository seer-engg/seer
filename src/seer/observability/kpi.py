"""
KPI query functions for Seer's top 4 strategic metrics.

1. Weekly Active Workflows (WAW) — north star usage metric
2. Time-to-first-workflow (TTFW) — DX/growth lever
3. Workflow step failure rate — retention driver
4. Dry-run adoption rate — differentiator metric (placeholder until dry-run ships)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from seer.database.models import User
from seer.database.organization_models import Organization, OrganizationType
from seer.database.workflow_models import Workflow, WorkflowRun, WorkflowRunStatus
from seer.logger import get_logger

logger = get_logger(__name__)


async def get_weekly_active_workflows(
    *,
    user: Optional[User] = None,
    organization: Optional[Organization] = None,
) -> int:
    """Count distinct workflows that had at least one run in the last 7 days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    qs = WorkflowRun.filter(
        created_at__gte=cutoff,
        workflow_id__not_isnull=True,
    )
    if organization and organization.type == OrganizationType.TEAM:
        qs = qs.filter(workflow__organization=organization)
    elif user:
        qs = qs.filter(user=user)

    result = await qs.distinct().values_list("workflow_id", flat=True)
    return len(set(result))


async def get_time_to_first_workflow(user: User) -> Optional[float]:
    """
    Seconds between user signup and their first workflow run.

    Returns None if user has no runs yet.
    """
    first_run = (
        await WorkflowRun.filter(user=user)
        .order_by("created_at")
        .first()
    )
    if not first_run:
        return None

    delta = first_run.created_at - user.created_at
    return delta.total_seconds()


async def get_workflow_failure_rate(
    *,
    user: Optional[User] = None,
    organization: Optional[Organization] = None,
    days: int = 7,
) -> dict:
    """
    Workflow run failure rate over a time window.

    Returns dict with total_runs, failed_runs, failure_rate (0.0-1.0).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    qs = WorkflowRun.filter(created_at__gte=cutoff)

    if organization and organization.type == OrganizationType.TEAM:
        qs = qs.filter(workflow__organization=organization)
    elif user:
        qs = qs.filter(user=user)

    # Only count terminal states
    terminal = qs.filter(
        status__in=[
            WorkflowRunStatus.SUCCEEDED,
            WorkflowRunStatus.FAILED,
        ]
    )
    total = await terminal.count()
    failed = await terminal.filter(status=WorkflowRunStatus.FAILED).count()

    return {
        "total_runs": total,
        "failed_runs": failed,
        "failure_rate": round(failed / total, 4) if total > 0 else 0.0,
    }



async def get_dry_run_adoption(
    *,
    user: Optional[User] = None,
    organization: Optional[Organization] = None,
    days: int = 30,  # pylint: disable=unused-argument  # Reason: reserved for future dry-run time-window queries
) -> dict:
    """
    Placeholder: dry-run adoption rate.

    Currently tracks workflows that used graph_preview (compile validation)
    as a proxy until the full dry-run feature ships.

    Returns dict with total_workflows, previewed_workflows, adoption_rate.
    """
    # Count active workflows
    wf_qs = Workflow.filter()
    if organization and organization.type == OrganizationType.TEAM:
        wf_qs = wf_qs.filter(organization=organization)
    elif user:
        wf_qs = wf_qs.filter(user=user)

    total_workflows = await wf_qs.count()

    # Dry-run not yet implemented — return zero adoption with explanation
    return {
        "total_workflows": total_workflows,
        "previewed_workflows": 0,
        "adoption_rate": 0.0,
        "status": "pending_implementation",
    }
