"""
PostHog analytics helpers for workflow node execution and Nexus agent tools.

Provides capture_workflow_event() for non-HTTP contexts (LangGraph nodes,
background tasks) that need to emit PostHog events.

Usage:
    from seer.analytics.workflow_tracking import capture_workflow_event

    await capture_workflow_event(
        event="workflow_node_executed",
        user_email="user@example.com",
        properties={"node_type": "tool", "tool_name": "gmail_send_email"},
    )
"""
from datetime import datetime, timezone
from typing import Any, Dict

from seer.config import config
from seer.logger import get_logger
from seer.observability.posthog_client import capture_event

logger = get_logger(__name__)


async def capture_workflow_event(
    event: str,
    user_email: str,
    properties: Dict[str, Any],
) -> None:
    """
    Capture a PostHog event from non-HTTP contexts (workflow nodes, Nexus tools).

    Adds timestamp to every event automatically.
    No-ops if PostHog is not configured or user_email is empty.

    Args:
        event: PostHog event name (e.g., "workflow_node_executed")
        user_email: User's email address used as distinct_id
        properties: Event-specific properties dict (will be mutated with common props)
    """
    if not config.is_posthog_configured:
        return

    if not user_email:
        logger.debug("capture_workflow_event: skipping %s — no user email", event)
        return

    enriched = {
        **properties,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    capture_event(
        distinct_id=user_email,
        event=event,
        properties=enriched,
    )


async def capture_kpi_event(
    user_email: str,
    kpi_name: str,
    value: float,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Emit a PostHog event for KPI tracking.

    Used to push computed KPIs into PostHog for dashboard visualization.
    Called after workflow execution completes.
    """
    properties: Dict[str, Any] = {
        "kpi_name": kpi_name,
        "value": value,
    }
    if metadata:
        properties.update(metadata)

    await capture_workflow_event(
        event="seer_kpi_recorded",
        user_email=user_email,
        properties=properties,
    )
