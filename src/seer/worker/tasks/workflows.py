from __future__ import annotations

from typing import Any, Dict, Optional

from seer.worker.broker_instance import broker
from seer.logger import get_logger
from seer.services.workflows.execution import execute_saved_workflow_run
from seer.database import User, WorkflowRun
from seer.observability.sentry_client import set_user_context, set_tag, set_context

logger = get_logger(__name__)


async def _set_sentry_context_for_workflow(run_id: int, user_id: int) -> None:
    """
    Set Sentry context for workflow execution error tracking.

    Sets user context (id, email, username) and workflow run context.
    All operations are non-blocking and fail silently.
    """
    set_tag("task_type", "workflow_execution")
    set_tag("run_id", str(run_id))

    try:
        user = await User.get(id=user_id)
        set_user_context(
            user_id=user.user_id,
            email=getattr(user, "email", None),
            username=f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip() or None,
        )
        # Set user tags for indexed searching in Sentry
        set_tag("user_id", user.user_id)
        if getattr(user, "email", None):
            set_tag("user_email", user.email)
        run = await WorkflowRun.get_or_none(id=run_id)
        if run:
            await run.fetch_related("workflow")
            set_context("workflow_run", {
                "run_id": run.run_id,
                "workflow_id": getattr(run.workflow, "workflow_id", None),
                "source": run.source.value if run.source else None,
            })
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: Sentry context setup must never block task execution
        logger.debug("Failed to set Sentry context for workflow", exc_info=True)


@broker.task
async def workflow_execution_task(
    run_id: int,
    user_id: int,
    trigger_envelope: Optional[Dict[str, Any]] = None
) -> None:
    """Execute a persisted workflow run asynchronously."""
    logger.info(
        "Executing saved workflow via Taskiq",
        extra={
            "run_id": run_id,
            "user_id": user_id,
            "has_trigger": bool(trigger_envelope)
        }
    )

    await _set_sentry_context_for_workflow(run_id, user_id)

    try:
        await execute_saved_workflow_run(
            run_id=run_id,
            user_id=user_id,
            trigger_envelope=trigger_envelope
        )
    except Exception:
        logger.exception("Worker task failed for workflow execution", extra={"run_id": run_id})
        raise


__all__ = ["workflow_execution_task"]
