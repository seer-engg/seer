from __future__ import annotations

from worker.broker import broker
from shared.logger import get_logger

logger = get_logger(__name__)


@broker.task
async def execute_saved_workflow(run_id: int, user_id: int) -> None:
    """Execute a persisted workflow run asynchronously."""
    logger.info("Executing saved workflow via Taskiq", extra={"run_id": run_id, "user_id": user_id})
    from api.workflows import services as workflow_services  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from shared.analytics import analytics  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from shared.database import User  # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports

    try:
        await workflow_services.execute_saved_workflow_run(run_id=run_id, user_id=user_id)
    except Exception as e:
        logger.exception("Worker task failed for workflow execution", extra={"run_id": run_id})

        # Track worker error to PostHog
        user = await User.get_or_none(id=user_id)
        if user:
            analytics.capture_error(
                distinct_id=user.user_id,
                error=e,
                context={"run_id": run_id, "task": "execute_saved_workflow"},
                error_location="worker_task",
            )
        raise


__all__ = ["execute_saved_workflow"]
