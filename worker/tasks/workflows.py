from __future__ import annotations

from worker.broker import broker
from shared.logger import get_logger

logger = get_logger(__name__)


@broker.task
async def execute_saved_workflow(run_id: int, user_id: int) -> None:
    """Execute a persisted workflow run asynchronously."""
    logger.info("Executing saved workflow via Taskiq", extra={"run_id": run_id, "user_id": user_id})
    from api.workflows import services as workflow_services  # local import to avoid cycles

    await workflow_services.execute_saved_workflow_run(run_id=run_id, user_id=user_id)


__all__ = ["execute_saved_workflow"]


