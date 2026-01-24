from __future__ import annotations

from typing import Any, Dict, Optional

from seer.worker.broker_instance import broker
from seer.logger import get_logger
from seer.services.workflows.execution import execute_saved_workflow_run

logger = get_logger(__name__)


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
