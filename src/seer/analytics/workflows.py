
from dataclasses import dataclass
from typing import Optional, Dict, Any
from seer.analytics import analytics
from seer.config import config as shared_config
from seer.database import WorkflowRun, User
import time
from seer.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ExecutionMetrics:
    """Tracks execution metadata without polluting ORM model."""

    start_time: float
    execution_mode: str
    duration_ms: Optional[float] = None


class WorkflowAnalytics:
    """Tracks workflow execution metadata."""

    @staticmethod
    def _capture_workflow_start(
            run: WorkflowRun, user: User, execution_mode: str, inputs: Dict[str, Any]
        ) -> None:
            analytics.capture(
                distinct_id=user.user_id,
                event="workflow_run_started",
                properties={
                    "run_id": run.run_id,
                    "workflow_id": run.workflow.workflow_id if run.workflow else None,
                    "workflow_name": run.workflow.name if run.workflow else "draft",
                    "execution_mode": execution_mode,
                    "has_inputs": bool(inputs),
                    "input_keys": list(inputs.keys()) if inputs else [],
                    "deployment_mode": shared_config.seer_mode,
                },
            )

    @staticmethod
    async def _complete_run(
            run: WorkflowRun, output: Dict[str, Any], metrics: ExecutionMetrics
        ) -> WorkflowRun:


            # Capture workflow completion event
            duration_ms = (time.time() - metrics.start_time) * 1000

            analytics.capture(
                distinct_id=run.user.user_id,
                event="workflow_run_completed",
                properties={
                    "run_id": run.run_id,
                    "workflow_id": run.workflow.workflow_id if run.workflow else None,
                    "workflow_name": run.workflow.name if run.workflow else "draft",
                    "execution_mode": metrics.execution_mode,
                    "duration_ms": round(duration_ms, 2),
                    "output_keys": list(output.keys()) if output else [],
                    "deployment_mode": shared_config.seer_mode,
                },
            )

            return run

    @staticmethod
    async def _handle_run_failure(
            run: WorkflowRun,
            user: User,
            error: Exception,
            metrics: ExecutionMetrics,
            error_type: str,
        ) -> None:

            duration_ms = (time.time() - metrics.start_time) * 1000

            analytics.capture(
                distinct_id=user.user_id,
                event="workflow_run_failed",
                properties={
                    "run_id": run.run_id,
                    "workflow_id": (
                        run.workflow.workflow_id if run.workflow else None
                    ),
                    "workflow_name": run.workflow.name if run.workflow else "draft",
                    "execution_mode": metrics.execution_mode,
                    "duration_ms": round(duration_ms, 2),
                    "error_type": error_type,
                    "error_message": str(error)[:500],
                    "deployment_mode": shared_config.seer_mode,
                },
            )
