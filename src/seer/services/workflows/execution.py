from typing import Dict, Any, Optional
from datetime import datetime, timezone
import time
import traceback
from fastapi import HTTPException


from seer.api.agents.checkpointer import get_checkpointer
from seer.core.errors import WorkflowCompilerError
from seer.database import WorkflowRun, User, WorkflowRunStatus
from seer.core.runtime.context import WorkflowRuntimeContext

from seer.analytics.workflows import ExecutionMetrics, WorkflowAnalytics
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton



from seer.logger import get_logger
logger = get_logger(__name__)

def _now() -> datetime:
    return datetime.now(timezone.utc)

async def _compile_workflow(
    user: User,
    spec: Dict[str, Any],
    checkpointer: Optional[Any] = None,
) -> Any:
    """
    Compile a workflow spec using the global compiler instance.

    This is a shared helper to avoid duplicating the compile pattern across
    history.py and execution.py.
    """
    compiler = WorkflowCompilerSingleton.instance()
    return await compiler.compile(
        user,
        spec,
        checkpointer=checkpointer,
    )


def _build_run_config(run: WorkflowRun, config_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensure LangGraph defaults (thread_id) are present so checkpoints can be recovered.

    IMPORTANT: Always uses run.run_id as thread_id to ensure checkpoint retrieval works.
    If config_payload contains a different thread_id, it will be overridden.
    """
    base_config = dict((config_payload or {}) or {})
    configurable = dict((base_config.get("configurable") or {}) or {})
    # Always use run.run_id as thread_id for checkpoint retrieval consistency
    # Don't use setdefault - explicitly set to ensure it matches execution config
    configurable["thread_id"] = run.thread_id or run.run_id
    base_config["configurable"] = configurable
    return base_config


async def _execute_run(
    run: WorkflowRun,
    user: User,
    *,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
    execution_mode: str,
    trigger_envelope: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], ExecutionMetrics]:
    """
    Fetches the workflow run object , compiles it using the global compiler instance and executes it.
    """
    logger.debug(
        "Preparing workflow run '%s' (workflow_id=%s) inputs_keys=%s "
        "config_payload_keys=%s user_id=%s",
        run.run_id,
        getattr(run.workflow, "workflow_id", None),
        sorted((inputs or {}).keys()),
        sorted((config_payload or {}).keys()),
        getattr(user, "id", None),
    )
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.RUNNING,
        started_at=_now(),
    )
    metrics = ExecutionMetrics(
        start_time=time.time(),
        execution_mode=execution_mode,
    )
    WorkflowAnalytics._capture_workflow_start(run, user, execution_mode, inputs)

    checkpointer = await get_checkpointer()
    try:
        compiled = await _compile_workflow(user, run.spec, checkpointer=checkpointer)
    except WorkflowCompilerError as exc:
        logger.error("Workflow compilation failed", exc_info=True)
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
        )
        await WorkflowAnalytics._handle_run_failure(run, user, exc, metrics, "CompilationError")
        raise
    try:
        run_config = dict(config_payload or {})
        logger.debug(
            "Invoking compiled workflow for run '%s' with config_keys=%s "
            "user_context_id=%s",
            run.run_id,
            sorted(run_config.keys()),
            getattr(user, "id", None),
        )
        effective_config = _build_run_config(run, run_config)
        logger.info(
            "Executing workflow run '%s' with config: %s",
            run.run_id,
            effective_config,
            extra={"run_id": run.run_id, "config": effective_config},
        )
        # Create runtime context with workflow_run_id for usage tracking

        runtime_context = WorkflowRuntimeContext(
            user=user,
            workflow_run_id=run.run_id,
        )
        result = await compiled.ainvoke(
            config=effective_config, context=runtime_context, trigger=trigger_envelope
        )
    except Exception as exc:
        print(f"{traceback.format_exc()}")
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
        )
        await WorkflowAnalytics._handle_run_failure(run, user, exc, metrics, "RuntimeError")
        raise

    return result, metrics





async def execute_saved_workflow_run(
    *,
    run_id: int,
    user_id: int,
    trigger_envelope: Optional[Dict[str, Any]] = None
) -> None:
    """
    Execute a saved workflow run asynchronously (invoked by Taskiq worker).
    """
    run = await WorkflowRun.get(id=run_id)
    await run.fetch_related("workflow", "user")

    user = run.user
    if user is None or getattr(user, "id", None) != user_id:
        user = await User.get(id=user_id)

    inputs = dict(run.inputs or {})
    config_payload = dict(run.config or {})

    try:
        output, metrics = await _execute_run(
            run,
            user,
            inputs=inputs,
            config_payload=config_payload,
            execution_mode="taskiq_worker",
            trigger_envelope=trigger_envelope,
        )
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.SUCCEEDED,
            finished_at=_now(),
            output=output,
        )
        await run.refresh_from_db()
        await WorkflowAnalytics._complete_run(run, output, metrics)
    except HTTPException:
        logger.exception(
            "Saved workflow run failed",
            extra={"run_id": run.run_id, "workflow_id": getattr(run.workflow, "workflow_id", None)},
        )
        raise
    except Exception:
        logger.exception(
            "Unexpected error during saved workflow run",
            extra={"run_id": run.run_id, "workflow_id": getattr(run.workflow, "workflow_id", None)},
        )
        raise
