# pylint: disable=too-many-lines  # Reason: Execution module covers full workflow lifecycle; refactor tracked separately
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import traceback
from fastapi import HTTPException

from langgraph.types import Command

from seer.api.agents.checkpointer import get_checkpointer
from seer.core.errors import ExecutionError, ProviderError, WorkflowCompilerError  # pylint: disable=no-name-in-module  # Reason: ProviderError defined in Task 6 - will exist after merge
from seer.database import WorkflowRun, User, WorkflowRunStatus
from seer.database.models import UserSettings
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.services.memory.runtime_adapter import WorkflowMemoryRuntimeAdapter
from seer.services.workflows.mcp_config_adapter import McpServerConfigResolverImpl
from seer.analytics.workflow_tracking import capture_workflow_run_event
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton

from seer.logger import get_logger
logger = get_logger(__name__)

def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _resolve_byok_credentials(runtime_context: WorkflowRuntimeContext) -> None:
    """Populate BYOK credentials on runtime_context if org is on BYOK plan."""
    if not runtime_context.organization_id:
        return
    from seer.database.byok_models import LLMApiKey  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.database.subscription_models import BillingSubscription, SubscriptionTier  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.services.byok.key_vault import get_key_vault  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    org_sub = await BillingSubscription.get_or_none(organization_id=runtime_context.organization_id)
    if not org_sub or org_sub.tier != SubscriptionTier.BYOK:
        return
    active_key = await LLMApiKey.get_or_none(
        organization_id=runtime_context.organization_id, is_active=True, status="active",
    )
    if not active_key:
        return
    vault = get_key_vault()
    decrypted = vault.decrypt(active_key.key_enc)
    if decrypted:
        runtime_context.byok_api_key = decrypted
        runtime_context.byok_base_url = active_key.base_url
        logger.info("BYOK key resolved for org %s", runtime_context.organization_id)


def _extract_hitl_interrupt(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract HITL interrupt payload from LangGraph result, or None."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    for interrupt_obj in interrupts:
        interrupt_value = getattr(interrupt_obj, "value", None)
        if isinstance(interrupt_value, dict) and interrupt_value.get("type") == "hitl":
            return interrupt_value
    return None


def _calculate_interrupt_expiry(timeout_seconds: Optional[int]) -> Optional[datetime]:
    """Calculate when the interrupt should expire based on timeout_seconds."""
    if not timeout_seconds or timeout_seconds <= 0:
        return None  # Indefinite wait
    return _now() + timedelta(seconds=timeout_seconds)


async def _send_hitl_notifications(
    run: WorkflowRun,
    user: User,
    interrupt_data: Dict[str, Any],
) -> None:
    """Send HITL notifications via delivery channels (fire-and-forget)."""
    delivery_channels = interrupt_data.get("delivery_channels", [])

    for channel in delivery_channels:
        channel_type = channel.get("type")

        if channel_type == "gmail" and channel.get("gmail"):
            # Import here to avoid circular dependency
            from seer.core.schema.models import GmailDeliveryConfig  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import
            from seer.services.workflows.hitl_email import send_hitl_gmail_notification  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

            gmail_config = GmailDeliveryConfig(**channel["gmail"])

            try:
                error = await send_hitl_gmail_notification(
                    user=user,
                    workflow_run=run,
                    interrupt_data=interrupt_data,
                    gmail_config=gmail_config,
                )
                if error:
                    logger.warning(
                        "HITL Gmail notification failed for run '%s': %s",
                        run.run_id,
                        error,
                        extra={"run_id": run.run_id, "error": error},
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught  # Intentional: notifications must not fail workflow
                # Log but don't fail - platform HITL still works
                logger.exception(
                    "Failed to send HITL Gmail notification for run '%s'",
                    run.run_id,
                    extra={"run_id": run.run_id, "error": str(exc)},
                )

        # Platform channel is the default - no action needed here
        # Users can poll GET /runs/{run_id}/interrupt to get HITL data


async def _compile_workflow(
    user: User,
    spec: Dict[str, Any],
    checkpointer: Optional[Any] = None,
    organization_id: Optional[int] = None,
) -> Any:
    """Compile a workflow spec using the global compiler singleton."""
    compiler = WorkflowCompilerSingleton.instance()
    return await compiler.compile(
        user,
        spec,
        checkpointer=checkpointer,
        organization_id=organization_id,
    )


def _build_run_config(run: WorkflowRun, config_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ensure LangGraph thread_id is set for checkpoint recovery."""
    base_config = dict((config_payload or {}) or {})
    configurable = dict((base_config.get("configurable") or {}) or {})
    # Always use run.run_id as thread_id for checkpoint retrieval consistency
    # Don't use setdefault - explicitly set to ensure it matches execution config
    configurable["thread_id"] = run.thread_id or run.run_id
    base_config["configurable"] = configurable
    return base_config


async def _execute_run(  # pylint: disable=too-many-locals,too-many-statements,too-complex  # Reason: Orchestrates full workflow run lifecycle including BYOK, cost caps, and error handling
    run: WorkflowRun,
    user: User,
    *,
    inputs: Dict[str, Any],
    config_payload: Dict[str, Any],
    trigger_envelope: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compile and execute a workflow run."""
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

    await capture_workflow_run_event("workflow_run_started", user.email, run.run_id, getattr(run.workflow, "workflow_id", None))
    checkpointer = await get_checkpointer()
    # Get organization_id from workflow for shared connection resolution
    organization_id = getattr(run.workflow, "organization_id", None)

    try:
        compiled = await _compile_workflow(user, run.spec, checkpointer=checkpointer, organization_id=organization_id)
    except WorkflowCompilerError as exc:
        logger.error("Workflow compilation failed", exc_info=True)
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
        )
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
        # Fetch user settings for cost cap
        user_settings, _ = await UserSettings.get_or_create(user=user)
        per_run_cost_cap_usd = user_settings.preferences.get("per_run_cost_cap_usd", 5.0)

        runtime_context = WorkflowRuntimeContext(
            user=user,
            workflow_run_id=run.run_id,
            thread_id=None,  # Not a chat thread
            per_run_cost_cap_usd=per_run_cost_cap_usd,
            accumulated_cost_usd=0.0,
            organization_id=organization_id,
            memory_access=WorkflowMemoryRuntimeAdapter(user=user, organization_id=organization_id),
            mcp_config_resolver=McpServerConfigResolverImpl(user=user),
        )

        await _resolve_byok_credentials(runtime_context)

        result = await compiled.ainvoke(
            config=effective_config,
            context=runtime_context,
            trigger=trigger_envelope,
        )

        # Check for HITL interrupt
        hitl_interrupt = _extract_hitl_interrupt(result)
        if hitl_interrupt:
            logger.info(
                "HITL interrupt detected for run '%s' at node '%s'",
                run.run_id,
                hitl_interrupt.get("node_id"),
                extra={
                    "run_id": run.run_id,
                    "node_id": hitl_interrupt.get("node_id"),
                    "title": hitl_interrupt.get("title"),
                },
            )
            timeout_seconds = hitl_interrupt.get("timeout_seconds")
            expires_at = _calculate_interrupt_expiry(timeout_seconds)

            await WorkflowRun.filter(id=run.id).update(
                status=WorkflowRunStatus.INTERRUPTED,
                pending_interrupt_node_id=hitl_interrupt.get("node_id"),
                pending_interrupt_data=hitl_interrupt,
                interrupt_expires_at=expires_at,
            )

            # Refresh run to get updated state for notification service
            await run.refresh_from_db()

            # Send HITL notifications via configured delivery channels
            await _send_hitl_notifications(run, user, hitl_interrupt)

            # Return result with interrupt flag for caller awareness
            return {"__interrupted__": True, "__interrupt_data__": hitl_interrupt, **result}

    except ProviderError as exc:
        logger.warning(
            "Provider error for workflow run '%s': %s",
            run.run_id,
            exc,
            extra={"run_id": run.run_id},
        )
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=f"[PROVIDER ERROR] {exc}",
        )
        await capture_workflow_run_event(
            "workflow_run_failed", user.email, run.run_id,
            getattr(run.workflow, "workflow_id", None), error=f"[PROVIDER ERROR] {exc}")
        raise

    except Exception as exc:
        # Conditional import here to avoid circular dependency during module initialization
        from seer.observability.exceptions import RunCostCapExceeded  # pylint: disable=import-outside-toplevel  # Reason: circular dependency

        # Handle cost cap exceeded with structured error
        if isinstance(exc, RunCostCapExceeded):
            logger.warning(
                "Run cost cap exceeded for workflow run '%s'",
                run.run_id,
                extra={
                    "run_id": run.run_id,
                    "accumulated_cost": exc.accumulated_cost,
                    "cost_cap": exc.cost_cap,
                },
            )
            await WorkflowRun.filter(id=run.id).update(
                status=WorkflowRunStatus.FAILED,
                finished_at=_now(),
                error=exc.to_dict(),
            )
            await capture_workflow_run_event(
                "workflow_run_failed", user.email, run.run_id,
                getattr(run.workflow, "workflow_id", None), error=str(exc))
            raise HTTPException(status_code=402, detail=exc.to_dict()) from exc

        # Handle other exceptions
        print(f"{traceback.format_exc()}")

        # Extract error trace from ExecutionError (persists node traces when checkpoints aren't written)
        node_traces = None
        if isinstance(exc, ExecutionError) and exc.trace_data:  # pylint: disable=no-member  # Reason: ExecutionError adds trace_data attribute in __init__
            node_traces = exc.trace_data  # pylint: disable=no-member  # Reason: ExecutionError adds trace_data attribute in __init__

        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
            node_traces=node_traces,
        )
        await capture_workflow_run_event(
            "workflow_run_failed", user.email, run.run_id,
            getattr(run.workflow, "workflow_id", None), error=str(exc))
        raise

    return result


async def _mark_run_succeeded(run: WorkflowRun, output: Dict[str, Any], user: Optional[User] = None) -> None:
    """Persist workflow success state and refresh the run instance."""
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.SUCCEEDED,
        finished_at=_now(),
        output=output,
        pending_interrupt_node_id=None,
        pending_interrupt_data=None,
        interrupt_expires_at=None,
    )
    await run.refresh_from_db()
    if user:
        await capture_workflow_run_event("workflow_run_completed", user.email, run.run_id, getattr(run.workflow, "workflow_id", None))


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
        output = await _execute_run(
            run,
            user,
            inputs=inputs,
            config_payload=config_payload,
            trigger_envelope=trigger_envelope,
        )
        # Check if workflow was interrupted (HITL)
        if output.get("__interrupted__"):
            logger.info(
                "Workflow run '%s' interrupted at HITL node",
                run.run_id,
                extra={"run_id": run.run_id},
            )
            # Run is already marked as INTERRUPTED by _execute_run
            return
        await _mark_run_succeeded(run, output, user=user)
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


async def _validate_resume_request(
    user: User,
    run: WorkflowRun,
) -> None:
    """Validate that resume request is valid."""
    # Verify ownership
    if run.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to resume this run")

    # Verify run is in INTERRUPTED state
    if run.status != WorkflowRunStatus.INTERRUPTED:
        raise HTTPException(
            status_code=400,
            detail=f"Run is not in INTERRUPTED state (current: {run.status})"
        )

    # Check if interrupt has expired
    if run.interrupt_expires_at and run.interrupt_expires_at < _now():
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error="HITL interrupt timed out",
            pending_interrupt_node_id=None,
            pending_interrupt_data=None,
            interrupt_expires_at=None,
        )
        raise HTTPException(status_code=408, detail="HITL interrupt has timed out")


async def _execute_resume(  # pylint: disable=too-many-locals  # Reason: Resume path carries same lifecycle state as execute including BYOK resolution
    run: WorkflowRun,
    user: User,
    compiled: Any,
    responses: Dict[str, Any],
    *,
    organization_id: Optional[int] = None,
    prev_interrupt_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the resume operation and handle result."""
    run_config = dict(run.config or {})
    effective_config = _build_run_config(run, run_config)

    # Fetch user settings for cost cap
    user_settings, _ = await UserSettings.get_or_create(user=user)
    per_run_cost_cap_usd = user_settings.preferences.get("per_run_cost_cap_usd", 5.0)

    runtime_context = WorkflowRuntimeContext(
        user=user,
        workflow_run_id=run.run_id,
        thread_id=None,
        per_run_cost_cap_usd=per_run_cost_cap_usd,
        accumulated_cost_usd=0.0,
        organization_id=organization_id,
        memory_access=WorkflowMemoryRuntimeAdapter(user=user, organization_id=organization_id),
        mcp_config_resolver=McpServerConfigResolverImpl(user=user),
    )

    await _resolve_byok_credentials(runtime_context)

    # Resume with user responses using LangGraph's Command
    resume_command = Command(resume=responses)
    result = await compiled.ainvoke(
        resume_command,
        config=effective_config,
        context=runtime_context,
    )

    # After resume, LangGraph may leave stale __interrupt__ data in the state
    # from the previous interrupt that was just resumed. Detect this by checking
    # if the interrupted node actually produced output (meaning it completed).
    hitl_interrupt = _extract_hitl_interrupt(result)
    if hitl_interrupt:
        interrupted_node_id = hitl_interrupt.get("node_id")
        # If the node that "interrupted" also has output in the result, it MAY be
        # stale (prior pause echo) or a NEW interrupt from the next loop iteration.
        # Distinguish by comparing display content: in a loop each iteration renders
        # different template values, so a truly stale interrupt has identical display.
        if interrupted_node_id and interrupted_node_id in result:
            if (prev_interrupt_data or {}).get("display") == hitl_interrupt.get("display"):
                logger.debug(
                    "Ignoring stale __interrupt__ after resume for run '%s' "
                    "(node '%s' has output, interrupt display unchanged)",
                    run.run_id,
                    interrupted_node_id,
                )
                hitl_interrupt = None
            else:
                logger.debug(
                    "Keeping new __interrupt__ for run '%s' "
                    "(node '%s' has output but display changed — new loop iteration)",
                    run.run_id,
                    interrupted_node_id,
                )
    if hitl_interrupt:
        logger.info(
            "Another HITL interrupt detected for run '%s' at node '%s'",
            run.run_id,
            hitl_interrupt.get("node_id"),
        )
        timeout_seconds = hitl_interrupt.get("timeout_seconds")
        expires_at = _calculate_interrupt_expiry(timeout_seconds)

        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.INTERRUPTED,
            pending_interrupt_node_id=hitl_interrupt.get("node_id"),
            pending_interrupt_data=hitl_interrupt,
            interrupt_expires_at=expires_at,
        )
        return {"__interrupted__": True, "__interrupt_data__": hitl_interrupt, **result}

    # Workflow completed successfully — strip stale __interrupt__ from output
    clean_output = {k: v for k, v in result.items() if k != "__interrupt__"}
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.SUCCEEDED,
        finished_at=_now(),
        output=clean_output,
        pending_interrupt_node_id=None,
        pending_interrupt_data=None,
        interrupt_expires_at=None,
    )
    return result


async def resume_workflow_run(
    user: User,
    run_id: str,
    responses: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resume a workflow run that is paused at an HITL interrupt.

    Args:
        user: The user resuming the run
        run_id: Public run ID (run_XXX format)
        responses: User responses keyed by input field ID

    Returns:
        Execution result or new interrupt data if another HITL node is reached

    Raises:
        HTTPException: If run is not found, not owned by user, or not in INTERRUPTED state
    """
    from seer.database.workflow_models import parse_run_public_id  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

    # Parse and fetch the run
    try:
        run_pk = parse_run_public_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}") from exc

    run = await WorkflowRun.get_or_none(id=run_pk)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    await run.fetch_related("user", "workflow")

    # Validate resume request
    await _validate_resume_request(user, run)

    logger.info(
        "Resuming workflow run '%s' with responses for node '%s'",
        run.run_id,
        run.pending_interrupt_node_id,
        extra={
            "run_id": run.run_id,
            "node_id": run.pending_interrupt_node_id,
            "response_keys": list(responses.keys()),
        },
    )

    # Save previous interrupt data before clearing (needed for stale-interrupt detection in loops)
    prev_interrupt_data = run.pending_interrupt_data

    # Mark run as running again
    await WorkflowRun.filter(id=run.id).update(
        status=WorkflowRunStatus.RUNNING,
        pending_interrupt_node_id=None,
        pending_interrupt_data=None,
        interrupt_expires_at=None,
    )

    # Compile workflow
    checkpointer = await get_checkpointer()
    # Get organization_id from workflow for shared connection resolution
    organization_id = getattr(run.workflow, "organization_id", None)

    try:
        compiled = await _compile_workflow(user, run.spec, checkpointer=checkpointer, organization_id=organization_id)
    except WorkflowCompilerError as exc:
        logger.error("Workflow compilation failed during resume", exc_info=True)
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Compilation failed: {exc}") from exc

    # Execute resume
    try:
        return await _execute_resume(run, user, compiled, responses, organization_id=organization_id, prev_interrupt_data=prev_interrupt_data)
    except Exception as exc:
        from seer.observability.exceptions import RunCostCapExceeded  # pylint: disable=import-outside-toplevel  # Reason: circular dependency

        if isinstance(exc, RunCostCapExceeded):
            logger.warning("Run cost cap exceeded during resume for '%s'", run.run_id)
            await WorkflowRun.filter(id=run.id).update(
                status=WorkflowRunStatus.FAILED,
                finished_at=_now(),
                error=exc.to_dict(),
            )
            raise HTTPException(status_code=402, detail=exc.to_dict()) from exc

        logger.exception("Error during workflow resume", extra={"run_id": run.run_id})
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.FAILED,
            finished_at=_now(),
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def get_workflow_run_interrupt(
    user: User,
    run_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get pending HITL interrupt data for a workflow run.

    Args:
        user: The user requesting interrupt data
        run_id: Public run ID (run_XXX format)

    Returns:
        Interrupt data dict if run is interrupted, None otherwise

    Raises:
        HTTPException: If run is not found or not owned by user
    """
    from seer.database.workflow_models import parse_run_public_id  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import

    try:
        run_pk = parse_run_public_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}") from exc

    run = await WorkflowRun.get_or_none(id=run_pk)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    await run.fetch_related("user")

    if run.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this run")

    if run.status != WorkflowRunStatus.INTERRUPTED:
        return None

    # Check if expired
    is_expired = run.interrupt_expires_at and run.interrupt_expires_at < _now()

    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "node_id": run.pending_interrupt_node_id,
        "interrupt_data": run.pending_interrupt_data,
        "expires_at": run.interrupt_expires_at.isoformat() if run.interrupt_expires_at else None,
        "is_expired": is_expired,
    }
