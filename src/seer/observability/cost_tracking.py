"""
Centralized cost tracking utilities for LLM usage.

Provides shared functionality for:
- Provider detection from model names
- Cost calculation and accumulation
- Per-run cost cap enforcement
- Async usage tracking with cross-thread event loop handling
"""
import asyncio
import threading
from typing import Any, Dict

from seer.core.event_loop import get_main_event_loop
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.observability.credit_calculator import calculate_cost
from seer.observability.exceptions import RunCostCapExceeded
from seer.observability.tracking import track_llm_usage
from seer.logger import get_logger

logger = get_logger(__name__)


def detect_provider_from_model(model: str) -> str:
    """
    Detect LLM provider from model name.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4.5")

    Returns:
        Provider name: "openai", "anthropic", or "unknown"
    """
    if model.startswith(("gpt-", "o3-", "o1-")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    return "unknown"


class CostTracker:
    """Centralized cost tracking with cap enforcement and async persistence."""

    @staticmethod
    async def track_and_enforce_cap(  # pylint: disable=too-complex  # Reason: complexity from cross-thread event loop handling, essential for correctness
        usage_metadata: Dict[str, Any],
        context: WorkflowRuntimeContext,
        operation: str = "workflow_execution",
        extra_metadata: Dict[str, Any] | None = None,
    ) -> None:
        """
        Calculate cost, accumulate, check cap, and track usage asynchronously.

        This method:
        1. Calculates cost from usage metadata
        2. Accumulates cost in runtime context
        3. Checks against per-run cost cap (raises if exceeded)
        4. Schedules async tracking to database (fire and forget)

        Args:
            usage_metadata: Dict with model, input_tokens, output_tokens, reasoning_tokens
            context: Runtime context containing user, cost cap, and accumulated cost
            operation: Operation type (e.g., "workflow_execution", "chat_message")
            extra_metadata: Additional metadata to include in tracking record

        Raises:
            RunCostCapExceeded: If accumulated cost exceeds per_run_cost_cap_usd
        """
        # Calculate cost
        cost = calculate_cost(
            model=usage_metadata["model"],
            input_tokens=usage_metadata["input_tokens"],
            output_tokens=usage_metadata["output_tokens"],
            reasoning_tokens=usage_metadata.get("reasoning_tokens", 0),
        )
        cost_float = float(cost)

        # Accumulate cost in context
        context.accumulated_cost_usd += cost_float

        # Check against per-run cost cap
        cost_cap = context.per_run_cost_cap_usd
        if cost_cap and context.accumulated_cost_usd > cost_cap:
            run_id = (
                context.workflow_run_id
                or context.thread_id
                or "unknown"
            )
            run_type = "workflow" if context.workflow_run_id else "chat"

            raise RunCostCapExceeded(
                run_identifier=run_id,
                accumulated_cost=context.accumulated_cost_usd,
                cost_cap=cost_cap,
                run_type=run_type,
            )

        # Detect provider from model name
        model = usage_metadata["model"]
        provider = detect_provider_from_model(model)

        # Prepare tracking metadata
        tracking_metadata = {
            "reasoning_tokens": usage_metadata.get("reasoning_tokens", 0),
        }
        if extra_metadata:
            tracking_metadata.update(extra_metadata)
        if context.thread_id:
            tracking_metadata["thread_id"] = context.thread_id

        # Define async tracking coroutine
        async def do_track():
            try:
                await track_llm_usage(
                    user=context.user,
                    provider=provider,
                    model=model,
                    input_tokens=usage_metadata["input_tokens"],
                    output_tokens=usage_metadata["output_tokens"],
                    cost=cost,
                    workflow_run_id=context.workflow_run_id,
                    operation=operation,
                    metadata=tracking_metadata,
                )
                logger.debug("Successfully tracked LLM usage to database")
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: tracking failures should not break workflow execution
                logger.error("Failed to track LLM usage to database: %s", e, exc_info=True)

        # Schedule tracking with cross-thread-safe event loop handling
        try:
            # Try to get running loop (will raise RuntimeError if not in async context)
            try:
                loop = asyncio.get_running_loop()
                # We're in async context - schedule task normally
                asyncio.create_task(do_track())
            except RuntimeError:
                # Not in async context - check if we're in main thread or thread pool
                if threading.current_thread() is threading.main_thread():
                    # Main thread - get or create event loop and run synchronously
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(do_track())
                else:
                    # Thread pool executor - schedule on main loop
                    main_loop = get_main_event_loop()
                    if main_loop is not None:
                        # Use run_coroutine_threadsafe to schedule on main loop from thread pool
                        asyncio.run_coroutine_threadsafe(do_track(), main_loop)
                    else:
                        logger.error("Main event loop not available - cannot track LLM usage from thread pool")
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: scheduling failures should not break workflow execution
            logger.error("Failed to schedule LLM usage tracking: %s", e, exc_info=True)

        logger.debug(
            "LLM usage: model=%s, input=%d, output=%d, cost=$%.6f, accumulated=$%.6f",
            model,
            usage_metadata["input_tokens"],
            usage_metadata["output_tokens"],
            cost_float,
            context.accumulated_cost_usd,
        )
