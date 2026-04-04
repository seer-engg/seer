"""
Centralized cost tracking utilities for LLM usage.

Provides shared functionality for:
- Provider detection from model names
- Cost calculation and accumulation
- Per-run cost cap enforcement
- Async usage tracking with cross-thread event loop handling
"""
from typing import Any, Dict, Optional

from seer.core.event_loop import schedule_async_task
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.database.organization_models import Organization
from seer.observability.credit_calculator import calculate_cost
from seer.observability.exceptions import RunCostCapExceeded
from seer.observability.tracking import track_llm_usage
from seer.logger import get_logger

logger = get_logger(__name__)


def detect_provider_from_model(model: str) -> str:
    """
    Detect LLM provider from model name.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4.5", "moonshotai/kimi-k2.5")

    Returns:
        Provider name: "openai", "anthropic", "openrouter", or "unknown"
    """
    if model.startswith(("gpt-", "o3-", "o1-", "openai/")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("moonshotai/"):
        return "openrouter"
    return "unknown"


class CostTracker:
    """Centralized cost tracking with cap enforcement and async persistence."""

    @staticmethod
    async def track_and_enforce_cap(
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

        # Resolve organization from context for team-level tracking
        organization: Optional[Organization] = None
        if context.organization_id:
            organization = await Organization.get_or_none(id=context.organization_id)

        # Determine if this call used the user's own API key (BYOK)
        is_byok = bool(context.byok_api_key)

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
                    organization=organization,  # Pass org for team-level tracking
                    is_byok=is_byok,
                )
                logger.debug("Successfully tracked LLM usage to database")
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: tracking failures should not break workflow execution
                logger.error("Failed to track LLM usage to database: %s", e, exc_info=True)

        # Schedule tracking with cross-thread-safe async handling
        schedule_async_task(
            coro=do_track(),
            logger=logger,
            error_message="Failed to schedule LLM usage tracking",
        )

        logger.debug(
            "LLM usage: model=%s, input=%d, output=%d, cost=$%.6f, accumulated=$%.6f",
            model,
            usage_metadata["input_tokens"],
            usage_metadata["output_tokens"],
            cost_float,
            context.accumulated_cost_usd,
        )
