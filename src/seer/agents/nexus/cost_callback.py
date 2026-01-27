"""Cost cap callback handler for Nexus chat agents."""
import asyncio
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from seer.core.runtime.context import WorkflowRuntimeContext
from seer.observability.exceptions import RunCostCapExceeded
from seer.observability.llm import extract_usage_metadata
from seer.logger import get_logger

logger = get_logger(__name__)

# Thread-local context variable for storing runtime context
_chat_runtime_context: ContextVar[WorkflowRuntimeContext | None] = ContextVar(
    '_chat_runtime_context',
    default=None
)


def set_chat_runtime_context(context: WorkflowRuntimeContext) -> None:
    """Set the runtime context for the current chat thread."""
    _chat_runtime_context.set(context)


def get_chat_runtime_context() -> WorkflowRuntimeContext | None:
    """Get the runtime context for the current chat thread."""
    return _chat_runtime_context.get()


def clear_chat_runtime_context() -> None:
    """Clear the runtime context for the current chat thread."""
    _chat_runtime_context.set(None)


class CostCapCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that tracks LLM token usage and enforces per-run cost caps.

    This handler:
    1. Intercepts LLM responses via on_llm_end()
    2. Extracts token usage metadata
    3. Calculates cost using existing pricing tables
    4. Accumulates cost in WorkflowRuntimeContext
    5. Checks against per_run_cost_cap_usd
    6. Raises RunCostCapExceeded if cap is exceeded
    7. Tracks usage to database asynchronously
    """

    # Tell LangChain to propagate exceptions instead of catching them
    raise_error: bool = True

    @staticmethod
    def _extract_model_name(response: LLMResult) -> str:
        """
        Extract model name from LLM response with fallback strategies.

        Args:
            response: LLMResult containing generations and metadata

        Returns:
            Model name, or "unknown" if not found
        """
        # Try llm_output first
        model = None
        if response.llm_output:
            model = response.llm_output.get("model_name")
        if not model and hasattr(response, "model_name"):
            model = response.model_name

        # Fallback: extract from first generation if available
        if not model and response.generations and len(response.generations) > 0:
            first_gen = response.generations[0]
            if len(first_gen) > 0 and hasattr(first_gen[0], "message"):
                msg = first_gen[0].message
                if hasattr(msg, "response_metadata"):
                    model = msg.response_metadata.get("model_name")

        if not model:
            logger.warning("Could not determine model name from LLM response")
            return "unknown"

        return model

    @staticmethod
    def _extract_usage_from_response(response: LLMResult, model: str) -> dict | None:
        """
        Extract usage metadata from LLM response.

        Args:
            response: LLMResult containing generations and metadata
            model: Model name to include in usage metadata

        Returns:
            Usage metadata dict with model, input_tokens, output_tokens, reasoning_tokens
            or None if extraction fails
        """
        if not response.generations or len(response.generations) == 0:
            return None

        first_gen = response.generations[0]
        if len(first_gen) == 0 or not hasattr(first_gen[0], "message"):
            return None

        return extract_usage_metadata(first_gen[0].message, model)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Called when LLM completes. Extract tokens, calculate cost, check cap.

        Args:
            response: LLMResult containing generations and metadata
            **kwargs: Additional callback arguments
        """
        from seer.observability.cost_tracking import CostTracker  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import at module level

        # Get runtime context
        context = get_chat_runtime_context()
        if not context or not context.user:
            logger.warning("No runtime context available for cost tracking")
            return

        # Extract model name
        model = self._extract_model_name(response)

        # Extract usage metadata
        usage_metadata = self._extract_usage_from_response(response, model)
        if not usage_metadata:
            logger.warning("No usage metadata available in LLM response")
            return

        try:
            # Delegate to shared cost tracking utility
            asyncio.create_task(
                CostTracker.track_and_enforce_cap(
                    usage_metadata=usage_metadata,
                    context=context,
                    operation="chat_message",
                )
            )
        except RunCostCapExceeded:
            # Re-raise cost cap exception immediately
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Don't fail chat for tracking errors
            # Log but don't fail chat for tracking errors
            logger.error(
                "Failed to track chat LLM usage: %s",
                str(e),
                exc_info=True,
                extra={
                    "user_id": context.user.user_id,
                    "thread_id": context.thread_id,
                    "model": usage_metadata.get("model") if usage_metadata else None,
                },
            )
