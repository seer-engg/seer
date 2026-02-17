"""
Unit tests for centralized cost tracking utilities.
"""
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.runtime.context import WorkflowRuntimeContext
from seer.database.models import User
from seer.observability.cost_tracking import CostTracker, detect_provider_from_model
from seer.observability.exceptions import RunCostCapExceeded


@pytest.mark.unit
class TestDetectProviderFromModel:
    """Tests for provider detection from model names."""

    def test_openai_gpt_models(self):
        """Test OpenAI GPT model detection."""
        assert detect_provider_from_model("gpt-4o") == "openai"
        assert detect_provider_from_model("gpt-4o-mini") == "openai"
        assert detect_provider_from_model("gpt-5") == "openai"
        assert detect_provider_from_model("gpt-5-nano") == "openai"

    def test_openai_o3_models(self):
        """Test OpenAI o3 model detection."""
        assert detect_provider_from_model("o3-mini") == "openai"

    def test_openai_o1_models(self):
        """Test OpenAI o1 model detection."""
        assert detect_provider_from_model("o1-preview") == "openai"
        assert detect_provider_from_model("o1-mini") == "openai"

    def test_anthropic_claude_models(self):
        """Test Anthropic Claude model detection."""
        assert detect_provider_from_model("claude-sonnet-4.5") == "anthropic"
        assert detect_provider_from_model("claude-opus-4.5") == "anthropic"
        assert detect_provider_from_model("claude-3-opus-20240229") == "anthropic"
        assert detect_provider_from_model("claude-3-5-sonnet-20241022") == "anthropic"

    def test_openrouter_moonshotai_models(self):
        """Test OpenRouter moonshotai model detection (used by browser node)."""
        assert detect_provider_from_model("moonshotai/kimi-k2.5") == "openrouter"
        assert detect_provider_from_model("moonshotai/kimi-k2-thinking") == "openrouter"

    def test_unknown_models(self):
        """Test unknown model detection."""
        assert detect_provider_from_model("llama-3") == "unknown"
        assert detect_provider_from_model("gemini-pro") == "unknown"
        assert detect_provider_from_model("custom-model") == "unknown"


@pytest.mark.asyncio
@pytest.mark.unit
class TestCostTracker:
    """Tests for CostTracker.track_and_enforce_cap."""

    # Note: mock_user fixture is provided by tests/unit/conftest.py

    @pytest.fixture
    def runtime_context(self, mock_user):
        """Create runtime context for testing."""
        return WorkflowRuntimeContext(
            user=mock_user,
            workflow_run_id="test-run-123",
            thread_id="test-thread-456",
            per_run_cost_cap_usd=1.0,
            accumulated_cost_usd=0.0,
        )

    @pytest.fixture
    def usage_metadata(self):
        """Create usage metadata fixture."""
        return {
            "model": "gpt-4o",
            "input_tokens": 1000,
            "output_tokens": 500,
            "reasoning_tokens": 0,
        }

    async def test_track_and_enforce_cap_success(self, runtime_context, usage_metadata):
        """Test successful cost tracking and cap enforcement."""
        import asyncio

        with patch("seer.observability.cost_tracking.track_llm_usage", new_callable=AsyncMock) as mock_track:
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=runtime_context,
                operation="workflow_execution",
            )

            # Wait for async task to complete
            await asyncio.sleep(0.1)

            # Verify cost was accumulated
            assert runtime_context.accumulated_cost_usd > 0

            # Verify tracking was called
            mock_track.assert_called_once()
            call_kwargs = mock_track.call_args.kwargs
            assert call_kwargs["user"] == runtime_context.user
            assert call_kwargs["provider"] == "openai"
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["input_tokens"] == 1000
            assert call_kwargs["output_tokens"] == 500

    async def test_cost_cap_exceeded_workflow(self, runtime_context, usage_metadata):
        """Test cost cap exceeded for workflow run."""
        # Set cap very low to trigger exception
        runtime_context.per_run_cost_cap_usd = 0.001

        with patch("seer.observability.cost_tracking.track_llm_usage", new_callable=AsyncMock):
            with pytest.raises(RunCostCapExceeded) as exc_info:
                await CostTracker.track_and_enforce_cap(
                    usage_metadata=usage_metadata,
                    context=runtime_context,
                    operation="workflow_execution",
                )

            # Verify exception details
            exc = exc_info.value
            assert exc.run_identifier == "test-run-123"
            assert exc.run_type == "workflow"
            assert exc.accumulated_cost > exc.cost_cap

    async def test_cost_cap_exceeded_chat(self, runtime_context, usage_metadata):
        """Test cost cap exceeded for chat."""
        # Make it a chat context (no workflow_run_id)
        runtime_context.workflow_run_id = None
        runtime_context.per_run_cost_cap_usd = 0.001

        with patch("seer.observability.cost_tracking.track_llm_usage", new_callable=AsyncMock):
            with pytest.raises(RunCostCapExceeded) as exc_info:
                await CostTracker.track_and_enforce_cap(
                    usage_metadata=usage_metadata,
                    context=runtime_context,
                    operation="chat_message",
                )

            # Verify exception details
            exc = exc_info.value
            assert exc.run_identifier == "test-thread-456"
            assert exc.run_type == "chat"

    async def test_anthropic_model_provider_detection(self, runtime_context, usage_metadata):
        """Test provider detection for Anthropic models."""
        import asyncio

        usage_metadata["model"] = "claude-sonnet-4.5"

        with patch("seer.observability.cost_tracking.track_llm_usage", new_callable=AsyncMock) as mock_track:
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=runtime_context,
                operation="workflow_execution",
            )

            await asyncio.sleep(0.1)
            call_kwargs = mock_track.call_args.kwargs
            assert call_kwargs["provider"] == "anthropic"

    async def test_reasoning_tokens_included(self, runtime_context, usage_metadata):
        """Test that reasoning tokens are included in cost calculation."""
        import asyncio

        usage_metadata["reasoning_tokens"] = 2000

        with patch("seer.observability.cost_tracking.track_llm_usage", new_callable=AsyncMock) as mock_track:
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=runtime_context,
                operation="workflow_execution",
            )

            await asyncio.sleep(0.1)
            # Verify reasoning tokens in metadata
            call_kwargs = mock_track.call_args.kwargs
            assert call_kwargs["metadata"]["reasoning_tokens"] == 2000

    async def test_extra_metadata_merged(self, runtime_context, usage_metadata):
        """Test that extra metadata is merged correctly."""
        import asyncio

        extra_metadata = {"custom_field": "custom_value"}

        with patch("seer.observability.cost_tracking.track_llm_usage", new_callable=AsyncMock) as mock_track:
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=runtime_context,
                operation="workflow_execution",
                extra_metadata=extra_metadata,
            )

            await asyncio.sleep(0.1)
            call_kwargs = mock_track.call_args.kwargs
            assert call_kwargs["metadata"]["custom_field"] == "custom_value"
            assert call_kwargs["metadata"]["reasoning_tokens"] == 0

    async def test_no_cost_cap_set(self, runtime_context, usage_metadata):
        """Test behavior when no cost cap is set."""
        runtime_context.per_run_cost_cap_usd = None

        with patch("seer.observability.cost_tracking.track_llm_usage", new_callable=AsyncMock):
            # Should not raise exception even with high usage
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=runtime_context,
                operation="workflow_execution",
            )

            # Cost should still be accumulated
            assert runtime_context.accumulated_cost_usd > 0

    async def test_accumulated_cost_increments(self, runtime_context, usage_metadata):
        """Test that accumulated cost increments across multiple calls."""
        import asyncio

        with patch("seer.observability.cost_tracking.track_llm_usage", new_callable=AsyncMock):
            # First call
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=runtime_context,
                operation="workflow_execution",
            )
            first_cost = runtime_context.accumulated_cost_usd

            # Second call
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=runtime_context,
                operation="workflow_execution",
            )
            second_cost = runtime_context.accumulated_cost_usd

            await asyncio.sleep(0.1)
            # Cost should have doubled
            assert second_cost > first_cost
            assert abs(second_cost - (first_cost * 2)) < 0.000001

    async def test_tracking_failure_logged(self, runtime_context, usage_metadata, caplog):
        """Test that tracking failures are logged but don't raise."""
        with patch(
            "seer.observability.cost_tracking.track_llm_usage",
            new_callable=AsyncMock,
            side_effect=Exception("Database error"),
        ):
            # Should not raise exception
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=runtime_context,
                operation="workflow_execution",
            )

            # Cost should still be accumulated even if tracking fails
            assert runtime_context.accumulated_cost_usd > 0
