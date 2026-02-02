"""Unit tests for CostCapCallbackHandler."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from seer.agents.nexus.cost_callback import (
    CostCapCallbackHandler,
    clear_chat_runtime_context,
    get_chat_runtime_context,
    set_chat_runtime_context,
)
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.database.models import User


@pytest.fixture
def mock_user():
    """Fixture providing a mock user."""
    user = MagicMock(spec=User)
    user.user_id = "test_user_123"
    return user


@pytest.fixture
def mock_context(mock_user):
    """Fixture providing a mock runtime context."""
    context = MagicMock(spec=WorkflowRuntimeContext)
    context.user = mock_user
    context.thread_id = "test_thread_456"
    context.workflow_run_id = None
    context.accumulated_cost_usd = 0.0
    context.per_run_cost_cap_usd = 1.0
    return context


@pytest.fixture
def callback_handler():
    """Fixture providing a callback handler instance."""
    return CostCapCallbackHandler()


@pytest.fixture
def llm_result_with_usage():
    """Fixture providing an LLM result with usage metadata."""
    message = AIMessage(
        content="Test response",
        response_metadata={
            "model_name": "gpt-4o",
        },
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
    )
    generation = ChatGeneration(message=message)
    result = LLMResult(
        generations=[[generation]],
        llm_output={"model_name": "gpt-4o"},
    )
    return result


def test_set_and_get_chat_runtime_context(mock_context):
    """Test setting and getting chat runtime context."""
    # Initially should be None
    assert get_chat_runtime_context() is None

    # Set context
    set_chat_runtime_context(mock_context)

    # Should now return the context
    assert get_chat_runtime_context() is mock_context

    # Clear context
    clear_chat_runtime_context()

    # Should be None again
    assert get_chat_runtime_context() is None


def test_extract_model_name_from_llm_output(callback_handler):
    """Test extracting model name from llm_output."""
    result = LLMResult(
        generations=[[]],
        llm_output={"model_name": "gpt-4o"},
    )

    model = callback_handler._extract_model_name(result)
    assert model == "gpt-4o"


def test_extract_model_name_from_response_metadata(callback_handler):
    """Test extracting model name from response_metadata."""
    message = AIMessage(
        content="Test",
        response_metadata={"model_name": "claude-3-5-sonnet"},
    )
    generation = ChatGeneration(message=message)
    result = LLMResult(
        generations=[[generation]],
        llm_output={},
    )

    model = callback_handler._extract_model_name(result)
    assert model == "claude-3-5-sonnet"


def test_extract_model_name_fallback_to_unknown(callback_handler):
    """Test that model name falls back to 'unknown' when not found."""
    result = LLMResult(
        generations=[[]],
        llm_output={},
    )

    model = callback_handler._extract_model_name(result)
    assert model == "unknown"


def test_extract_usage_from_response_with_empty_generations(callback_handler):
    """Test extracting usage from response with empty generations."""
    result = LLMResult(generations=[])

    usage = callback_handler._extract_usage_from_response(result, "gpt-4o")
    assert usage is None


def test_extract_usage_from_response_with_no_message(callback_handler):
    """Test extracting usage from response with no message."""
    result = LLMResult(generations=[[]])

    usage = callback_handler._extract_usage_from_response(result, "gpt-4o")
    assert usage is None


def test_on_llm_end_with_no_runtime_context(callback_handler, llm_result_with_usage):
    """Test on_llm_end with no runtime context doesn't crash."""
    clear_chat_runtime_context()

    # Should not raise exception, just log warning
    callback_handler.on_llm_end(llm_result_with_usage)


def test_on_llm_end_with_no_usage_metadata(callback_handler, mock_context):
    """Test on_llm_end with no usage metadata doesn't crash."""
    set_chat_runtime_context(mock_context)

    # Create result without usage metadata
    message = AIMessage(content="Test", response_metadata={"model_name": "gpt-4o"})
    generation = ChatGeneration(message=message)
    result = LLMResult(generations=[[generation]], llm_output={"model_name": "gpt-4o"})

    with patch("seer.agents.nexus.cost_callback.extract_usage_metadata", return_value=None):
        # Should not raise exception, just log warning
        callback_handler.on_llm_end(result)

    clear_chat_runtime_context()


@pytest.mark.asyncio
async def test_on_llm_end_schedules_tracking(callback_handler, mock_context, llm_result_with_usage):
    """Test on_llm_end schedules cost tracking."""
    set_chat_runtime_context(mock_context)

    # Mock schedule_async_task to verify it's called
    with patch("seer.agents.nexus.cost_callback.schedule_async_task") as mock_schedule:
        callback_handler.on_llm_end(llm_result_with_usage)

        # Verify schedule_async_task was called
        mock_schedule.assert_called_once()
        call_kwargs = mock_schedule.call_args.kwargs
        assert "coro" in call_kwargs
        assert "logger" in call_kwargs
        assert call_kwargs["error_message"] == "Failed to schedule chat LLM usage tracking"

    clear_chat_runtime_context()


def test_callback_handler_raise_error_flag(callback_handler):
    """Test that callback handler has raise_error flag set to True."""
    assert callback_handler.raise_error is True
