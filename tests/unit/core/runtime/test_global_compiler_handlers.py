"""
Tests for LLM handler error detection in WorkflowCompilerSingleton.

These tests verify that the _build_text_handler and _build_json_handler methods
properly detect and raise errors for:
1. LLM responses with finish_reason="error"
2. Empty responses (None or empty dict/string)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from seer.core.compiler.parse import parse_workflow_spec
from seer.core.errors import ExecutionError, ValidationPhaseError
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton

pytestmark = pytest.mark.unit


def _create_mock_response(content: str = "Hello", finish_reason: str = "stop"):
    """Create a mock LangChain AIMessage-like response.

    Uses spec to prevent MagicMock from auto-creating arbitrary attributes
    like _last_response which would confuse the handler's attribute checks.
    """
    # Define the attributes we want to exist
    response = MagicMock(spec=["content", "response_metadata", "usage_metadata"])
    response.content = content
    response.response_metadata = {
        "finish_reason": finish_reason,
        "model_name": "test-model",
    }
    response.usage_metadata = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    return response


class TestTextHandlerErrorDetection:
    """Tests for _build_text_handler error detection."""

    @patch("seer.core.runtime.global_compiler.get_llm")
    def test_finish_reason_error_raises_execution_error(self, mock_get_llm):
        """Test that finish_reason='error' raises ExecutionError."""
        # Arrange: Mock LLM returns response with finish_reason="error"
        mock_llm = MagicMock()
        mock_response = _create_mock_response(content="", finish_reason="error")
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        compiler = WorkflowCompilerSingleton()
        handler = compiler._build_text_handler("test-model")  # pylint: disable=protected-access

        invocation = {"prompt": "Test prompt", "parameters": {}}

        # Act & Assert
        with pytest.raises(ExecutionError) as exc_info:
            handler(invocation)

        assert "finish_reason='error'" in str(exc_info.value)
        assert "test-model" in str(exc_info.value)

    @patch("seer.core.runtime.global_compiler.get_llm")
    def test_empty_text_response_raises_execution_error(self, mock_get_llm):
        """Test that empty text response raises ExecutionError."""
        # Arrange: Mock LLM returns empty content but with finish_reason="stop"
        mock_llm = MagicMock()
        mock_response = _create_mock_response(content="", finish_reason="stop")
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        compiler = WorkflowCompilerSingleton()
        handler = compiler._build_text_handler("test-model")  # pylint: disable=protected-access

        invocation = {"prompt": "Test prompt", "parameters": {}}

        # Act & Assert
        with pytest.raises(ExecutionError) as exc_info:
            handler(invocation)

        assert "empty text output" in str(exc_info.value)
        assert "test-model" in str(exc_info.value)

    @patch("seer.core.runtime.global_compiler.get_llm")
    def test_whitespace_only_response_raises_execution_error(self, mock_get_llm):
        """Test that whitespace-only text response raises ExecutionError."""
        mock_llm = MagicMock()
        mock_response = _create_mock_response(content="   \n\t  ", finish_reason="stop")
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        compiler = WorkflowCompilerSingleton()
        handler = compiler._build_text_handler("test-model")  # pylint: disable=protected-access

        invocation = {"prompt": "Test prompt", "parameters": {}}

        # Act & Assert
        with pytest.raises(ExecutionError) as exc_info:
            handler(invocation)

        assert "empty text output" in str(exc_info.value)

    @patch("seer.core.runtime.global_compiler.get_llm")
    def test_valid_text_response_succeeds(self, mock_get_llm):
        """Test that valid text response returns correctly."""
        mock_llm = MagicMock()
        mock_response = _create_mock_response(content="Hello, world!", finish_reason="stop")
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        compiler = WorkflowCompilerSingleton()
        handler = compiler._build_text_handler("test-model")  # pylint: disable=protected-access

        invocation = {"prompt": "Test prompt", "parameters": {}}

        # Act
        result, usage_metadata = handler(invocation)

        # Assert
        assert result == "Hello, world!"
        assert "input_tokens" in usage_metadata


class TestJsonHandlerErrorDetection:
    """Tests for _build_json_handler error detection."""

    @patch("seer.core.runtime.global_compiler.get_llm")
    def test_finish_reason_error_raises_execution_error(self, mock_get_llm):
        """Test that finish_reason='error' in structured output raises ExecutionError."""
        # Arrange: Mock structured LLM returns error
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()

        # Structured output returns dict directly, but _last_response has metadata
        mock_structured_llm.invoke.return_value = {}
        mock_underlying_response = _create_mock_response(content="", finish_reason="error")
        mock_structured_llm._last_response = mock_underlying_response

        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_llm.return_value = mock_llm

        compiler = WorkflowCompilerSingleton()
        handler = compiler._build_json_handler("test-model")  # pylint: disable=protected-access

        invocation = {"prompt": "Test prompt", "parameters": {}}
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Act & Assert
        with pytest.raises(ExecutionError) as exc_info:
            handler(invocation, schema)

        assert "finish_reason='error'" in str(exc_info.value)
        assert "test-model" in str(exc_info.value)

    @patch("seer.core.runtime.global_compiler.get_llm")
    def test_empty_dict_response_raises_execution_error(self, mock_get_llm):
        """Test that empty dict {} response raises ExecutionError."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()

        # Return empty dict with normal finish_reason
        mock_structured_llm.invoke.return_value = {}
        mock_underlying_response = _create_mock_response(content="", finish_reason="stop")
        mock_structured_llm._last_response = mock_underlying_response

        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_llm.return_value = mock_llm

        compiler = WorkflowCompilerSingleton()
        handler = compiler._build_json_handler("test-model")  # pylint: disable=protected-access

        invocation = {"prompt": "Test prompt", "parameters": {}}
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Act & Assert
        with pytest.raises(ExecutionError) as exc_info:
            handler(invocation, schema)

        assert "empty structured output" in str(exc_info.value)

    @patch("seer.core.runtime.global_compiler.get_llm")
    def test_none_response_raises_execution_error(self, mock_get_llm):
        """Test that None response raises ExecutionError."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()

        # Return None
        mock_structured_llm.invoke.return_value = None
        mock_underlying_response = _create_mock_response(content="", finish_reason="stop")
        mock_structured_llm._last_response = mock_underlying_response

        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_llm.return_value = mock_llm

        compiler = WorkflowCompilerSingleton()
        handler = compiler._build_json_handler("test-model")  # pylint: disable=protected-access

        invocation = {"prompt": "Test prompt", "parameters": {}}
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Act & Assert
        with pytest.raises(ExecutionError) as exc_info:
            handler(invocation, schema)

        assert "empty structured output" in str(exc_info.value)

    @patch("seer.core.runtime.global_compiler.get_llm")
    def test_valid_json_response_succeeds(self, mock_get_llm):
        """Test that valid JSON response returns correctly."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()

        # Return valid data
        mock_structured_llm.invoke.return_value = {"name": "Alice", "age": 30}
        mock_underlying_response = _create_mock_response(content="", finish_reason="stop")
        mock_structured_llm._last_response = mock_underlying_response

        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_get_llm.return_value = mock_llm

        compiler = WorkflowCompilerSingleton()
        handler = compiler._build_json_handler("test-model")  # pylint: disable=protected-access

        invocation = {"prompt": "Test prompt", "parameters": {}}
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # Act
        result, usage_metadata = handler(invocation, schema)

        # Assert
        assert result == {"name": "Alice", "age": 30}
        assert "input_tokens" in usage_metadata


class TestAgentModelValidation:
    """Tests for parse-time validation of allowed agent models."""

    def test_allowed_agent_model_passes_dependency_validation(self):
        spec = parse_workflow_spec(
            {
                "version": "2",
                "nodes": [
                    {
                        "id": "agent1",
                        "type": "agent",
                        "inputs": {
                            "model": "openai/gpt-oss-120b",
                            "prompt": "Summarize this",
                        },
                    }
                ],
                "edges": [],
                "triggers": [],
            }
        )

        compiler = WorkflowCompilerSingleton()
        compiler._ensure_dependencies(spec)  # pylint: disable=protected-access

    def test_disallowed_agent_model_raises_validation_phase_error_on_parse(self):
        with pytest.raises(ValidationPhaseError) as exc_info:
            parse_workflow_spec(
                {
                    "version": "2",
                    "nodes": [
                        {
                            "id": "agent1",
                            "type": "agent",
                            "inputs": {
                                "model": "openai/gpt-4o",
                                "prompt": "Summarize this",
                            },
                        }
                    ],
                    "edges": [],
                    "triggers": [],
                }
            )

        assert "not allowed" in str(exc_info.value)
        assert "openai/gpt-4o" in str(exc_info.value)
