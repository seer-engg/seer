"""Tests for BrowserService - browser task execution."""
import asyncio
import base64
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from seer.services.browser.browser_service import (
    BrowserService,
    _json_type_to_python,
    _strip_markdown_fencing,
    format_browser_history,
    json_schema_to_pydantic,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure singleton is reset between tests."""
    BrowserService._instance = None
    yield
    BrowserService._instance = None


@pytest.fixture
def browser_service():
    """Create a fresh BrowserService instance."""
    return BrowserService()


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock()
    user.user_id = "user-123"
    user.id = "user-123"
    return user


@pytest.fixture
def sample_extraction_schema():
    """Sample JSON schema for structured output."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "in_stock": {"type": "boolean"},
        },
        "required": ["name", "price"],
    }


@pytest.fixture
def mock_agent_history():
    """Mock BrowserUse agent history with usage data."""
    history = MagicMock()
    history.usage = MagicMock()
    history.usage.total_tokens = 1500
    history.usage.total_prompt_tokens = 1000
    history.usage.total_completion_tokens = 500
    history.usage.entry_count = 5
    history.final_result = MagicMock(return_value='{"name": "Widget", "price": 29.99}')
    history.screenshots = MagicMock(return_value=[])
    history.__str__ = MagicMock(return_value="Task completed successfully")

    # Methods needed for format_browser_history
    history.action_names = MagicMock(return_value=["navigate", "click"])

    mock_result = MagicMock()
    mock_result.extracted_content = "Clicked the login button"
    mock_result.long_term_memory = None
    mock_result.error = None
    history.action_results = MagicMock(return_value=[mock_result])

    history.is_done = MagicMock(return_value=True)
    history.is_successful = MagicMock(return_value=True)
    history.has_errors = MagicMock(return_value=False)
    return history


@pytest.fixture
def mock_file_system():
    """Mock WorkflowFileSystem for screenshot saving."""
    fs = MagicMock()
    fs.store_file_with_record = AsyncMock()
    return fs


@pytest.mark.unit
class TestStripMarkdownFencing:
    """Test _strip_markdown_fencing function.

    This function fixes the issue where LLMs return JSON wrapped in markdown
    code fences (```json ... ```), which breaks JSON parsing.
    Fixes Sentry issue PYTHON-FASTAPI-V.
    """

    def test_strip_json_fencing(self):
        """Test stripping ```json prefix and ``` suffix."""
        text = '```json\n{"name": "test", "value": 42}\n```'
        result = _strip_markdown_fencing(text)
        assert result == '{"name": "test", "value": 42}'

    def test_strip_json_fencing_uppercase(self):
        """Test stripping ```JSON (uppercase) prefix."""
        text = '```JSON\n{"key": "value"}\n```'
        result = _strip_markdown_fencing(text)
        assert result == '{"key": "value"}'

    def test_strip_generic_fencing(self):
        """Test stripping generic ``` fencing without language tag."""
        text = '```\n{"data": [1, 2, 3]}\n```'
        result = _strip_markdown_fencing(text)
        assert result == '{"data": [1, 2, 3]}'

    def test_no_fencing_passthrough(self):
        """Test that clean JSON passes through unchanged."""
        text = '{"already": "clean"}'
        result = _strip_markdown_fencing(text)
        assert result == '{"already": "clean"}'

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is stripped."""
        text = '  \n```json\n{"test": true}\n```  \n'
        result = _strip_markdown_fencing(text)
        assert result == '{"test": true}'

    def test_only_closing_fence(self):
        """Test handling when only closing fence is present."""
        text = '{"value": 1}\n```'
        result = _strip_markdown_fencing(text)
        assert result == '{"value": 1}'

    def test_only_opening_fence(self):
        """Test handling when only opening fence is present."""
        text = '```json\n{"value": 1}'
        result = _strip_markdown_fencing(text)
        assert result == '{"value": 1}'

    def test_real_world_example_from_sentry(self):
        """Test with actual format seen in Sentry error PYTHON-FASTAPI-V."""
        text = '''```json
{
  "thinking": "analyzing the page",
  "shops": [{"name": "Store A", "address": "123 Main St"}]
}
```'''
        result = _strip_markdown_fencing(text)
        # Should be valid JSON after stripping
        import json
        parsed = json.loads(result)
        assert "thinking" in parsed
        assert "shops" in parsed


@pytest.mark.unit
class TestFormatBrowserHistory:
    """Test format_browser_history function.

    This function transforms verbose AgentHistoryList output into a
    human-readable summary for the /history API response.
    """

    def test_format_none_history(self):
        """Test formatting None history returns empty structure."""
        result = format_browser_history(None)
        assert result == {"steps": [], "completed": False, "success": False}

    def test_format_with_steps(self):
        """Test formatting history with multiple steps."""
        mock_history = MagicMock()
        mock_history.action_names.return_value = ["navigate", "input_text", "click"]

        mock_result1 = MagicMock()
        mock_result1.extracted_content = "Navigated to https://example.com"
        mock_result1.long_term_memory = None
        mock_result1.error = None

        mock_result2 = MagicMock()
        mock_result2.extracted_content = "Typed 'search query'"
        mock_result2.long_term_memory = None
        mock_result2.error = None

        mock_result3 = MagicMock()
        mock_result3.extracted_content = "Clicked Search button"
        mock_result3.long_term_memory = None
        mock_result3.error = None

        mock_history.action_results.return_value = [mock_result1, mock_result2, mock_result3]
        mock_history.is_done.return_value = True
        mock_history.is_successful.return_value = True
        mock_history.has_errors.return_value = False

        result = format_browser_history(mock_history)

        assert len(result["steps"]) == 3
        assert result["steps"][0]["action"] == "navigate"
        assert result["steps"][0]["description"] == "Navigated to https://example.com"
        assert result["steps"][1]["action"] == "input_text"
        assert result["steps"][2]["action"] == "click"
        assert result["completed"] is True
        assert result["success"] is True

    def test_format_with_errors(self):
        """Test formatting history that contains errors."""
        mock_history = MagicMock()
        mock_history.action_names.return_value = ["click"]

        mock_result = MagicMock()
        mock_result.extracted_content = "Attempted to click button"
        mock_result.long_term_memory = None
        mock_result.error = "Element not found"

        mock_history.action_results.return_value = [mock_result]
        mock_history.is_done.return_value = True
        mock_history.is_successful.return_value = False
        mock_history.has_errors.return_value = True
        mock_history.errors.return_value = ["Element not found"]

        result = format_browser_history(mock_history)

        assert result["steps"][0]["error"] == "Element not found"
        assert result["success"] is False
        assert "errors" in result
        assert "Element not found" in result["errors"]

    def test_format_uses_long_term_memory_fallback(self):
        """Test that long_term_memory is used when extracted_content is None."""
        mock_history = MagicMock()
        mock_history.action_names.return_value = ["navigate"]

        mock_result = MagicMock()
        mock_result.extracted_content = None
        mock_result.long_term_memory = "Navigated to target page"
        mock_result.error = None

        mock_history.action_results.return_value = [mock_result]
        mock_history.is_done.return_value = True
        mock_history.is_successful.return_value = True
        mock_history.has_errors.return_value = False

        result = format_browser_history(mock_history)

        assert result["steps"][0]["description"] == "Navigated to target page"

    def test_format_fallback_description(self):
        """Test fallback description when both extracted_content and long_term_memory are None."""
        mock_history = MagicMock()
        mock_history.action_names.return_value = ["click"]

        mock_result = MagicMock()
        mock_result.extracted_content = None
        mock_result.long_term_memory = None
        mock_result.error = None

        mock_history.action_results.return_value = [mock_result]
        mock_history.is_done.return_value = True
        mock_history.is_successful.return_value = True
        mock_history.has_errors.return_value = False

        result = format_browser_history(mock_history)

        assert result["steps"][0]["description"] == "Performed click action"

    def test_format_handles_missing_action_names(self):
        """Test handling when action_names has fewer items than action_results."""
        mock_history = MagicMock()
        mock_history.action_names.return_value = ["navigate"]  # Only 1

        mock_result1 = MagicMock()
        mock_result1.extracted_content = "Step 1"
        mock_result1.long_term_memory = None
        mock_result1.error = None

        mock_result2 = MagicMock()
        mock_result2.extracted_content = "Step 2"
        mock_result2.long_term_memory = None
        mock_result2.error = None

        mock_history.action_results.return_value = [mock_result1, mock_result2]  # 2 results
        mock_history.is_done.return_value = True
        mock_history.is_successful.return_value = True
        mock_history.has_errors.return_value = False

        result = format_browser_history(mock_history)

        assert result["steps"][0]["action"] == "navigate"
        assert result["steps"][1]["action"] == "unknown"

    def test_format_handles_malformed_history(self):
        """Test graceful handling of malformed history objects."""
        mock_history = MagicMock()
        mock_history.action_names.side_effect = AttributeError("No such method")

        result = format_browser_history(mock_history)

        # Should not raise, should return fallback with format_error
        assert "format_error" in result
        assert result["steps"] == []
        assert result["completed"] is False
        assert result["success"] is False

    def test_format_handles_missing_methods(self):
        """Test handling history object missing some methods."""
        mock_history = MagicMock(spec=[])  # Empty spec = no methods

        result = format_browser_history(mock_history)

        # Should handle gracefully
        assert result["steps"] == []
        assert result["completed"] is False


@pytest.mark.unit
class TestJsonTypeToPython:
    """Test _json_type_to_python function."""

    def test_string_type(self):
        assert _json_type_to_python({"type": "string"}) == str

    def test_number_type(self):
        assert _json_type_to_python({"type": "number"}) == float

    def test_integer_type(self):
        assert _json_type_to_python({"type": "integer"}) == int

    def test_boolean_type(self):
        assert _json_type_to_python({"type": "boolean"}) == bool

    def test_object_type(self):
        result = _json_type_to_python({"type": "object"})
        assert result == Dict[str, Any]

    def test_array_of_strings(self):
        schema = {"type": "array", "items": {"type": "string"}}
        result = _json_type_to_python(schema)
        assert result == List[str]

    def test_array_of_numbers(self):
        schema = {"type": "array", "items": {"type": "number"}}
        result = _json_type_to_python(schema)
        assert result == List[float]

    def test_unknown_type_fallback(self):
        result = _json_type_to_python({"type": "unknown"})
        assert result == Any

    def test_no_type_fallback(self):
        result = _json_type_to_python({})
        assert result == Any


@pytest.mark.unit
class TestJsonSchemaToPydantic:
    """Test json_schema_to_pydantic function."""

    def test_basic_object_schema(self, sample_extraction_schema):
        model = json_schema_to_pydantic(sample_extraction_schema)

        assert issubclass(model, BaseModel)
        # Check field types
        fields = model.model_fields
        assert "name" in fields
        assert "price" in fields
        assert "in_stock" in fields

    def test_required_vs_optional(self, sample_extraction_schema):
        model = json_schema_to_pydantic(sample_extraction_schema)
        fields = model.model_fields

        # Required fields should not have default
        assert fields["name"].is_required()
        assert fields["price"].is_required()
        # Optional fields should have default
        assert not fields["in_stock"].is_required()

    def test_non_object_schema_wraps_in_data(self):
        schema = {"type": "string"}
        model = json_schema_to_pydantic(schema)

        fields = model.model_fields
        assert "data" in fields

    def test_array_field(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        model = json_schema_to_pydantic(schema)
        fields = model.model_fields
        assert "tags" in fields

    def test_empty_properties(self):
        schema = {"type": "object", "properties": {}}
        model = json_schema_to_pydantic(schema)
        assert issubclass(model, BaseModel)

    def test_custom_model_name(self):
        """Test that custom model name is used as prefix for generated model."""
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        model = json_schema_to_pydantic(schema, model_name="CustomModel")
        # Model name includes a counter suffix for uniqueness
        assert model.__name__.startswith("CustomModel")


@pytest.mark.asyncio
@pytest.mark.unit
class TestSingleton:
    """Test singleton behavior."""

    def test_instance_returns_same(self):
        service1 = BrowserService.instance()
        service2 = BrowserService.instance()
        assert service1 is service2

    def test_instance_creates_new_after_reset(self):
        service1 = BrowserService.instance()
        BrowserService._instance = None
        service2 = BrowserService.instance()
        assert service1 is not service2


@pytest.mark.asyncio
@pytest.mark.unit
class TestLoadStorageState:
    """Test _load_storage_state method."""

    async def test_load_with_profile_id(self, browser_service, mock_user):
        storage_state = {"cookies": [{"name": "test"}]}
        browser_service._session_context.load_session_state = AsyncMock(return_value=storage_state)
        profile_id = str(uuid4())

        result = await browser_service._load_storage_state(mock_user, profile_id)

        assert result == storage_state
        browser_service._session_context.load_session_state.assert_called_once()

    async def test_load_no_profile_id(self, browser_service, mock_user):
        result = await browser_service._load_storage_state(mock_user, None)
        assert result is None

    async def test_load_no_user(self, browser_service):
        result = await browser_service._load_storage_state(None, "profile-123")
        assert result is None

    async def test_load_handles_exception(self, browser_service, mock_user):
        browser_service._session_context.load_session_state = AsyncMock(
            side_effect=RuntimeError("Load failed")
        )
        profile_id = str(uuid4())

        result = await browser_service._load_storage_state(mock_user, profile_id)

        assert result is None


@pytest.mark.unit
class TestCreateOutputModel:
    """Test _create_output_model method."""

    def test_creates_pydantic_model(self, browser_service, sample_extraction_schema):
        model = browser_service._create_output_model(sample_extraction_schema)

        assert model is not None
        assert issubclass(model, BaseModel)

    def test_returns_none_for_no_schema(self, browser_service):
        result = browser_service._create_output_model(None)
        assert result is None

    def test_handles_invalid_schema(self, browser_service):
        # This won't actually fail due to how json_schema_to_pydantic handles edge cases
        # but we test that it doesn't raise
        result = browser_service._create_output_model({"invalid": True})
        # Will wrap in data field for non-object schemas
        assert result is not None


@pytest.mark.unit
class TestBuildErrorResult:
    """Test _build_error_result static method."""

    def test_error_result_structure(self):
        result = BrowserService._build_error_result("Something went wrong")

        assert result["success"] is False
        assert result["result"] == "Something went wrong"
        assert result["extracted_data"] == {}
        assert result["final_url"] is None
        assert result["urls"] == []
        assert result["duration_seconds"] is None
        assert result["steps_count"] is None
        assert result["extracted_content"] == []
        assert result["model_thoughts"] == []
        assert result["model_actions"] == []
        assert result["screenshots"] == []


@pytest.mark.unit
class TestEnhanceTask:
    """Test _enhance_task method."""

    def test_enhance_with_inputs(self, browser_service):
        task = "Navigate to the website"
        inputs = {"url": "https://example.com", "action": "click"}

        result = browser_service._enhance_task(task, inputs)

        assert "Navigate to the website" in result
        assert "Additional context:" in result
        assert "https://example.com" in result

    def test_enhance_empty_inputs(self, browser_service):
        task = "Navigate to the website"

        result = browser_service._enhance_task(task, {})

        assert result == task

    def test_enhance_none_inputs(self, browser_service):
        task = "Navigate to the website"

        result = browser_service._enhance_task(task, None)

        assert result == task


@pytest.mark.unit
class TestExtractUsageMetadata:
    """Test _extract_usage_metadata static method."""

    @patch("seer.services.browser.browser_service.config")
    def test_extracts_usage_with_default_model(self, mock_config, mock_agent_history):
        """Test usage extraction uses config default when model not specified."""
        mock_config.default_llm_model = "moonshotai/kimi-k2.5"

        result = BrowserService._extract_usage_metadata(mock_agent_history)

        assert result is not None
        assert result["model"] == "moonshotai/kimi-k2.5"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["steps_taken"] == 5

    @patch("seer.services.browser.browser_service.config")
    def test_extracts_usage_with_provided_model(self, mock_config, mock_agent_history):
        """Test usage extraction uses provided model when specified."""
        mock_config.default_llm_model = "moonshotai/kimi-k2.5"

        result = BrowserService._extract_usage_metadata(mock_agent_history, model="openai/gpt-4o")

        assert result is not None
        assert result["model"] == "openai/gpt-4o"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500

    @patch("seer.services.browser.browser_service.config")
    def test_extracts_usage_falls_back_to_config(self, mock_config, mock_agent_history):
        """Test usage extraction falls back to config when model is None."""
        mock_config.default_llm_model = "anthropic/claude-sonnet-4.5"

        result = BrowserService._extract_usage_metadata(mock_agent_history, model=None)

        assert result is not None
        assert result["model"] == "anthropic/claude-sonnet-4.5"

    def test_no_usage_attribute(self):
        history = MagicMock(spec=[])  # No usage attribute
        del history.usage

        result = BrowserService._extract_usage_metadata(history)

        assert result is None

    def test_usage_is_none(self):
        history = MagicMock()
        history.usage = None

        result = BrowserService._extract_usage_metadata(history)

        assert result is None

    def test_zero_tokens(self):
        history = MagicMock()
        history.usage = MagicMock()
        history.usage.total_tokens = 0

        result = BrowserService._extract_usage_metadata(history)

        assert result is None

    def test_handles_exception(self):
        history = MagicMock()
        history.usage = MagicMock()
        # Make total_tokens raise an exception
        type(history.usage).total_tokens = property(lambda self: (_ for _ in ()).throw(RuntimeError("Error")))

        result = BrowserService._extract_usage_metadata(history)

        assert result is None


@pytest.mark.unit
class TestExtractHistoryData:
    """Test new browser history extraction helper methods."""

    def test_extract_final_url_success(self):
        """Test extracting final URL from history."""
        history = MagicMock()
        history.urls.return_value = ["https://example.com", "https://example.com/page", "https://example.com/final"]

        result = BrowserService._extract_final_url(history)

        assert result == "https://example.com/final"

    def test_extract_final_url_empty(self):
        """Test final URL when no URLs visited."""
        history = MagicMock()
        history.urls.return_value = []

        result = BrowserService._extract_final_url(history)

        assert result is None

    def test_extract_final_url_no_method(self):
        """Test final URL when history has no urls method."""
        history = MagicMock(spec=[])

        result = BrowserService._extract_final_url(history)

        assert result is None

    def test_extract_urls_success(self):
        """Test extracting all URLs from history."""
        history = MagicMock()
        history.urls.return_value = ["https://a.com", "https://b.com"]

        result = BrowserService._extract_urls(history)

        assert result == ["https://a.com", "https://b.com"]

    def test_extract_urls_empty(self):
        """Test URLs when none visited."""
        history = MagicMock()
        history.urls.return_value = []

        result = BrowserService._extract_urls(history)

        assert result == []

    def test_extract_duration_success(self):
        """Test extracting duration from history."""
        history = MagicMock()
        history.total_duration_seconds.return_value = 45.5

        result = BrowserService._extract_duration(history)

        assert result == 45.5

    def test_extract_duration_no_method(self):
        """Test duration when method doesn't exist."""
        history = MagicMock(spec=[])

        result = BrowserService._extract_duration(history)

        assert result is None

    def test_extract_steps_count_success(self):
        """Test extracting step count from history."""
        history = MagicMock()
        history.number_of_steps.return_value = 12

        result = BrowserService._extract_steps_count(history)

        assert result == 12

    def test_extract_steps_count_no_method(self):
        """Test step count when method doesn't exist."""
        history = MagicMock(spec=[])

        result = BrowserService._extract_steps_count(history)

        assert result is None

    def test_extract_all_content_success(self):
        """Test extracting content from all steps."""
        history = MagicMock()
        history.extracted_content.return_value = ["Found title", None, "Clicked button", ""]

        result = BrowserService._extract_all_content(history)

        # Should filter out None and empty strings
        assert result == ["Found title", "Clicked button"]

    def test_extract_all_content_empty(self):
        """Test content when none extracted."""
        history = MagicMock()
        history.extracted_content.return_value = []

        result = BrowserService._extract_all_content(history)

        assert result == []

    def test_extract_model_thoughts_success(self):
        """Test extracting model thoughts (AgentBrain) from history."""
        history = MagicMock()

        mock_thought1 = MagicMock()
        mock_thought1.model_dump.return_value = {"evaluation_previous_goal": "Success", "next_action": "click"}

        mock_thought2 = MagicMock()
        mock_thought2.model_dump.return_value = {"evaluation_previous_goal": "Failed", "next_action": "retry"}

        history.model_thoughts.return_value = [mock_thought1, None, mock_thought2]

        result = BrowserService._extract_model_thoughts(history)

        assert len(result) == 2
        assert result[0]["evaluation_previous_goal"] == "Success"
        assert result[1]["evaluation_previous_goal"] == "Failed"

    def test_extract_model_thoughts_no_method(self):
        """Test thoughts when method doesn't exist."""
        history = MagicMock(spec=[])

        result = BrowserService._extract_model_thoughts(history)

        assert result == []

    def test_extract_model_actions_success(self):
        """Test extracting model actions from history."""
        history = MagicMock()

        mock_action1 = MagicMock()
        mock_action1.model_dump.return_value = {"action_type": "click", "element_index": 5}

        mock_action2 = MagicMock()
        mock_action2.model_dump.return_value = {"action_type": "input_text", "text": "hello"}

        history.model_actions.return_value = [mock_action1, None, mock_action2]

        result = BrowserService._extract_model_actions(history)

        assert len(result) == 2
        assert result[0]["action_type"] == "click"
        assert result[1]["action_type"] == "input_text"

    def test_extract_model_actions_no_method(self):
        """Test actions when method doesn't exist."""
        history = MagicMock(spec=[])

        result = BrowserService._extract_model_actions(history)

        assert result == []

    def test_extract_with_exception(self):
        """Test that exceptions are caught and return safe defaults."""
        history = MagicMock()
        history.urls.side_effect = RuntimeError("Unexpected error")
        history.total_duration_seconds.side_effect = RuntimeError("Unexpected error")

        # Should not raise, return safe defaults
        assert BrowserService._extract_final_url(history) is None
        assert BrowserService._extract_urls(history) == []
        assert BrowserService._extract_duration(history) is None


@pytest.mark.unit
class TestExtractData:
    """Test _extract_data method."""

    def test_extract_none(self, browser_service):
        result = browser_service._extract_data(None)
        assert result == {}

    def test_extract_dict(self, browser_service):
        data = {"key": "value"}
        result = browser_service._extract_data(data)
        assert result == data

    def test_extract_json_string(self, browser_service):
        json_str = '{"key": "value", "number": 42}'
        result = browser_service._extract_data(json_str)
        assert result == {"key": "value", "number": 42}

    def test_extract_invalid_json(self, browser_service):
        result = browser_service._extract_data("not valid json")
        assert result == {"raw": "not valid json"}

    def test_extract_other_type(self, browser_service):
        result = browser_service._extract_data(12345)
        assert result == {"raw": "12345"}


@pytest.mark.asyncio
@pytest.mark.unit
class TestExtractStructuredData:
    """Test _extract_structured_data method."""

    async def test_with_schema_parses_final_result(self, browser_service, sample_extraction_schema):
        history = MagicMock()
        history.final_result = MagicMock(return_value='{"name": "Product", "price": 19.99}')

        result = await browser_service._extract_structured_data(history, sample_extraction_schema)

        assert result == {"name": "Product", "price": 19.99}

    async def test_without_schema_uses_extract_data(self, browser_service):
        history = MagicMock()

        with patch.object(browser_service, "_extract_data", return_value={"raw": "data"}) as mock_extract:
            result = await browser_service._extract_structured_data(history, None)

        mock_extract.assert_called_once_with(history)
        assert result == {"raw": "data"}

    async def test_handles_json_decode_error(self, browser_service, sample_extraction_schema):
        """Test that JSON decode error returns None to signal extraction failure.

        This is the fix for Sentry issue PYTHON-FASTAPI-V and PYTHON-FASTAPI-W.
        When extraction fails, we return None instead of {} to prevent downstream
        validation errors on empty extracted_data.
        """
        history = MagicMock()
        history.final_result = MagicMock(return_value="not valid json")

        result = await browser_service._extract_structured_data(history, sample_extraction_schema)

        # Returns None to signal extraction failure when schema was provided
        assert result is None

    async def test_handles_no_final_result(self, browser_service, sample_extraction_schema):
        """Test that missing final_result returns None to signal extraction failure."""
        history = MagicMock()
        history.final_result = MagicMock(return_value=None)

        result = await browser_service._extract_structured_data(history, sample_extraction_schema)

        # Returns None to signal extraction failure when schema was provided
        assert result is None

    async def test_strips_markdown_fencing(self, browser_service, sample_extraction_schema):
        """Test that markdown-fenced JSON is correctly parsed.

        This is the fix for Sentry issue PYTHON-FASTAPI-V where LLM returns
        JSON wrapped in ```json ... ``` markers.
        """
        history = MagicMock()
        history.final_result = MagicMock(return_value='```json\n{"name": "Widget", "price": 29.99}\n```')

        result = await browser_service._extract_structured_data(history, sample_extraction_schema)

        assert result == {"name": "Widget", "price": 29.99}

    async def test_final_result_is_dict(self, browser_service, sample_extraction_schema):
        history = MagicMock()
        history.final_result = MagicMock(return_value={"name": "Test", "price": 10})

        result = await browser_service._extract_structured_data(history, sample_extraction_schema)

        assert result == {"name": "Test", "price": 10}


@pytest.mark.asyncio
@pytest.mark.unit
class TestSaveScreenshots:
    """Test _save_screenshots method."""

    async def test_save_screenshots_disabled(self, browser_service, mock_agent_history):
        result = await browser_service._save_screenshots(
            history=mock_agent_history,
            save_screenshots=False,
            file_system=MagicMock(),
            workflow_run_id="run-123",
            user=MagicMock(),
        )

        assert result == []

    async def test_save_screenshots_missing_context(self, browser_service, mock_agent_history, mock_user):
        # Missing file_system
        result = await browser_service._save_screenshots(
            history=mock_agent_history,
            save_screenshots=True,
            file_system=None,
            workflow_run_id="run-123",
            user=mock_user,
        )
        assert result == []

        # Missing run_id
        result = await browser_service._save_screenshots(
            history=mock_agent_history,
            save_screenshots=True,
            file_system=MagicMock(),
            workflow_run_id=None,
            user=mock_user,
        )
        assert result == []

        # Missing user
        result = await browser_service._save_screenshots(
            history=mock_agent_history,
            save_screenshots=True,
            file_system=MagicMock(),
            workflow_run_id="run-123",
            user=None,
        )
        assert result == []

    async def test_save_screenshots_success(self, browser_service, mock_user, mock_file_system):
        # Create mock history with screenshots
        history = MagicMock()
        # Base64 encoded PNG data
        screenshot_b64 = base64.b64encode(b"fake_png_data").decode()
        history.screenshots = MagicMock(return_value=[screenshot_b64])

        mock_file_ref = MagicMock()
        mock_file_ref.to_dict = MagicMock(return_value={"url": "s3://bucket/screenshot.png"})
        mock_file_system.store_file_with_record = AsyncMock(return_value=mock_file_ref)

        result = await browser_service._save_screenshots(
            history=history,
            save_screenshots=True,
            file_system=mock_file_system,
            workflow_run_id="run-123",
            user=mock_user,
        )

        assert len(result) == 1
        assert result[0]["url"] == "s3://bucket/screenshot.png"
        mock_file_system.store_file_with_record.assert_called_once()

    async def test_save_screenshots_handles_errors(self, browser_service, mock_user, mock_file_system):
        history = MagicMock()
        screenshot_b64 = base64.b64encode(b"fake_png_data").decode()
        history.screenshots = MagicMock(return_value=[screenshot_b64])

        mock_file_system.store_file_with_record = AsyncMock(side_effect=RuntimeError("S3 error"))

        # Should not raise, just log warning and return empty
        result = await browser_service._save_screenshots(
            history=history,
            save_screenshots=True,
            file_system=mock_file_system,
            workflow_run_id="run-123",
            user=mock_user,
        )

        assert result == []

    async def test_save_screenshots_no_screenshots(self, browser_service, mock_user, mock_file_system):
        history = MagicMock()
        history.screenshots = MagicMock(return_value=[])

        result = await browser_service._save_screenshots(
            history=history,
            save_screenshots=True,
            file_system=mock_file_system,
            workflow_run_id="run-123",
            user=mock_user,
        )

        assert result == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestExecuteTask:
    """Test execute_task method."""

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_success(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, browser_service, mock_user, mock_agent_history
    ):
        mock_config.openrouter_api_key = "test-api-key"

        # Mock pool
        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)
        browser_service._session_context.save_session_state = AsyncMock()

        result = await browser_service.execute_task(
            user=mock_user,
            task="Click the login button",
            inputs={"url": "https://example.com"},
        )

        assert result["success"] is True
        assert "usage" in result

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_creates_pool_session(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, browser_service, mock_user, mock_agent_history
    ):
        mock_config.openrouter_api_key = "test-api-key"

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        await browser_service.execute_task(
            user=mock_user,
            task="Click the login button",
            inputs={},
            timeout_seconds=120,
        )

        mock_pool.create_session.assert_called_once()
        call_kwargs = mock_pool.create_session.call_args.kwargs
        assert call_kwargs["user_id"] == str(mock_user.user_id)
        assert call_kwargs["timeout"] == 120

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_loads_storage_state(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, browser_service, mock_user, mock_agent_history
    ):
        mock_config.openrouter_api_key = "test-api-key"
        profile_id = str(uuid4())

        storage_state = {"cookies": [{"name": "auth"}]}
        browser_service._session_context.load_session_state = AsyncMock(return_value=storage_state)

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
            browser_profile_id=profile_id,
        )

        # Verify pool was called with storage_state
        call_kwargs = mock_pool.create_session.call_args.kwargs
        assert call_kwargs["storage_state"] == storage_state

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_releases_session(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, browser_service, mock_user, mock_agent_history
    ):
        mock_config.openrouter_api_key = "test-api-key"

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
        )

        mock_pool.release_session.assert_called_once_with("sess-123")

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_saves_session_state(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, browser_service, mock_user, mock_agent_history
    ):
        mock_config.openrouter_api_key = "test-api-key"
        profile_id = str(uuid4())

        final_state = {"cookies": [{"name": "new_cookie"}]}

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=final_state)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)
        browser_service._session_context.save_session_state = AsyncMock()

        await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
            browser_profile_id=profile_id,
        )

        browser_service._session_context.save_session_state.assert_called_once()

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_timeout_error(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, browser_service, mock_user
    ):
        mock_config.openrouter_api_key = "test-api-key"

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        result = await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
            timeout_seconds=60,
        )

        assert result["success"] is False
        assert "timed out" in result["result"]
        assert result["usage"] is None

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_general_error(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, browser_service, mock_user
    ):
        mock_config.openrouter_api_key = "test-api-key"

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Browser crashed"))
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        result = await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
        )

        assert result["success"] is False
        assert "Browser crashed" in result["result"]
        assert result["usage"] is None

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_extraction_failure_with_schema(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls,
        browser_service, mock_user, sample_extraction_schema
    ):
        """Test that execute_task returns failure when extraction fails with schema.

        This is the fix for Sentry issues PYTHON-FASTAPI-V and PYTHON-FASTAPI-W.
        When a schema is provided but extraction fails (e.g., invalid JSON from LLM),
        we return success=False to prevent downstream validation errors.
        """
        mock_config.openrouter_api_key = "test-api-key"

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        # Mock agent that returns history with invalid JSON
        mock_history = MagicMock()
        mock_history.final_result = MagicMock(return_value="completely invalid json that can't be parsed")
        mock_history.usage = MagicMock()
        mock_history.usage.total_tokens = 100
        mock_history.usage.total_prompt_tokens = 80
        mock_history.usage.total_completion_tokens = 20
        mock_history.usage.entry_count = 3

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        result = await browser_service.execute_task(
            user=mock_user,
            task="Extract shop data",
            inputs={},
            extraction_schema=sample_extraction_schema,  # Schema provided
        )

        # Should return failure because extraction failed
        assert result["success"] is False
        assert "Failed to extract structured data" in result["result"]
        # Usage should still be tracked even on extraction failure
        assert result["usage"] is not None

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_execute_task_anonymous_user(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, browser_service, mock_agent_history
    ):
        mock_config.openrouter_api_key = "test-api-key"

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        result = await browser_service.execute_task(
            user=None,  # Anonymous
            task="Click button",
            inputs={},
        )

        assert result["success"] is True
        # Verify pool was called with "anonymous"
        call_kwargs = mock_pool.create_session.call_args.kwargs
        assert call_kwargs["user_id"] == "anonymous"


@pytest.mark.unit
class TestGetAgentLlm:
    """Test _get_agent_llm method."""

    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    def test_creates_openrouter_llm_with_default_model(self, mock_config, mock_openai_cls, browser_service):
        """Test LLM creation uses config default when no model specified."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.default_llm_model = "moonshotai/kimi-k2.5"

        browser_service._get_agent_llm()

        mock_openai_cls.assert_called_once()
        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["api_key"] == "test-api-key"
        assert "openrouter" in call_kwargs["base_url"]
        assert call_kwargs["model"] == "moonshotai/kimi-k2.5"

    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    def test_uses_provided_model(self, mock_config, mock_openai_cls, browser_service):
        """Test LLM creation uses provided model when specified."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.default_llm_model = "moonshotai/kimi-k2.5"

        browser_service._get_agent_llm(model="openai/gpt-4o")

        mock_openai_cls.assert_called_once()
        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"

    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    def test_falls_back_to_config_default_when_model_none(self, mock_config, mock_openai_cls, browser_service):
        """Test LLM creation falls back to config default when model is None."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.default_llm_model = "anthropic/claude-sonnet-4.5"

        browser_service._get_agent_llm(model=None)

        mock_openai_cls.assert_called_once()
        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4.5"

    @patch("seer.services.browser.browser_service.config")
    def test_raises_on_missing_api_key(self, mock_config, browser_service):
        mock_config.openrouter_api_key = None

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            browser_service._get_agent_llm()

    @patch("seer.services.browser.browser_service.config")
    def test_raises_on_empty_api_key(self, mock_config, browser_service):
        mock_config.openrouter_api_key = ""

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            browser_service._get_agent_llm()


@pytest.mark.asyncio
@pytest.mark.unit
class TestWorkflowRecording:
    """Test workflow session recording integration."""

    @patch("seer.services.browser.browser_service.RecordingService")
    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_starts_recording_when_enabled(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, mock_rec_cls,
        browser_service, mock_user, mock_agent_history
    ):
        """Test that recording is started when config.browser_recording_enabled=True."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.browser_recording_enabled = True

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.recording_id = None
        mock_managed.start_url = None

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.start_recording = AsyncMock(return_value="rec-456")
        mock_recorder.save_recording = AsyncMock(return_value="rec-456")
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={"url": "https://example.com"},
        )

        # Verify start_recording was called
        mock_recorder.start_recording.assert_called_once()
        call_args = mock_recorder.start_recording.call_args
        assert call_args[0][0] == "sess-123"  # session_id
        assert call_args[1]["start_url"] == "https://example.com"

    @patch("seer.services.browser.browser_service.RecordingService")
    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_saves_recording_with_workflow_run_id(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, mock_rec_cls,
        browser_service, mock_user, mock_agent_history
    ):
        """Test that recording is saved with correct workflow_run_id and session_type."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.browser_recording_enabled = True

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.recording_id = "rec-456"
        mock_managed.start_url = "https://example.com"

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.start_recording = AsyncMock(return_value="rec-456")
        mock_recorder.save_recording = AsyncMock(return_value="rec-456")
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
            workflow_run_id="run_789",
        )

        # Verify save_recording was called with correct params
        mock_recorder.save_recording.assert_called_once()
        call_kwargs = mock_recorder.save_recording.call_args.kwargs
        assert call_kwargs["workflow_run_id"] == "run_789"
        assert call_kwargs["session_type"] == "workflow"

    @patch("seer.services.browser.browser_service.RecordingService")
    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_recording_skipped_when_disabled(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, mock_rec_cls,
        browser_service, mock_user, mock_agent_history
    ):
        """Test that recording is not started when config.browser_recording_enabled=False."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.recording_id = None

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
        )

        # Verify RecordingService was never accessed
        mock_rec_cls.get_instance.assert_not_called()

    @patch("seer.services.browser.browser_service.RecordingService")
    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_recording_failure_does_not_fail_workflow(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, mock_rec_cls,
        browser_service, mock_user, mock_agent_history
    ):
        """Test that recording failures don't cause workflow failure."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.browser_recording_enabled = True

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.recording_id = None
        mock_managed.start_url = None

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.start_recording = AsyncMock(side_effect=RuntimeError("Recording failed"))
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        # Should not raise despite recording failure
        result = await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
        )

        assert result["success"] is True

    @patch("seer.services.browser.browser_service.RecordingService")
    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_recording_skipped_when_no_user(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, mock_rec_cls,
        browser_service, mock_agent_history
    ):
        """Test that recording save is skipped when user is None (anonymous)."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.browser_recording_enabled = True

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.recording_id = "rec-456"  # Recording was started
        mock_managed.start_url = "https://example.com"

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.start_recording = AsyncMock(return_value="rec-456")
        mock_recorder.save_recording = AsyncMock()
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        await browser_service.execute_task(
            user=None,  # Anonymous user
            task="Click button",
            inputs={},
        )

        # save_recording should NOT be called when user is None
        mock_recorder.save_recording.assert_not_called()

    @patch("seer.services.browser.browser_service.RecordingService")
    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_extracts_start_url_from_inputs(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls, mock_rec_cls,
        browser_service, mock_user, mock_agent_history
    ):
        """Test that start_url is extracted from inputs['url'] or inputs['start_url']."""
        mock_config.openrouter_api_key = "test-api-key"
        mock_config.browser_recording_enabled = True

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()
        mock_managed.recording_id = None
        mock_managed.start_url = None

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_recorder = MagicMock()
        mock_recorder.start_recording = AsyncMock(return_value="rec-456")
        mock_recorder.save_recording = AsyncMock()
        mock_rec_cls.get_instance = AsyncMock(return_value=mock_recorder)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        # Test with 'start_url' key
        await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={"start_url": "https://start-url-example.com"},
        )

        call_kwargs = mock_recorder.start_recording.call_args[1]
        assert call_kwargs["start_url"] == "https://start-url-example.com"


@pytest.mark.unit
class TestUsesCustomSubmitTool:
    """Test _uses_custom_submit_tool model detection method."""

    def test_detects_kimi_model(self):
        """Kimi models should use custom submit_result tool."""
        assert BrowserService._uses_custom_submit_tool("moonshotai/kimi-k2.5") is True
        assert BrowserService._uses_custom_submit_tool("kimi-k2.5") is True
        assert BrowserService._uses_custom_submit_tool("KIMI-K2.5") is True  # Case insensitive

    def test_detects_moonshot_model(self):
        """Moonshot models should use custom submit_result tool."""
        assert BrowserService._uses_custom_submit_tool("moonshot-v1") is True
        assert BrowserService._uses_custom_submit_tool("MOONSHOT-V1") is True

    def test_openai_models_use_standard(self):
        """OpenAI models should not use custom tool."""
        assert BrowserService._uses_custom_submit_tool("gpt-4") is False
        assert BrowserService._uses_custom_submit_tool("gpt-4-turbo") is False
        assert BrowserService._uses_custom_submit_tool("gpt-3.5-turbo") is False

    def test_anthropic_models_use_standard(self):
        """Anthropic models should not use custom tool."""
        assert BrowserService._uses_custom_submit_tool("claude-3-opus") is False
        assert BrowserService._uses_custom_submit_tool("claude-3-sonnet") is False
        assert BrowserService._uses_custom_submit_tool("claude-2") is False

    def test_empty_model_name(self):
        """Empty model name should not use custom tool."""
        assert BrowserService._uses_custom_submit_tool("") is False


@pytest.mark.unit
class TestEnhanceTaskWithSchema:
    """Test _enhance_task method with extraction schema parameters."""

    def test_enhance_with_schema_submit_tool(self, browser_service):
        """Test enhancement with extraction schema and use_submit_tool=True."""
        task = "Extract product data"
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
        }

        result = browser_service._enhance_task(task, {}, schema, use_submit_tool=True)

        assert "Extract product data" in result
        assert "submit_result" in result
        assert "name" in result
        assert "price" in result
        assert "MUST call" in result

    def test_enhance_with_schema_standard(self, browser_service):
        """Test enhancement with extraction schema and use_submit_tool=False."""
        task = "Extract product data"
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }

        result = browser_service._enhance_task(task, {}, schema, use_submit_tool=False)

        assert "Extract product data" in result
        assert "name" in result
        assert "submit_result" not in result  # Should NOT mention submit_result

    def test_enhance_with_inputs_and_schema(self, browser_service):
        """Test enhancement with both inputs and extraction schema."""
        task = "Navigate and extract"
        inputs = {"url": "https://example.com"}
        schema = {"type": "object", "properties": {"data": {"type": "string"}}}

        result = browser_service._enhance_task(task, inputs, schema, use_submit_tool=True)

        assert "Navigate and extract" in result
        assert "https://example.com" in result
        assert "submit_result" in result
        assert "data" in result

    def test_enhance_without_schema_unchanged(self, browser_service):
        """Test that task without schema doesn't get extraction instructions."""
        task = "Click the button"
        inputs = {"url": "https://example.com"}

        result = browser_service._enhance_task(task, inputs, None, use_submit_tool=False)

        assert "Click the button" in result
        assert "https://example.com" in result
        assert "submit_result" not in result
        assert "IMPORTANT" not in result


@pytest.mark.asyncio
@pytest.mark.unit
class TestModelAwareExecution:
    """Test execute_task with model-aware structured output."""

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_kimi_uses_custom_tools_for_extraction(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls,
        browser_service, mock_user, sample_extraction_schema
    ):
        """Test that Kimi model uses CustomBrowserTools when extraction_schema is provided."""
        mock_config.openrouter_api_key = "test-api-key"

        # Mock LLM to return kimi model
        mock_llm = MagicMock()
        mock_llm.model = "moonshotai/kimi-k2.5"
        mock_openai_cls.return_value = mock_llm

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        # Mock agent with history that has usage
        mock_history = MagicMock()
        mock_history.usage = MagicMock()
        mock_history.usage.total_tokens = 100
        mock_history.usage.total_prompt_tokens = 80
        mock_history.usage.total_completion_tokens = 20
        mock_history.usage.entry_count = 3
        mock_history.screenshots = MagicMock(return_value=[])

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        # Execute with extraction schema
        result = await browser_service.execute_task(
            user=mock_user,
            task="Extract data",
            inputs={},
            extraction_schema=sample_extraction_schema,
        )

        # Verify Agent was called with tools parameter (CustomBrowserTools)
        agent_call = mock_agent_cls.call_args
        assert agent_call.kwargs.get("tools") is not None
        # And output_model_schema should be None for custom tool approach
        assert agent_call.kwargs.get("output_model_schema") is None

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_openai_uses_standard_output_model(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls,
        browser_service, mock_user, sample_extraction_schema, mock_agent_history
    ):
        """Test that non-Kimi model uses standard output_model_schema."""
        mock_config.openrouter_api_key = "test-api-key"

        # Mock LLM to return OpenAI model
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_openai_cls.return_value = mock_llm

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        # Execute with extraction schema
        result = await browser_service.execute_task(
            user=mock_user,
            task="Extract data",
            inputs={},
            extraction_schema=sample_extraction_schema,
        )

        # Verify Agent was called with output_model_schema, not tools
        agent_call = mock_agent_cls.call_args
        assert agent_call.kwargs.get("output_model_schema") is not None
        # tools should not be set (or None) for standard approach
        assert agent_call.kwargs.get("tools") is None

    @patch("seer.services.browser.browser_service.BrowserPoolManager")
    @patch("seer.services.browser.browser_service.Agent")
    @patch("seer.services.browser.browser_service.ChatOpenAI")
    @patch("seer.services.browser.browser_service.config")
    async def test_kimi_without_schema_no_custom_tools(
        self, mock_config, mock_openai_cls, mock_agent_cls, mock_pool_cls,
        browser_service, mock_user, mock_agent_history
    ):
        """Test that Kimi without extraction_schema doesn't use CustomBrowserTools."""
        mock_config.openrouter_api_key = "test-api-key"

        mock_llm = MagicMock()
        mock_llm.model = "moonshotai/kimi-k2.5"
        mock_openai_cls.return_value = mock_llm

        mock_managed = MagicMock()
        mock_managed.id = "sess-123"
        mock_managed.session = MagicMock()

        mock_pool = MagicMock()
        mock_pool.create_session = AsyncMock(return_value=mock_managed)
        mock_pool.release_session = AsyncMock(return_value=None)
        mock_pool_cls.get_instance = AsyncMock(return_value=mock_pool)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_history)
        mock_agent_cls.return_value = mock_agent

        browser_service._session_context.load_session_state = AsyncMock(return_value=None)

        # Execute WITHOUT extraction schema
        result = await browser_service.execute_task(
            user=mock_user,
            task="Click button",
            inputs={},
            extraction_schema=None,  # No schema
        )

        # Verify Agent was called without tools (no custom tools needed)
        agent_call = mock_agent_cls.call_args
        assert agent_call.kwargs.get("tools") is None
        assert agent_call.kwargs.get("output_model_schema") is None


# ==============================================================================
# TARGET DETACHMENT RECOVERY TESTS
# ==============================================================================
#
# Tests for CDP target detachment handling - a key durability feature that
# prevents total task failure when browser tabs disconnect temporarily.

@pytest.mark.asyncio
@pytest.mark.unit
class TestCategorizeError:
    """Test error categorization for metrics and handling."""

    def test_categorizes_target_detachment(self, browser_service):
        """Test that 'no valid agent focus' errors are categorized correctly."""
        error = ValueError("No valid agent focus available - target may have detached")
        assert browser_service._categorize_error(error) == "target_detachment"

    def test_categorizes_cdp_error(self, browser_service):
        """Test that CDP-related errors are categorized."""
        error = RuntimeError("CDP connection lost to Chrome instance")
        assert browser_service._categorize_error(error) == "cdp_error"

    def test_categorizes_timeout(self, browser_service):
        """Test that timeout errors are categorized."""
        error = asyncio.TimeoutError("Operation timed out after 30s")
        # TimeoutError doesn't have a message, test with generic Exception
        error_with_msg = Exception("timeout waiting for response")
        assert browser_service._categorize_error(error_with_msg) == "timeout"

    def test_categorizes_connection_error(self, browser_service):
        """Test that connection errors are categorized."""
        error = ConnectionError("Failed to establish connection to browser")
        assert browser_service._categorize_error(error) == "connection_error"

    def test_categorizes_unknown_error(self, browser_service):
        """Test that unrecognized errors are categorized as unknown."""
        error = ValueError("Something unexpected happened")
        assert browser_service._categorize_error(error) == "unknown"


@pytest.mark.asyncio
@pytest.mark.unit
class TestTryRecoverFocus:
    """Test browser focus recovery mechanism."""

    async def test_recover_focus_succeeds_when_page_accessible(self, browser_service):
        """Test that recovery succeeds when browser page is accessible."""
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_session.must_get_current_page = AsyncMock(return_value=mock_page)

        result = await browser_service._try_recover_focus(mock_session)

        assert result is True
        mock_session.must_get_current_page.assert_called_once()

    async def test_recover_focus_fails_on_exception(self, browser_service):
        """Test that recovery returns False when page access fails."""
        mock_session = MagicMock()
        mock_session.must_get_current_page = AsyncMock(
            side_effect=RuntimeError("Target closed")
        )

        result = await browser_service._try_recover_focus(mock_session)

        assert result is False

    async def test_recover_focus_fails_on_timeout(self, browser_service):
        """Test that recovery returns False when page access times out."""
        mock_session = MagicMock()

        async def slow_get_page():
            await asyncio.sleep(10)  # Longer than timeout
            return MagicMock()

        mock_session.must_get_current_page = slow_get_page

        result = await browser_service._try_recover_focus(mock_session)

        assert result is False


@pytest.mark.asyncio
@pytest.mark.unit
class TestRunBrowserAgentWithRecovery:
    """Test retry logic for transient target detachment."""

    async def test_succeeds_on_first_attempt(self, browser_service):
        """Test that successful execution returns immediately."""
        mock_history = MagicMock()
        mock_session = MagicMock()

        with patch.object(browser_service, "_run_browser_agent") as mock_run:
            mock_run.return_value = (mock_history, None)

            result = await browser_service._run_browser_agent_with_recovery(
                task="Click button",
                inputs={},
                extraction_schema=None,
                max_steps=10,
                timeout_seconds=60,
                browser_session=mock_session,
            )

            assert result == (mock_history, None)
            assert mock_run.call_count == 1

    async def test_retries_on_focus_error(self, browser_service):
        """Test that focus errors trigger retry with recovery."""
        from seer.services.browser.exceptions import TargetDetachmentError

        mock_history = MagicMock()
        mock_session = MagicMock()
        mock_managed = MagicMock()
        mock_managed.id = "test-session"

        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("No valid agent focus available")
            return (mock_history, None)

        with patch.object(browser_service, "_run_browser_agent", side_effect=mock_run):
            with patch.object(browser_service, "_try_recover_focus", return_value=True):
                result = await browser_service._run_browser_agent_with_recovery(
                    task="Click button",
                    inputs={},
                    extraction_schema=None,
                    max_steps=10,
                    timeout_seconds=60,
                    browser_session=mock_session,
                    managed_session=mock_managed,
                )

                assert result == (mock_history, None)
                assert call_count == 2

    async def test_raises_after_max_attempts(self, browser_service):
        """Test that TargetDetachmentError is raised after max retry attempts."""
        from seer.services.browser.exceptions import TargetDetachmentError

        mock_session = MagicMock()
        mock_managed = MagicMock()
        mock_managed.id = "test-session"

        with patch.object(browser_service, "_run_browser_agent") as mock_run:
            mock_run.side_effect = ValueError("No valid agent focus available")
            with patch.object(browser_service, "_try_recover_focus", return_value=True):
                with pytest.raises(TargetDetachmentError) as exc_info:
                    await browser_service._run_browser_agent_with_recovery(
                        task="Click button",
                        inputs={},
                        extraction_schema=None,
                        max_steps=10,
                        timeout_seconds=60,
                        browser_session=mock_session,
                        managed_session=mock_managed,
                    )

                assert "test-session" in str(exc_info.value)
                assert mock_run.call_count == 3  # MAX_ATTEMPTS

    async def test_raises_immediately_on_recovery_failure(self, browser_service):
        """Test that TargetDetachmentError is raised if recovery fails."""
        from seer.services.browser.exceptions import TargetDetachmentError

        mock_session = MagicMock()
        mock_managed = MagicMock()
        mock_managed.id = "test-session"

        with patch.object(browser_service, "_run_browser_agent") as mock_run:
            mock_run.side_effect = ValueError("No valid agent focus available")
            with patch.object(browser_service, "_try_recover_focus", return_value=False):
                with pytest.raises(TargetDetachmentError):
                    await browser_service._run_browser_agent_with_recovery(
                        task="Click button",
                        inputs={},
                        extraction_schema=None,
                        max_steps=10,
                        timeout_seconds=60,
                        browser_session=mock_session,
                        managed_session=mock_managed,
                    )

                # Should fail after first attempt since recovery failed
                assert mock_run.call_count == 1

    async def test_reraises_non_focus_errors(self, browser_service):
        """Test that non-focus errors are re-raised immediately."""
        mock_session = MagicMock()

        with patch.object(browser_service, "_run_browser_agent") as mock_run:
            mock_run.side_effect = ValueError("Some other error")

            with pytest.raises(ValueError) as exc_info:
                await browser_service._run_browser_agent_with_recovery(
                    task="Click button",
                    inputs={},
                    extraction_schema=None,
                    max_steps=10,
                    timeout_seconds=60,
                    browser_session=mock_session,
                )

            assert "Some other error" in str(exc_info.value)
            assert mock_run.call_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
class TestEmergencySaveRecording:
    """Test emergency recording save on session failure."""

    @patch("seer.services.browser.browser_service.RecordingService")
    @patch("seer.services.browser.browser_service.config")
    async def test_saves_recording_on_failure(
        self, mock_config, mock_recording_service_cls, browser_service, mock_user
    ):
        """Test that emergency save attempts to save recording data."""
        mock_config.browser_recording_enabled = True

        mock_recorder = MagicMock()
        mock_recorder.save_recording = AsyncMock(return_value="recording-123")
        mock_recording_service_cls.get_instance = AsyncMock(return_value=mock_recorder)

        mock_managed = MagicMock()
        mock_managed.id = "session-123"
        mock_managed.recording_id = "rec-123"
        mock_managed.start_url = "https://example.com"

        result = await browser_service._emergency_save_recording(
            mock_managed, mock_user, "profile-123", "run-456"
        )

        assert result is True
        mock_recorder.save_recording.assert_called_once_with(
            "session-123",
            mock_user,
            profile_id="profile-123",
            workflow_run_id="run-456",
            session_type="workflow",
            start_url="https://example.com",
        )

    @patch("seer.services.browser.browser_service.config")
    async def test_returns_false_when_no_recording_id(
        self, mock_config, browser_service, mock_user
    ):
        """Test that emergency save returns False when no recording was started."""
        mock_config.browser_recording_enabled = True

        mock_managed = MagicMock()
        mock_managed.id = "session-123"
        mock_managed.recording_id = None  # No recording

        result = await browser_service._emergency_save_recording(
            mock_managed, mock_user, "profile-123", "run-456"
        )

        assert result is False

    @patch("seer.services.browser.browser_service.config")
    async def test_returns_false_when_recording_disabled(
        self, mock_config, browser_service, mock_user
    ):
        """Test that emergency save returns False when recording is disabled."""
        mock_config.browser_recording_enabled = False

        mock_managed = MagicMock()
        mock_managed.id = "session-123"
        mock_managed.recording_id = "rec-123"

        result = await browser_service._emergency_save_recording(
            mock_managed, mock_user, "profile-123", "run-456"
        )

        assert result is False

    @patch("seer.services.browser.browser_service.RecordingService")
    @patch("seer.services.browser.browser_service.config")
    async def test_handles_save_error_gracefully(
        self, mock_config, mock_recording_service_cls, browser_service, mock_user
    ):
        """Test that emergency save handles errors without raising."""
        mock_config.browser_recording_enabled = True

        mock_recorder = MagicMock()
        mock_recorder.save_recording = AsyncMock(
            side_effect=RuntimeError("Database error")
        )
        mock_recording_service_cls.get_instance = AsyncMock(return_value=mock_recorder)

        mock_managed = MagicMock()
        mock_managed.id = "session-123"
        mock_managed.recording_id = "rec-123"
        mock_managed.start_url = "https://example.com"

        # Should not raise
        result = await browser_service._emergency_save_recording(
            mock_managed, mock_user, "profile-123", "run-456"
        )

        assert result is False
