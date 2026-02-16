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
    return history


@pytest.fixture
def mock_file_system():
    """Mock WorkflowFileSystem for screenshot saving."""
    fs = MagicMock()
    fs.store_file_with_record = AsyncMock()
    return fs


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
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        model = json_schema_to_pydantic(schema, model_name="CustomModel")
        assert model.__name__ == "CustomModel"


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

    def test_extracts_usage_from_history(self, mock_agent_history):
        result = BrowserService._extract_usage_metadata(mock_agent_history)

        assert result is not None
        assert result["model"] == "moonshotai/kimi-k2.5"
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["steps_taken"] == 5

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
        history = MagicMock()
        history.final_result = MagicMock(return_value="not valid json")

        result = await browser_service._extract_structured_data(history, sample_extraction_schema)

        # Falls back to _extract_data
        assert "raw" in result or result == {}

    async def test_handles_no_final_result(self, browser_service, sample_extraction_schema):
        history = MagicMock()
        history.final_result = MagicMock(return_value=None)

        result = await browser_service._extract_structured_data(history, sample_extraction_schema)

        assert result == {}

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
    def test_creates_openrouter_llm(self, mock_config, mock_openai_cls, browser_service):
        mock_config.openrouter_api_key = "test-api-key"

        browser_service._get_agent_llm()

        mock_openai_cls.assert_called_once()
        call_kwargs = mock_openai_cls.call_args.kwargs
        assert call_kwargs["api_key"] == "test-api-key"
        assert "openrouter" in call_kwargs["base_url"]
        assert call_kwargs["model"] == "moonshotai/kimi-k2.5"

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
