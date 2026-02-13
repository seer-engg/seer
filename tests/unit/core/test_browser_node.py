"""
Unit tests for BrowserNode implementation.

Tests schema validation, type environment registration, and workflow spec parsing.
"""

import pytest
from pydantic import BaseModel, ValidationError

from seer.core.compiler.type_env import build_type_environment
from seer.core.nodes.base import TypeRegistrationContext
from seer.core.nodes.registry import node_type_registry
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    BrowserNode,
    Edge,
    LLMNode,
    OutputContract,
    OutputMode,
    WorkflowSpec,
)
from seer.core.schema.schema_registry import SchemaRegistry
from seer.services.browser.browser_service import json_schema_to_pydantic, _json_type_to_python


# =============================================================================
# Schema Validation Tests
# =============================================================================


def test_browser_node_basic_valid():
    """Test basic valid BrowserNode creation."""
    node = BrowserNode(
        id="browse",
        task="Go to example.com and extract the page title",
    )

    assert node.id == "browse"
    assert node.type == "browser"
    assert node.task == "Go to example.com and extract the page title"
    assert node.browser_profile_id is None
    assert node.max_steps == 25
    assert node.timeout_seconds == 300


def test_browser_node_with_profile():
    """Test BrowserNode with profile ID configured."""
    node = BrowserNode(
        id="authenticated_browse",
        task="Go to Slack and get messages from #general",
        browser_profile_id="550e8400-e29b-41d4-a716-446655440000",
        max_steps=50,
        timeout_seconds=600,
    )

    assert node.browser_profile_id == "550e8400-e29b-41d4-a716-446655440000"
    assert node.max_steps == 50
    assert node.timeout_seconds == 600


def test_browser_node_with_inputs():
    """Test BrowserNode with input expressions."""
    node = BrowserNode(
        id="dynamic_browse",
        task="Search for ${trigger.query}",
        inputs={
            "query": "${trigger.data.search_term}",
            "max_results": 10,
        },
    )

    assert "${trigger.query}" in node.task
    assert len(node.inputs) == 2
    assert node.inputs["max_results"] == 10


def test_browser_node_with_expect_outputs():
    """Test BrowserNode with expected output schema."""
    node = BrowserNode(
        id="structured_browse",
        task="Extract product details",
        expect_outputs=OutputContract(
            mode=OutputMode.json,
            schema={"id": "product_schema"},
        ),
    )

    assert node.expect_outputs is not None
    assert node.expect_outputs.mode == OutputMode.json


def test_browser_node_save_screenshots_default():
    """Test that save_screenshots defaults to False."""
    node = BrowserNode(
        id="browse",
        task="Take screenshots of the page",
    )

    assert node.save_screenshots is False


def test_browser_node_with_save_screenshots_enabled():
    """Test BrowserNode with save_screenshots enabled."""
    node = BrowserNode(
        id="screenshot_browse",
        task="Navigate and capture screenshots",
        save_screenshots=True,
    )

    assert node.save_screenshots is True


def test_browser_node_with_all_features():
    """Test BrowserNode with all features configured."""
    node = BrowserNode(
        id="full_featured",
        task="Extract data and capture screenshots",
        browser_profile_id="550e8400-e29b-41d4-a716-446655440000",
        max_steps=50,
        timeout_seconds=600,
        expect_outputs=OutputContract(
            mode=OutputMode.json,
            schema={"id": "data_schema"},
        ),
        save_screenshots=True,
        inputs={"url": "https://example.com"},
    )

    assert node.browser_profile_id == "550e8400-e29b-41d4-a716-446655440000"
    assert node.max_steps == 50
    assert node.timeout_seconds == 600
    assert node.expect_outputs is not None
    assert node.expect_outputs.mode == OutputMode.json
    assert node.save_screenshots is True
    assert node.inputs == {"url": "https://example.com"}


def test_browser_node_max_steps_validation():
    """Test max_steps validation bounds."""
    # Valid minimum
    node = BrowserNode(id="min_steps", task="test", max_steps=1)
    assert node.max_steps == 1

    # Valid maximum
    node = BrowserNode(id="max_steps", task="test", max_steps=100)
    assert node.max_steps == 100

    # Invalid: below minimum
    with pytest.raises(ValidationError):
        BrowserNode(id="invalid", task="test", max_steps=0)

    # Invalid: above maximum
    with pytest.raises(ValidationError):
        BrowserNode(id="invalid", task="test", max_steps=101)


def test_browser_node_timeout_validation():
    """Test timeout_seconds validation bounds."""
    # Valid minimum
    node = BrowserNode(id="min_timeout", task="test", timeout_seconds=30)
    assert node.timeout_seconds == 30

    # Valid maximum
    node = BrowserNode(id="max_timeout", task="test", timeout_seconds=1800)
    assert node.timeout_seconds == 1800

    # Invalid: below minimum
    with pytest.raises(ValidationError):
        BrowserNode(id="invalid", task="test", timeout_seconds=29)

    # Invalid: above maximum
    with pytest.raises(ValidationError):
        BrowserNode(id="invalid", task="test", timeout_seconds=1801)


# =============================================================================
# Type Environment Tests
# =============================================================================


def test_type_env_registers_browser_node_default_schema():
    """Test that browser node registers default output schema."""
    env = TypeEnvironment()
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    node = BrowserNode(
        id="browse",
        task="Extract data from webpage",
    )

    # Use registry-based approach
    node_impl = node_type_registry.get("browser")
    assert node_impl is not None, "BrowserNodeType should be registered"
    ctx = TypeRegistrationContext(schema_registry=schema_registry, tool_registry=tool_registry)
    node_impl.register_type_sync(node, env, ctx)

    # Verify schema was registered
    schema = env.get("browse")
    assert schema is not None
    assert schema["type"] == "object"
    assert "success" in schema["properties"]
    assert "result" in schema["properties"]
    assert "extracted_data" in schema["properties"]
    assert "final_url" in schema["properties"]


def test_type_env_browser_node_in_workflow():
    """Test browser node type environment in full workflow context."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            BrowserNode(
                id="scrape",
                task="Go to example.com and get the title",
            ),
            LLMNode(
                id="process",
                inputs={
                    "model": "gpt-4o",
                    "prompt": "Summarize: ${scrape.result}",
                },
            ),
        ],
        edges=[
            Edge(source="scrape", target="process"),
        ],
    )

    tool_registry = ToolRegistry()
    schema_registry = SchemaRegistry()

    env = build_type_environment(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )

    # Verify browser node output is registered
    scrape_schema = env.get("scrape")
    assert scrape_schema is not None
    assert scrape_schema["type"] == "object"


def test_type_env_browser_node_default_schema_includes_screenshots():
    """Test that browser node default schema includes screenshots field."""
    env = TypeEnvironment()
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    node = BrowserNode(
        id="browse",
        task="Extract data from webpage",
        save_screenshots=True,
    )

    # Use registry-based approach
    node_impl = node_type_registry.get("browser")
    assert node_impl is not None, "BrowserNodeType should be registered"
    ctx = TypeRegistrationContext(schema_registry=schema_registry, tool_registry=tool_registry)
    node_impl.register_type_sync(node, env, ctx)

    # Verify schema includes screenshots field
    schema = env.get("browse")
    assert schema is not None
    assert "screenshots" in schema["properties"]
    assert schema["properties"]["screenshots"]["type"] == "array"


def test_type_env_browser_node_with_expect_outputs():
    """Test browser node type environment with expect_outputs specified.

    Browser nodes ALWAYS output {success, result, extracted_data, final_url, screenshots}.
    When expect_outputs is provided, the user's schema applies to extracted_data only.
    """
    env = TypeEnvironment()
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    # Register a schema in the registry
    product_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
        },
    }
    schema_registry.register("product_schema", product_schema)

    node = BrowserNode(
        id="extract",
        task="Extract product details",
        expect_outputs=OutputContract(
            mode=OutputMode.json,
            schema={"id": "product_schema"},
        ),
    )

    # Use registry-based approach
    node_impl = node_type_registry.get("browser")
    assert node_impl is not None, "BrowserNodeType should be registered"
    ctx = TypeRegistrationContext(schema_registry=schema_registry, tool_registry=tool_registry)
    node_impl.register_type_sync(node, env, ctx)

    # Verify full browser output schema is registered with user schema in extracted_data
    schema = env.get("extract")
    assert schema is not None
    assert schema["type"] == "object"
    # Browser envelope fields should be present
    assert "success" in schema["properties"]
    assert "result" in schema["properties"]
    assert "final_url" in schema["properties"]
    assert "screenshots" in schema["properties"]
    # User's expect_outputs schema should be in extracted_data
    assert "extracted_data" in schema["properties"]
    assert schema["properties"]["extracted_data"] == product_schema


def test_type_env_browser_node_extracted_data_reference():
    """Test that ${browser_id.extracted_data.field} references work correctly.

    This regression test ensures that workflows referencing browser node outputs
    via extracted_data (e.g., ${browser-1.extracted_data.features}) compile
    without validation errors.
    """
    from seer.core.compiler.validate_refs import validate_references

    spec = WorkflowSpec(
        version="2",
        nodes=[
            BrowserNode(
                id="browser-1",
                task="Extract pricing data",
                expect_outputs=OutputContract(
                    mode=OutputMode.json,
                    schema={
                        "schema": {
                            "type": "object",
                            "properties": {
                                "features": {"type": "array"},
                                "pricing_tiers": {"type": "array"},
                            },
                        }
                    },
                ),
            ),
            LLMNode(
                id="llm-1",
                inputs={
                    "model": "gpt-4o",
                    "prompt": "What are these features? ${browser-1.extracted_data.features}",
                },
            ),
        ],
        edges=[
            Edge(source="browser-1", target="llm-1"),
        ],
    )

    tool_registry = ToolRegistry()
    schema_registry = SchemaRegistry()

    env = build_type_environment(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )

    # This should NOT raise - extracted_data.features is valid
    # validate_references raises ValidationPhaseError on failures
    validate_references(spec, env)  # Should complete without exception


# =============================================================================
# WorkflowSpec Parsing Tests
# =============================================================================


def test_workflow_spec_with_browser_node():
    """Test parsing a workflow spec containing a browser node."""
    spec_dict = {
        "version": "2",
        "nodes": [
            {
                "id": "browse",
                "type": "browser",
                "task": "Navigate to the target website and extract data",
                "browser_profile_id": "123e4567-e89b-12d3-a456-426614174000",
                "max_steps": 30,
                "timeout_seconds": 180,
            }
        ],
        "edges": [],
        "triggers": [],
    }

    spec = WorkflowSpec.model_validate(spec_dict)

    assert len(spec.nodes) == 1
    node = spec.nodes[0]
    assert isinstance(node, BrowserNode)
    assert node.type == "browser"
    assert node.task == "Navigate to the target website and extract data"
    assert node.browser_profile_id == "123e4567-e89b-12d3-a456-426614174000"


def test_workflow_spec_browser_with_llm_chain():
    """Test workflow with browser node feeding into LLM node."""
    spec_dict = {
        "version": "2",
        "nodes": [
            {
                "id": "scrape_slack",
                "type": "browser",
                "task": "Go to Slack and get messages from #general",
                "browser_profile_id": "work-profile-id",
            },
            {
                "id": "summarize",
                "type": "llm",
                "inputs": {
                    "model": "gpt-4o",
                    "prompt": "Summarize these messages: ${scrape_slack.result}",
                },
                "outputs": {"mode": "text"},
            },
        ],
        "edges": [
            {"source": "scrape_slack", "target": "summarize"},
        ],
        "triggers": [],
    }

    spec = WorkflowSpec.model_validate(spec_dict)

    assert len(spec.nodes) == 2
    assert isinstance(spec.nodes[0], BrowserNode)
    assert isinstance(spec.nodes[1], LLMNode)


def test_workflow_spec_browser_node_minimal():
    """Test minimal browser node configuration."""
    spec_dict = {
        "version": "2",
        "nodes": [
            {
                "id": "simple_browse",
                "type": "browser",
                "task": "Check if example.com is accessible",
            }
        ],
        "edges": [],
    }

    spec = WorkflowSpec.model_validate(spec_dict)
    node = spec.nodes[0]

    assert isinstance(node, BrowserNode)
    assert node.browser_profile_id is None
    assert node.max_steps == 25  # default
    assert node.timeout_seconds == 300  # default
    assert node.save_screenshots is False  # default


def test_workflow_spec_browser_with_save_screenshots():
    """Test parsing workflow spec with save_screenshots enabled."""
    spec_dict = {
        "version": "2",
        "nodes": [
            {
                "id": "screenshot_browse",
                "type": "browser",
                "task": "Navigate to page and capture screenshots",
                "save_screenshots": True,
            }
        ],
        "edges": [],
    }

    spec = WorkflowSpec.model_validate(spec_dict)
    node = spec.nodes[0]

    assert isinstance(node, BrowserNode)
    assert node.save_screenshots is True


def test_workflow_spec_browser_with_structured_output_and_screenshots():
    """Test browser node with both structured output and screenshot saving."""
    spec_dict = {
        "version": "2",
        "nodes": [
            {
                "id": "full_browser",
                "type": "browser",
                "task": "Extract product data and capture page screenshots",
                "save_screenshots": True,
                "expect_outputs": {
                    "mode": "json",
                    # InlineSchema format: "schema" key contains the actual JSON schema
                    "schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "product_name": {"type": "string"},
                                "price": {"type": "number"},
                            },
                        },
                    },
                },
            }
        ],
        "edges": [],
    }

    spec = WorkflowSpec.model_validate(spec_dict)
    node = spec.nodes[0]

    assert isinstance(node, BrowserNode)
    assert node.save_screenshots is True
    assert node.expect_outputs is not None
    assert node.expect_outputs.mode == OutputMode.json


# =============================================================================
# Node Discriminator Tests
# =============================================================================


def test_node_discriminator_identifies_browser():
    """Test that the Node union correctly identifies browser nodes."""
    from seer.core.schema.models import Node

    # Create spec with browser node
    spec_dict = {
        "version": "2",
        "nodes": [
            {"id": "n1", "type": "browser", "task": "Test task"},
            {"id": "n2", "type": "tool", "tool": "test_tool"},
            {"id": "n3", "type": "llm", "inputs": {"model": "gpt-4o", "prompt": "Hello"}},
        ],
        "edges": [],
    }

    spec = WorkflowSpec.model_validate(spec_dict)

    # Verify discriminator correctly identified types
    assert isinstance(spec.nodes[0], BrowserNode)
    assert spec.nodes[0].type == "browser"


# =============================================================================
# JSON Schema to Pydantic Conversion Tests
# =============================================================================


def test_json_type_to_python_basic_types():
    """Test basic JSON type to Python type mapping."""
    assert _json_type_to_python({"type": "string"}) == str
    assert _json_type_to_python({"type": "number"}) == float
    assert _json_type_to_python({"type": "integer"}) == int
    assert _json_type_to_python({"type": "boolean"}) == bool


def test_json_type_to_python_array():
    """Test JSON array type to Python List mapping."""
    from typing import List

    # Array of strings
    array_schema = {"type": "array", "items": {"type": "string"}}
    result = _json_type_to_python(array_schema)
    assert result == List[str]

    # Array of integers
    int_array_schema = {"type": "array", "items": {"type": "integer"}}
    result = _json_type_to_python(int_array_schema)
    assert result == List[int]


def test_json_type_to_python_object():
    """Test JSON object type to Python Dict mapping."""
    from typing import Any, Dict

    object_schema = {"type": "object"}
    result = _json_type_to_python(object_schema)
    assert result == Dict[str, Any]


def test_json_type_to_python_unknown():
    """Test unknown/missing type defaults to Any."""
    from typing import Any

    assert _json_type_to_python({}) == Any
    assert _json_type_to_python({"type": "unknown"}) == Any


def test_json_schema_to_pydantic_basic():
    """Test basic JSON schema to Pydantic model conversion."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
        },
        "required": ["name"],
    }

    Model = json_schema_to_pydantic(schema, "ProductModel")

    # Verify it's a Pydantic model
    assert issubclass(Model, BaseModel)

    # Verify required field works
    instance = Model(name="Widget")
    assert instance.name == "Widget"
    assert instance.price is None  # Optional, defaults to None

    # Verify with all fields
    instance = Model(name="Gadget", price=29.99)
    assert instance.name == "Gadget"
    assert instance.price == 29.99


def test_json_schema_to_pydantic_all_optional():
    """Test schema where all fields are optional."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "count": {"type": "integer"},
        },
        # No "required" field means all are optional
    }

    Model = json_schema_to_pydantic(schema)

    # All fields default to None
    instance = Model()
    assert instance.title is None
    assert instance.count is None


def test_json_schema_to_pydantic_with_arrays():
    """Test schema with array fields."""
    schema = {
        "type": "object",
        "properties": {
            "features": {"type": "array", "items": {"type": "string"}},
            "pricing_tiers": {"type": "array", "items": {"type": "object"}},
        },
    }

    Model = json_schema_to_pydantic(schema, "PricingModel")

    instance = Model(features=["Fast", "Reliable"], pricing_tiers=[{"name": "Pro", "price": 10}])
    assert instance.features == ["Fast", "Reliable"]
    assert instance.pricing_tiers == [{"name": "Pro", "price": 10}]


def test_json_schema_to_pydantic_non_object():
    """Test non-object schema wraps in a data field."""
    from typing import Any

    # String schema
    schema = {"type": "string"}
    Model = json_schema_to_pydantic(schema)

    # Should have a 'data' field
    assert "data" in Model.model_fields


def test_json_schema_to_pydantic_required_validation():
    """Test that required fields are enforced."""
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
        },
        "required": ["id", "name"],
    }

    Model = json_schema_to_pydantic(schema)

    # Should fail without required fields
    with pytest.raises(ValidationError):
        Model()

    # Should fail with only one required field
    with pytest.raises(ValidationError):
        Model(id=1)

    # Should succeed with all required fields
    instance = Model(id=1, name="Test")
    assert instance.id == 1
    assert instance.name == "Test"


def test_json_schema_to_pydantic_browser_use_case():
    """Test typical browser extraction schema like pricing data."""
    # This is the schema from the plan's verification example
    schema = {
        "type": "object",
        "properties": {
            "features": {"type": "array", "items": {"type": "string"}},
            "pricing_tiers": {"type": "array", "items": {"type": "object"}},
        },
    }

    Model = json_schema_to_pydantic(schema, "BrowserOutputModel")

    # Verify the model can be instantiated with typical browser output
    instance = Model(
        features=["AI-powered automation", "No-code builder", "Integrations"],
        pricing_tiers=[
            {"name": "Free", "price": 0, "features": ["Basic"]},
            {"name": "Pro", "price": 29, "features": ["All features"]},
        ],
    )

    assert len(instance.features) == 3
    assert len(instance.pricing_tiers) == 2

    # Verify JSON serialization (what BrowserUse does internally)
    json_output = instance.model_dump(mode="json")
    assert "features" in json_output
    assert "pricing_tiers" in json_output


# =============================================================================
# Usage Metadata Extraction Tests
# =============================================================================


def test_extract_usage_metadata_from_history():
    """Test extraction of usage metadata from browser_use history."""
    from unittest.mock import MagicMock
    from seer.services.browser.browser_service import BrowserService

    mock_history = MagicMock()
    mock_usage = MagicMock()
    mock_usage.total_prompt_tokens = 5000
    mock_usage.total_completion_tokens = 2000
    mock_usage.total_tokens = 7000
    mock_usage.entry_count = 10
    mock_history.usage = mock_usage

    result = BrowserService._extract_usage_metadata(mock_history)

    assert result is not None
    assert result["model"] == "moonshotai/kimi-k2.5"
    assert result["input_tokens"] == 5000
    assert result["output_tokens"] == 2000
    assert result["reasoning_tokens"] == 0
    assert result["total_tokens"] == 7000
    assert result["steps_taken"] == 10


def test_extract_usage_metadata_no_usage():
    """Test extraction when usage is None."""
    from unittest.mock import MagicMock
    from seer.services.browser.browser_service import BrowserService

    mock_history = MagicMock()
    mock_history.usage = None

    result = BrowserService._extract_usage_metadata(mock_history)
    assert result is None


def test_extract_usage_metadata_zero_tokens():
    """Test extraction when all tokens are zero."""
    from unittest.mock import MagicMock
    from seer.services.browser.browser_service import BrowserService

    mock_history = MagicMock()
    mock_usage = MagicMock()
    mock_usage.total_prompt_tokens = 0
    mock_usage.total_completion_tokens = 0
    mock_usage.total_tokens = 0
    mock_usage.entry_count = 0
    mock_history.usage = mock_usage

    result = BrowserService._extract_usage_metadata(mock_history)
    assert result is None


def test_extract_usage_metadata_no_usage_attr():
    """Test extraction when history lacks usage attribute."""
    from seer.services.browser.browser_service import BrowserService

    # Plain object without usage attribute
    result = BrowserService._extract_usage_metadata(object())
    assert result is None


# =============================================================================
# Browser Node Cost Tracking Tests
# =============================================================================


@pytest.mark.asyncio
class TestBrowserNodeCostTracking:
    """Tests for browser node credit checking and cost tracking."""

    async def test_check_credit_limit_called(self, mock_user):
        """Test that credit check is called before browser execution."""
        from unittest.mock import AsyncMock, MagicMock, patch

        browser_node_type = node_type_registry.get("browser")

        mock_context = MagicMock()
        mock_context.user = mock_user

        with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock) as mock_check:
            await browser_node_type._check_credit_limit(mock_context)
            mock_check.assert_called_once_with(mock_user)

    async def test_check_credit_limit_no_context(self):
        """Test credit check is skipped when no context."""
        from unittest.mock import AsyncMock, patch

        browser_node_type = node_type_registry.get("browser")

        with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock) as mock_check:
            await browser_node_type._check_credit_limit(None)
            mock_check.assert_not_called()

    async def test_check_credit_limit_no_user(self):
        """Test credit check is skipped when context has no user."""
        from unittest.mock import AsyncMock, MagicMock, patch

        browser_node_type = node_type_registry.get("browser")

        mock_context = MagicMock()
        mock_context.user = None

        with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock) as mock_check:
            await browser_node_type._check_credit_limit(mock_context)
            mock_check.assert_not_called()

    async def test_track_usage_calls_cost_tracker(self, mock_user):
        """Test that _track_usage_async calls CostTracker with correct params."""
        from unittest.mock import AsyncMock, patch
        from seer.core.runtime.context import WorkflowRuntimeContext

        browser_node_type = node_type_registry.get("browser")

        runtime_context = WorkflowRuntimeContext(
            user=mock_user,
            workflow_run_id="run_test_123",
            per_run_cost_cap_usd=5.0,
            accumulated_cost_usd=0.0,
        )

        usage_metadata = {
            "model": "moonshotai/kimi-k2.5",
            "input_tokens": 5000,
            "output_tokens": 2000,
            "reasoning_tokens": 0,
            "steps_taken": 10,
        }

        with patch("seer.observability.cost_tracking.CostTracker.track_and_enforce_cap", new_callable=AsyncMock) as mock_track:
            await browser_node_type._track_usage_async(usage_metadata, runtime_context, "browse-1")

            mock_track.assert_called_once()
            call_kwargs = mock_track.call_args.kwargs
            assert call_kwargs["operation"] == "browser_execution"
            assert call_kwargs["extra_metadata"]["node_id"] == "browse-1"
            assert call_kwargs["extra_metadata"]["steps_taken"] == 10
            assert call_kwargs["extra_metadata"]["aggregated"] is True

    async def test_track_usage_no_user_context(self):
        """Test usage tracking is skipped when no user context."""
        from unittest.mock import AsyncMock, patch

        browser_node_type = node_type_registry.get("browser")

        with patch("seer.observability.cost_tracking.CostTracker.track_and_enforce_cap", new_callable=AsyncMock) as mock_track:
            await browser_node_type._track_usage_async({}, None, "browse-1")
            mock_track.assert_not_called()

    async def test_track_usage_propagates_cost_cap_exceeded(self, mock_user):
        """Test that RunCostCapExceeded propagates from _track_usage_async."""
        from unittest.mock import AsyncMock, patch
        from seer.core.runtime.context import WorkflowRuntimeContext
        from seer.observability.exceptions import RunCostCapExceeded

        browser_node_type = node_type_registry.get("browser")

        runtime_context = WorkflowRuntimeContext(
            user=mock_user,
            workflow_run_id="run_test_123",
            per_run_cost_cap_usd=5.0,
            accumulated_cost_usd=0.0,
        )

        usage_metadata = {
            "model": "moonshotai/kimi-k2.5",
            "input_tokens": 5000,
            "output_tokens": 2000,
            "reasoning_tokens": 0,
        }

        with patch(
            "seer.observability.cost_tracking.CostTracker.track_and_enforce_cap",
            new_callable=AsyncMock,
            side_effect=RunCostCapExceeded(
                run_identifier="run_test_123",
                accumulated_cost=6.0,
                cost_cap=5.0,
                run_type="workflow",
            ),
        ):
            with pytest.raises(RunCostCapExceeded):
                await browser_node_type._track_usage_async(usage_metadata, runtime_context, "browse-1")
