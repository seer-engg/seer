"""
Unit tests for BrowserNode implementation.

Tests schema validation, type environment registration, and workflow spec parsing.
"""
from unittest.mock import ANY

import pytest
from pydantic import BaseModel, ValidationError

from seer.core.compiler.type_env import build_type_environment
from seer.core.errors import ValidationPhaseError
from seer.core.nodes.base import TypeRegistrationContext
from seer.core.nodes.registry import node_type_registry
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    AgentNode,
    BrowserNode,
    Edge,
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


def test_browser_node_with_custom_model():
    """Test BrowserNode with custom model specified."""
    node = BrowserNode(
        id="browse",
        task="Extract data from webpage",
        model="openai/gpt-4o",
    )

    assert node.model == "openai/gpt-4o"


def test_browser_node_model_defaults_to_gemini():
    """Test BrowserNode model defaults to qwen3-vl."""
    node = BrowserNode(
        id="browse",
        task="Extract data from webpage",
    )

    assert node.model == "qwen/qwen3-vl-8b-thinking"


def test_browser_node_with_claude_model():
    """Test BrowserNode with Claude model."""
    node = BrowserNode(
        id="browse",
        task="Extract data from webpage",
        model="anthropic/claude-sonnet-4.5",
    )

    assert node.model == "anthropic/claude-sonnet-4.5"


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
        model="openai/gpt-4o",
    )

    assert node.browser_profile_id == "550e8400-e29b-41d4-a716-446655440000"
    assert node.max_steps == 50
    assert node.timeout_seconds == 600
    assert node.expect_outputs is not None
    assert node.expect_outputs.mode == OutputMode.json
    assert node.save_screenshots is True
    assert node.inputs == {"url": "https://example.com"}
    assert node.model == "openai/gpt-4o"


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
    """Test that browser node registers default output schema with all fields."""
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

    # Core fields
    assert "success" in schema["properties"]
    assert "result" in schema["properties"]
    assert "extracted_data" in schema["properties"]
    assert "final_url" in schema["properties"]
    assert "screenshots" in schema["properties"]

    # New enhanced output fields
    assert "urls" in schema["properties"]
    assert schema["properties"]["urls"]["type"] == "array"
    assert schema["properties"]["urls"]["items"]["type"] == "string"

    assert "duration_seconds" in schema["properties"]
    assert "number" in schema["properties"]["duration_seconds"]["type"]

    assert "steps_count" in schema["properties"]
    assert "integer" in schema["properties"]["steps_count"]["type"]

    assert "extracted_content" in schema["properties"]
    assert schema["properties"]["extracted_content"]["type"] == "array"

    assert "model_thoughts" in schema["properties"]
    assert schema["properties"]["model_thoughts"]["type"] == "array"
    assert schema["properties"]["model_thoughts"]["items"]["type"] == "object"

    assert "model_actions" in schema["properties"]
    assert schema["properties"]["model_actions"]["type"] == "array"
    assert schema["properties"]["model_actions"]["items"]["type"] == "object"


def test_type_env_browser_node_in_workflow():
    """Test browser node type environment in full workflow context."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            BrowserNode(
                id="scrape",
                task="Go to example.com and get the title",
            ),
            AgentNode(
                id="process",
                inputs={
                    "model": "openai/gpt-oss-120b",
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
    # New enhanced fields should also be present
    assert "urls" in schema["properties"]
    assert "duration_seconds" in schema["properties"]
    assert "steps_count" in schema["properties"]
    assert "extracted_content" in schema["properties"]
    assert "model_thoughts" in schema["properties"]
    assert "model_actions" in schema["properties"]
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
                            "required": ["features", "pricing_tiers"],
                        }
                    },
                ),
            ),
            AgentNode(
                id="llm-1",
                inputs={
                    "model": "openai/gpt-oss-120b",
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


def test_type_env_browser_node_rejects_invalid_property_access():
    """Test that invalid references like ${browser_id.shops} are caught at compile time.

    This regression test ensures that the type checker catches invalid property access
    on browser node outputs. Users must use ${node.extracted_data.field} not ${node.field}.

    Bug context: Browser node schema had additionalProperties={} which allowed any
    property access to pass validation. Changed to additionalProperties=False to
    enforce strict type checking.
    """
    from seer.core.compiler.validate_refs import validate_references
    from seer.core.errors import ValidationPhaseError

    spec = WorkflowSpec(
        version="2",
        nodes=[
            BrowserNode(
                id="scrape_maps",
                task="Find coffee shops",
                expect_outputs=OutputContract(
                    mode=OutputMode.json,
                    schema={
                        "schema": {
                            "type": "object",
                            "properties": {
                                "shops": {"type": "array"},
                            },
                            "required": ["shops"],
                        }
                    },
                ),
            ),
            AgentNode(
                id="format_data",
                inputs={
                    "model": "openai/gpt-oss-120b",
                    # WRONG: should be ${scrape_maps.extracted_data.shops}
                    "prompt": "Format these shops: ${scrape_maps.shops}",
                },
            ),
        ],
        edges=[
            Edge(source="scrape_maps", target="format_data"),
        ],
    )

    tool_registry = ToolRegistry()
    schema_registry = SchemaRegistry()

    env = build_type_environment(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )

    # This SHOULD raise - shops is not a direct property of browser output
    # The correct reference is ${scrape_maps.extracted_data.shops}
    with pytest.raises(ValidationPhaseError) as exc_info:
        validate_references(spec, env)

    assert "shops" in str(exc_info.value)
    assert "scrape_maps" in str(exc_info.value)


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
                "type": "agent",
                "inputs": {
                    "model": "openai/gpt-oss-120b",
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
    assert isinstance(spec.nodes[1], AgentNode)


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


def test_workflow_spec_browser_with_model():
    """Test parsing workflow spec with custom model specified."""
    spec_dict = {
        "version": "2",
        "nodes": [
            {
                "id": "browse",
                "type": "browser",
                "task": "Extract data from webpage",
                "model": "openai/gpt-4o",
            }
        ],
        "edges": [],
    }

    spec = WorkflowSpec.model_validate(spec_dict)
    node = spec.nodes[0]

    assert isinstance(node, BrowserNode)
    assert node.model == "openai/gpt-4o"


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
                            "required": ["product_name", "price"],
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
            {"id": "n3", "type": "agent", "inputs": {"model": "openai/gpt-oss-120b", "prompt": "Hello"}},
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


def test_json_schema_to_pydantic_nested_objects():
    """Test that nested objects are converted to proper Pydantic models, not Dict[str, Any].

    This is critical for OpenAI's strict JSON Schema validation. When browser-use
    extracts the schema via .model_json_schema(), nested objects must be proper models
    to generate valid schemas with properties and required fields.
    """
    # Schema with nested objects (like the user's shops extraction schema)
    schema = {
        "type": "object",
        "required": ["shops"],
        "properties": {
            "shops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "address"],
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                        "phone": {"type": "string"},
                        "rating": {"type": "number"},
                    },
                },
            }
        },
    }

    Model = json_schema_to_pydantic(schema)

    # Verify shops field exists and is a List
    assert "shops" in Model.model_fields
    shops_field = Model.model_fields["shops"]

    # Get the annotation (should be List[SomeNestedModel])
    from typing import get_origin, get_args

    shops_type = shops_field.annotation
    assert get_origin(shops_type) is list, "shops should be a List type"

    # Get the item type (should be a Pydantic BaseModel, not dict)
    item_type = get_args(shops_type)[0]
    assert issubclass(item_type, BaseModel), (
        f"Array items should be a Pydantic model, got {item_type}. "
        "Dict[str, Any] would break OpenAI's strict JSON Schema validation."
    )

    # Verify nested model has the expected fields
    assert "name" in item_type.model_fields
    assert "address" in item_type.model_fields
    assert "phone" in item_type.model_fields
    assert "rating" in item_type.model_fields

    # Verify we can instantiate with proper data
    instance = Model(
        shops=[
            {"name": "Coffee Shop", "address": "123 Main St", "phone": "555-1234", "rating": 4.5},
            {"name": "Tea House", "address": "456 Oak Ave"},  # phone/rating are optional
        ]
    )
    assert len(instance.shops) == 2
    assert instance.shops[0].name == "Coffee Shop"
    assert instance.shops[1].address == "456 Oak Ave"

    # Verify JSON schema has proper nested structure (what browser-use sends to OpenAI)
    json_schema = Model.model_json_schema()
    # The shops items should have properties, not just {"type": "object"}
    shops_items_schema = json_schema.get("$defs", {})
    assert len(shops_items_schema) > 0, "Nested models should create $defs entries"


# =============================================================================
# Usage Metadata Extraction Tests
# =============================================================================


def test_extract_usage_metadata_from_history():
    """Test extraction of usage metadata from browser_use history."""
    from unittest.mock import MagicMock, patch
    from seer.services.browser.browser_service import BrowserService

    mock_history = MagicMock()
    mock_usage = MagicMock()
    mock_usage.total_prompt_tokens = 5000
    mock_usage.total_completion_tokens = 2000
    mock_usage.total_tokens = 7000
    mock_usage.entry_count = 10
    mock_history.usage = mock_usage

    with patch("seer.services.browser.browser_service.config") as mock_config:
        mock_config.default_llm_model = "test-model"
        result = BrowserService._extract_usage_metadata(mock_history)

    assert result is not None
    assert result["model"] == "test-model"
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

        with patch("seer.core.nodes.browser_node.check_runtime_credit_limit", new_callable=AsyncMock) as mock_check:
            await browser_node_type._check_credit_limit(mock_context)
            mock_check.assert_awaited_once_with(mock_context, ANY)

    async def test_check_credit_limit_no_context(self):
        """Test credit check is skipped when no context."""
        from unittest.mock import AsyncMock, patch

        browser_node_type = node_type_registry.get("browser")

        with patch("seer.core.nodes.browser_node.check_runtime_credit_limit", new_callable=AsyncMock) as mock_check:
            await browser_node_type._check_credit_limit(None)
            mock_check.assert_awaited_once_with(None, ANY)

    async def test_check_credit_limit_no_user(self):
        """Test credit check is skipped when context has no user."""
        from unittest.mock import AsyncMock, MagicMock, patch

        browser_node_type = node_type_registry.get("browser")

        mock_context = MagicMock()
        mock_context.user = None

        with patch("seer.core.nodes.browser_node.check_runtime_credit_limit", new_callable=AsyncMock) as mock_check:
            await browser_node_type._check_credit_limit(mock_context)
            mock_check.assert_awaited_once_with(mock_context, ANY)

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


# =============================================================================
# BrowserNodeType execute_async Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.unit
class TestBrowserNodeExecuteAsync:
    """Tests for BrowserNodeType.execute_async method."""

    async def test_execute_async_success(self, mock_user):
        """Test successful browser task execution."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from seer.core.nodes.base import NodeExecutionContext
        from seer.core.runtime.context import WorkflowRuntimeContext
        from seer.core.expr.typecheck import TypeEnvironment

        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Go to example.com and get the title",
        )

        runtime_context = WorkflowRuntimeContext(
            user=mock_user,
            workflow_run_id="run_test_123",
            per_run_cost_cap_usd=10.0,
            accumulated_cost_usd=0.0,
        )

        ctx = NodeExecutionContext(
            state={},
            config={},
            locals_ctx={},
            trigger={},
            runtime_context=runtime_context,
        )

        mock_services = MagicMock()
        mock_services.type_env = TypeEnvironment()

        mock_result = {
            "success": True,
            "result": "Page title: Example Domain",
            "extracted_data": {},
            "final_url": "https://example.com",
            "screenshots": [],
            "usage": {
                "model": "moonshotai/kimi-k2.5",
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
                "steps_taken": 3,
            },
        }

        with patch("seer.services.browser.BrowserService") as mock_browser_service_cls:
            mock_instance = MagicMock()
            mock_instance.execute_task = AsyncMock(return_value=mock_result.copy())
            mock_browser_service_cls.instance.return_value = mock_instance

            with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock):
                with patch("seer.observability.cost_tracking.CostTracker.track_and_enforce_cap", new_callable=AsyncMock):
                    result = await browser_node_type.execute_async(node, ctx, mock_services)

        assert "browse-1" in result
        assert result["browse-1"]["success"] is True
        assert result["browse-1"]["result"] == "Page title: Example Domain"

    async def test_execute_async_with_inputs(self, mock_user):
        """Test browser task execution with inputs provided."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from seer.core.nodes.base import NodeExecutionContext
        from seer.core.runtime.context import WorkflowRuntimeContext
        from seer.core.expr.typecheck import TypeEnvironment

        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Search for test query",
            inputs={
                "query": "test search term",
                "max_results": 10,
            },
        )

        runtime_context = WorkflowRuntimeContext(
            user=mock_user,
            workflow_run_id="run_test_456",
        )

        ctx = NodeExecutionContext(
            state={},
            config={},
            locals_ctx={},
            trigger={},
            runtime_context=runtime_context,
        )

        mock_services = MagicMock()
        mock_services.type_env = TypeEnvironment()

        mock_result = {
            "success": True,
            "result": "Found 5 results",
            "extracted_data": {},
            "final_url": "https://search.example.com",
            "screenshots": [],
        }

        with patch("seer.services.browser.BrowserService") as mock_browser_service_cls:
            mock_instance = MagicMock()
            mock_instance.execute_task = AsyncMock(return_value=mock_result.copy())
            mock_browser_service_cls.instance.return_value = mock_instance

            with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock):
                result = await browser_node_type.execute_async(node, ctx, mock_services)

        # Verify execute_task was called with evaluated inputs
        call_kwargs = mock_instance.execute_task.call_args.kwargs
        assert "inputs" in call_kwargs
        assert call_kwargs["inputs"]["query"] == "test search term"
        assert call_kwargs["inputs"]["max_results"] == 10

    async def test_execute_async_credit_limit_exceeded(self, mock_user):
        """Test that CreditLimitExceeded is raised from execute_async."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from seer.core.nodes.base import NodeExecutionContext
        from seer.core.runtime.context import WorkflowRuntimeContext
        from seer.core.expr.typecheck import TypeEnvironment
        from seer.observability.exceptions import CreditLimitExceeded
        from seer.database.subscription_models import SubscriptionTier

        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Go to example.com",
        )

        runtime_context = WorkflowRuntimeContext(
            user=mock_user,
            workflow_run_id="run_test_789",
        )

        ctx = NodeExecutionContext(
            state={},
            config={},
            locals_ctx={},
            trigger={},
            runtime_context=runtime_context,
        )

        mock_services = MagicMock()
        mock_services.type_env = TypeEnvironment()

        with patch(
            "seer.observability.credit_gate.check_credit_limit",
            new_callable=AsyncMock,
            side_effect=CreditLimitExceeded(
                limit=50.0,
                current=100.0,
                tier=SubscriptionTier.FREE,
            ),
        ):
            with pytest.raises(CreditLimitExceeded):
                await browser_node_type.execute_async(node, ctx, mock_services)

    async def test_execute_async_browser_task_failure(self, mock_user):
        """Test that browser task failures raise ExecutionError."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from seer.core.nodes.base import NodeExecutionContext
        from seer.core.runtime.context import WorkflowRuntimeContext
        from seer.core.expr.typecheck import TypeEnvironment
        from seer.core.errors import ExecutionError

        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Go to nonexistent.example.com",
        )

        runtime_context = WorkflowRuntimeContext(
            user=mock_user,
            workflow_run_id="run_test_error",
        )

        ctx = NodeExecutionContext(
            state={},
            config={},
            locals_ctx={},
            trigger={},
            runtime_context=runtime_context,
        )

        mock_services = MagicMock()
        mock_services.type_env = TypeEnvironment()

        with patch("seer.services.browser.BrowserService") as mock_browser_service_cls:
            mock_instance = MagicMock()
            mock_instance.execute_task = AsyncMock(
                side_effect=RuntimeError("Browser timeout")
            )
            mock_browser_service_cls.instance.return_value = mock_instance

            with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock):
                with pytest.raises(ExecutionError) as exc_info:
                    await browser_node_type.execute_async(node, ctx, mock_services)

        assert "Browser task failed" in str(exc_info.value)
        # Verify trace data was added to state
        assert any("browse-1" in key for key in ctx.state.keys())

    async def test_execute_async_timeout_skips_validation(self, mock_user):
        """Regression test: timeout with expect_outputs should NOT raise validation error.

        When browser task times out, success=False and extracted_data={}.
        Even if expect_outputs requires fields, validation should be skipped
        because the task failed.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from seer.core.nodes.base import NodeExecutionContext
        from seer.core.runtime.context import WorkflowRuntimeContext
        from seer.core.expr.typecheck import TypeEnvironment

        browser_node_type = node_type_registry.get("browser")

        # Node with expect_outputs that requires fields
        node = BrowserNode(
            id="scrape-1",
            task="Extract data from page",
            expect_outputs=OutputContract(
                mode=OutputMode.json,
                schema={
                    "schema": {
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["headline", "description"],
                    }
                },
            ),
        )

        runtime_context = WorkflowRuntimeContext(
            user=mock_user,
            workflow_run_id="run_timeout_test",
        )

        ctx = NodeExecutionContext(
            state={},
            config={},
            locals_ctx={},
            trigger={},
            runtime_context=runtime_context,
        )

        # Build type_env with the schema
        mock_services = MagicMock()
        type_env = TypeEnvironment()
        type_env.register(
            "scrape-1",
            {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": "string"},
                    "extracted_data": {
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["headline", "description"],
                    },
                    "final_url": {"type": ["string", "null"]},
                    "screenshots": {"type": "array", "items": {"type": "string"}},
                },
            },
        )
        mock_services.type_env = type_env

        # Simulate timeout result: success=False, extracted_data={}
        timeout_result = {
            "success": False,
            "result": "Task timed out after 120 seconds",
            "extracted_data": {},
            "final_url": None,
            "screenshots": [],
        }

        with patch("seer.services.browser.BrowserService") as mock_browser_service_cls:
            mock_instance = MagicMock()
            mock_instance.execute_task = AsyncMock(return_value=timeout_result.copy())
            mock_browser_service_cls.instance.return_value = mock_instance

            with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock):
                # This should NOT raise ExecutionError for validation
                result = await browser_node_type.execute_async(node, ctx, mock_services)

        # Verify the timeout result is returned without validation error
        assert "scrape-1" in result
        assert result["scrape-1"]["success"] is False
        assert "timed out" in result["scrape-1"]["result"]
        assert result["scrape-1"]["extracted_data"] == {}


# =============================================================================
# BrowserNodeType Helper Method Tests
# =============================================================================


@pytest.mark.unit
class TestBrowserNodeHelpers:
    """Tests for BrowserNodeType helper methods."""

    def test_get_extraction_schema_with_json_mode(self):
        """Test extraction schema retrieval with JSON output mode."""
        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="extract-1",
            task="Extract product data",
            expect_outputs=OutputContract(
                mode=OutputMode.json,
                schema={"id": "product_schema"},
            ),
        )

        type_schemas = {
            "extract-1": {
                "type": "object",
                "properties": {
                    "extracted_data": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    }
                },
            }
        }

        result = browser_node_type._get_extraction_schema(node, type_schemas)

        assert result is not None
        assert result["type"] == "object"
        assert "name" in result["properties"]

    def test_get_extraction_schema_without_expect_outputs(self):
        """Test that no schema is returned when expect_outputs is not specified."""
        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Just browse",
        )

        type_schemas = {}

        result = browser_node_type._get_extraction_schema(node, type_schemas)

        assert result is None

    def test_get_extraction_schema_text_mode(self):
        """Test that no schema is returned for text output mode."""
        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Just browse",
            expect_outputs=OutputContract(
                mode=OutputMode.text,
            ),
        )

        type_schemas = {}

        result = browser_node_type._get_extraction_schema(node, type_schemas)

        assert result is None

    def test_get_screenshot_context_enabled(self):
        """Test screenshot context retrieval when enabled."""
        from unittest.mock import MagicMock
        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="screenshot-1",
            task="Take screenshots",
            save_screenshots=True,
        )

        runtime_context = MagicMock()
        runtime_context.has_file_system = True
        runtime_context.file_system = MagicMock()
        runtime_context.workflow_run_id = "run_123"

        file_system, workflow_run_id = browser_node_type._get_screenshot_context(node, runtime_context)

        assert file_system is runtime_context.file_system
        assert workflow_run_id == "run_123"

    def test_get_screenshot_context_disabled(self):
        """Test screenshot context when save_screenshots is False.

        Note: workflow_run_id is still returned even when save_screenshots=False
        because session recordings need it for associating with workflow runs.
        """
        from unittest.mock import MagicMock
        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Browse without screenshots",
            save_screenshots=False,
        )

        runtime_context = MagicMock()
        runtime_context.workflow_run_id = "run_456"

        file_system, workflow_run_id = browser_node_type._get_screenshot_context(node, runtime_context)

        # file_system should be None when save_screenshots=False
        assert file_system is None
        # workflow_run_id is always returned (needed for session recordings)
        assert workflow_run_id == "run_456"

    def test_get_screenshot_context_no_runtime_context(self):
        """Test screenshot context when runtime_context is None."""
        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Browse",
            save_screenshots=True,
        )

        file_system, workflow_run_id = browser_node_type._get_screenshot_context(node, None)

        assert file_system is None
        assert workflow_run_id is None

    def test_evaluate_inputs_success(self):
        """Test input evaluation with valid expressions."""
        from unittest.mock import MagicMock
        from seer.core.expr.evaluator import EvaluationContext

        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Search",
            inputs={
                "static_value": "hello",
                "number_value": 42,
            },
        )

        eval_ctx = EvaluationContext(
            state={},
            locals={},
            config={},
            trigger={},
        )

        result = browser_node_type._evaluate_inputs(node, eval_ctx)

        assert result["static_value"] == "hello"
        assert result["number_value"] == 42

    def test_evaluate_inputs_with_error(self):
        """Test input evaluation captures errors in result."""
        from seer.core.expr.evaluator import EvaluationContext

        browser_node_type = node_type_registry.get("browser")

        node = BrowserNode(
            id="browse-1",
            task="Search",
            inputs={
                "invalid_expr": "${undefined.path.to.value}",
            },
        )

        eval_ctx = EvaluationContext(
            state={},
            locals={},
            config={},
            trigger={},
        )

        result = browser_node_type._evaluate_inputs(node, eval_ctx)

        # Error should be captured in result
        assert "__error__" in result["invalid_expr"]
        assert "__expression__" in result["invalid_expr"]
        assert result["invalid_expr"]["__expression__"] == "${undefined.path.to.value}"
