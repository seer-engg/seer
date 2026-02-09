"""
Unit tests for BrowserNode implementation.

Tests schema validation, type environment registration, and workflow spec parsing.
"""

import pytest
from pydantic import ValidationError

from seer.core.compiler.type_env import (
    build_type_environment,
    _process_browser_node,
)
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


def test_browser_node_empty_task_invalid():
    """Test that empty task is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        BrowserNode(
            id="invalid",
            task="",
        )

    assert "min_length" in str(exc_info.value).lower() or "at least 1" in str(exc_info.value).lower()


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

    node = BrowserNode(
        id="browse",
        task="Extract data from webpage",
    )

    _process_browser_node(node, env, schema_registry)

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
