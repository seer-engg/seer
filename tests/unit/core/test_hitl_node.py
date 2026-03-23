"""
Unit tests for HITL (Human-In-The-Loop) node implementation.

Tests schema validation, type environment, reference validation, and runtime executor.
"""

import pytest
from unittest.mock import MagicMock, patch

from seer.core.compiler.type_env import build_type_environment
from seer.core.nodes.hitl_node import _build_hitl_output_schema, _evaluate_input_fields
from seer.core.compiler.validate_refs import validate_references, _validate_hitl
from seer.core.expr.evaluator import EvaluationContext
from seer.core.errors import NodeError
from seer.core.expr.typecheck import TypeEnvironment, Scope
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    Edge,
    HITLDisplayItem,
    HITLInputField,
    HITLInputOption,
    HITLInputType,
    AgentNode,
    HITLNode,
    ToolNode,
    WorkflowSpec,
)
from seer.core.schema.schema_registry import SchemaRegistry

pytestmark = pytest.mark.unit


# =============================================================================
# Schema Validation Tests
# =============================================================================


def test_hitl_node_basic_valid():
    """Test basic valid HITL node creation."""
    node = HITLNode(
        id="approval",
        title="Approval Required",
        description="Please approve this action",
        display=[
            HITLDisplayItem(label="Summary", value="${previous_node.summary}"),
        ],
        inputs=[
            HITLInputField(
                id="decision",
                question="Do you approve?",
                input_type=HITLInputType.single_choice,
                options=[
                    HITLInputOption(value="approve", label="Approve"),
                    HITLInputOption(value="reject", label="Reject"),
                ],
                required=True,
            ),
        ],
        timeout_seconds=3600,
    )

    assert node.id == "approval"
    assert node.type == "hitl"
    assert node.title == "Approval Required"
    assert len(node.display) == 1
    assert len(node.inputs) == 1


def test_hitl_node_all_input_types():
    """Test HITL node with all input types."""
    node = HITLNode(
        id="survey",
        title="User Survey",
        inputs=[
            HITLInputField(
                id="rating",
                question="Rate 1-10",
                input_type=HITLInputType.number,
                required=True,
            ),
            HITLInputField(
                id="feedback",
                question="Comments",
                input_type=HITLInputType.text,
                required=False,
            ),
            HITLInputField(
                id="recommend",
                question="Would you recommend?",
                input_type=HITLInputType.boolean,
                required=True,
            ),
            HITLInputField(
                id="category",
                question="Select category",
                input_type=HITLInputType.single_choice,
                options=[
                    HITLInputOption(value="a", label="A"),
                    HITLInputOption(value="b", label="B"),
                ],
                required=True,
            ),
            HITLInputField(
                id="features",
                question="Select features",
                input_type=HITLInputType.multi_choice,
                options=[
                    HITLInputOption(value="f1", label="Feature 1"),
                    HITLInputOption(value="f2", label="Feature 2"),
                ],
                required=False,
            ),
        ],
    )

    assert len(node.inputs) == 5


def test_hitl_node_no_inputs():
    """Test HITL node with no inputs (display only)."""
    node = HITLNode(
        id="info_display",
        title="Information",
        display=[
            HITLDisplayItem(label="Status", value="${status.value}"),
        ],
        inputs=[],
    )

    assert len(node.inputs) == 0
    assert len(node.display) == 1


def test_hitl_node_no_timeout():
    """Test HITL node with no timeout (indefinite wait)."""
    node = HITLNode(
        id="no_timeout",
        title="No Timeout",
        inputs=[
            HITLInputField(
                id="response",
                question="Your response",
                input_type=HITLInputType.text,
            ),
        ],
        timeout_seconds=None,
    )

    assert node.timeout_seconds is None


def test_hitl_node_zero_timeout():
    """Test HITL node with zero timeout (also indefinite)."""
    node = HITLNode(
        id="zero_timeout",
        title="Zero Timeout",
        inputs=[
            HITLInputField(
                id="response",
                question="Your response",
                input_type=HITLInputType.text,
            ),
        ],
        timeout_seconds=0,
    )

    assert node.timeout_seconds == 0


def test_hitl_input_option_requires_text():
    """Test HITL option with requires_text flag."""
    option = HITLInputOption(
        value="other",
        label="Other",
        requires_text=True,
    )

    assert option.requires_text is True


def test_hitl_input_with_default_value():
    """Test HITL input with default value."""
    field = HITLInputField(
        id="name",
        question="Your name",
        input_type=HITLInputType.text,
        default_value="John Doe",
        placeholder="Enter your name",
    )

    assert field.default_value == "John Doe"
    assert field.placeholder == "Enter your name"


# =============================================================================
# Schema Validation Error Tests
# =============================================================================


def test_hitl_single_choice_requires_options():
    """Test that single_choice input type requires options."""
    with pytest.raises(ValueError, match="requires at least 2 options"):
        HITLInputField(
            id="choice",
            question="Pick one",
            input_type=HITLInputType.single_choice,
            options=None,
        )


def test_hitl_single_choice_requires_at_least_two_options():
    """Test that single_choice requires at least 2 options."""
    with pytest.raises(ValueError, match="requires at least 2 options"):
        HITLInputField(
            id="choice",
            question="Pick one",
            input_type=HITLInputType.single_choice,
            options=[HITLInputOption(value="a", label="A")],
        )


def test_hitl_multi_choice_requires_options():
    """Test that multi_choice input type requires options."""
    with pytest.raises(ValueError, match="requires at least 2 options"):
        HITLInputField(
            id="multi",
            question="Pick many",
            input_type=HITLInputType.multi_choice,
            options=[],
        )


def test_hitl_text_should_not_have_options():
    """Test that text input type should not have options."""
    with pytest.raises(ValueError, match="should not have options"):
        HITLInputField(
            id="text",
            question="Enter text",
            input_type=HITLInputType.text,
            options=[
                HITLInputOption(value="a", label="A"),
                HITLInputOption(value="b", label="B"),
            ],
        )


def test_hitl_number_should_not_have_options():
    """Test that number input type should not have options."""
    with pytest.raises(ValueError, match="should not have options"):
        HITLInputField(
            id="num",
            question="Enter number",
            input_type=HITLInputType.number,
            options=[
                HITLInputOption(value="1", label="1"),
                HITLInputOption(value="2", label="2"),
            ],
        )


def test_hitl_boolean_should_not_have_options():
    """Test that boolean input type should not have options."""
    with pytest.raises(ValueError, match="should not have options"):
        HITLInputField(
            id="bool",
            question="Yes or no?",
            input_type=HITLInputType.boolean,
            options=[
                HITLInputOption(value="yes", label="Yes"),
                HITLInputOption(value="no", label="No"),
            ],
        )


def test_hitl_node_duplicate_input_ids():
    """Test that duplicate input IDs are rejected."""
    with pytest.raises(ValueError, match="duplicate input IDs"):
        HITLNode(
            id="dup_inputs",
            title="Duplicate Inputs",
            inputs=[
                HITLInputField(
                    id="response",
                    question="First response",
                    input_type=HITLInputType.text,
                ),
                HITLInputField(
                    id="response",  # Duplicate ID
                    question="Second response",
                    input_type=HITLInputType.text,
                ),
            ],
        )


# =============================================================================
# Type Environment Tests
# =============================================================================


def test_build_hitl_output_schema_text():
    """Test output schema for text input."""
    node = HITLNode(
        id="text_node",
        title="Text Input",
        inputs=[
            HITLInputField(
                id="comment",
                question="Your comment",
                input_type=HITLInputType.text,
                required=True,
            ),
        ],
    )

    schema = _build_hitl_output_schema(node)

    assert schema["type"] == "object"
    assert "comment" in schema["properties"]
    assert schema["properties"]["comment"]["type"] == "string"
    assert "comment" in schema["required"]


def test_build_hitl_output_schema_number():
    """Test output schema for number input."""
    node = HITLNode(
        id="number_node",
        title="Number Input",
        inputs=[
            HITLInputField(
                id="amount",
                question="Enter amount",
                input_type=HITLInputType.number,
                required=False,
            ),
        ],
    )

    schema = _build_hitl_output_schema(node)

    assert schema["properties"]["amount"]["type"] == "number"
    assert "amount" not in schema["required"]


def test_build_hitl_output_schema_boolean():
    """Test output schema for boolean input."""
    node = HITLNode(
        id="bool_node",
        title="Boolean Input",
        inputs=[
            HITLInputField(
                id="confirmed",
                question="Confirm?",
                input_type=HITLInputType.boolean,
                required=True,
            ),
        ],
    )

    schema = _build_hitl_output_schema(node)

    assert schema["properties"]["confirmed"]["type"] == "boolean"
    assert "confirmed" in schema["required"]


def test_build_hitl_output_schema_single_choice():
    """Test output schema for single_choice input."""
    node = HITLNode(
        id="choice_node",
        title="Single Choice",
        inputs=[
            HITLInputField(
                id="selection",
                question="Select one",
                input_type=HITLInputType.single_choice,
                options=[
                    HITLInputOption(value="a", label="A"),
                    HITLInputOption(value="b", label="B"),
                    HITLInputOption(value="c", label="C"),
                ],
            ),
        ],
    )

    schema = _build_hitl_output_schema(node)

    prop = schema["properties"]["selection"]
    assert prop["type"] == "string"
    assert prop["enum"] == ["a", "b", "c"]


def test_build_hitl_output_schema_multi_choice():
    """Test output schema for multi_choice input."""
    node = HITLNode(
        id="multi_node",
        title="Multi Choice",
        inputs=[
            HITLInputField(
                id="selections",
                question="Select many",
                input_type=HITLInputType.multi_choice,
                options=[
                    HITLInputOption(value="x", label="X"),
                    HITLInputOption(value="y", label="Y"),
                ],
            ),
        ],
    )

    schema = _build_hitl_output_schema(node)

    prop = schema["properties"]["selections"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"
    assert prop["items"]["enum"] == ["x", "y"]


def test_build_hitl_output_schema_mixed_inputs():
    """Test output schema with mixed input types."""
    node = HITLNode(
        id="mixed_node",
        title="Mixed Inputs",
        inputs=[
            HITLInputField(id="name", question="Name", input_type=HITLInputType.text, required=True),
            HITLInputField(id="age", question="Age", input_type=HITLInputType.number, required=True),
            HITLInputField(id="active", question="Active?", input_type=HITLInputType.boolean, required=False),
        ],
    )

    schema = _build_hitl_output_schema(node)

    assert schema["type"] == "object"
    assert len(schema["properties"]) == 3
    assert schema["required"] == ["name", "age"]
    assert schema["additionalProperties"] is False


def test_build_type_environment_with_hitl_node():
    """Test building type environment with HITL node."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            HITLNode(
                id="approval",
                title="Approval",
                inputs=[
                    HITLInputField(
                        id="decision",
                        question="Approve?",
                        input_type=HITLInputType.single_choice,
                        options=[
                            HITLInputOption(value="yes", label="Yes"),
                            HITLInputOption(value="no", label="No"),
                        ],
                    ),
                    HITLInputField(
                        id="notes",
                        question="Notes",
                        input_type=HITLInputType.text,
                        required=False,
                    ),
                ],
            ),
        ],
        edges=[],
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "approval" in symbols
    assert symbols["approval"]["type"] == "object"
    assert "decision" in symbols["approval"]["properties"]
    assert "notes" in symbols["approval"]["properties"]


# =============================================================================
# Reference Validation Tests
# =============================================================================


def test_validate_hitl_display_expressions_valid():
    """Test validation of valid display expressions."""
    env = TypeEnvironment()
    env.register("previous_node", {"type": "object", "properties": {"summary": {"type": "string"}}})
    env.register("previous_node.summary", {"type": "string"})

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[
            HITLDisplayItem(label="Summary", value="${previous_node.summary}"),
        ],
        inputs=[],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 0


def test_validate_hitl_display_expressions_invalid():
    """Test validation catches invalid display expressions."""
    env = TypeEnvironment()
    # Don't register 'unknown_node'

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[
            HITLDisplayItem(label="Data", value="${unknown_node.value}"),
        ],
        inputs=[],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 1
    assert "unknown_node" in errors[0].message


def test_validate_hitl_multiple_display_items():
    """Test validation of multiple display items."""
    env = TypeEnvironment()
    env.register("node1", {"type": "object", "properties": {"a": {"type": "string"}}})
    env.register("node1.a", {"type": "string"})
    # Don't register node2

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[
            HITLDisplayItem(label="Valid", value="${node1.a}"),
            HITLDisplayItem(label="Invalid", value="${node2.b}"),
        ],
        inputs=[],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 1
    assert errors[0].location == "display[1].value"


def test_validate_references_hitl_in_workflow():
    """Test reference validation for HITL node in full workflow."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            AgentNode(
                id="llm1",
                inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "Generate text"},
            ),
            HITLNode(
                id="review",
                title="Review",
                display=[
                    HITLDisplayItem(label="Generated Text", value="${llm1}"),
                ],
                inputs=[
                    HITLInputField(
                        id="approved",
                        question="Approve?",
                        input_type=HITLInputType.boolean,
                    ),
                ],
            ),
        ],
        edges=[
            Edge(source="llm1", target="review"),
        ],
    )

    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    # Should not raise
    validate_references(spec, env)


# =============================================================================
# WorkflowSpec Integration Tests
# =============================================================================


def test_workflow_spec_with_hitl_node():
    """Test WorkflowSpec accepts HITL node in discriminated union."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            HITLNode(
                id="hitl1",
                title="Approval",
                inputs=[
                    HITLInputField(
                        id="approve",
                        question="Approve?",
                        input_type=HITLInputType.boolean,
                    ),
                ],
            ),
        ],
        edges=[],
    )

    assert len(spec.nodes) == 1
    assert spec.nodes[0].type == "hitl"


def test_workflow_spec_mixed_nodes():
    """Test WorkflowSpec with mixed node types including HITL."""
    spec = WorkflowSpec(
        version="2",
        nodes=[
            AgentNode(
                id="llm1",
                inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "Generate"},
            ),
            HITLNode(
                id="review",
                title="Review",
                display=[
                    HITLDisplayItem(label="Output", value="${llm1}"),
                ],
                inputs=[
                    HITLInputField(
                        id="decision",
                        question="Continue?",
                        input_type=HITLInputType.single_choice,
                        options=[
                            HITLInputOption(value="yes", label="Yes"),
                            HITLInputOption(value="no", label="No"),
                        ],
                    ),
                ],
            ),
        ],
        edges=[
            Edge(source="llm1", target="review"),
        ],
    )

    assert len(spec.nodes) == 2
    assert spec.nodes[0].type == "agent"
    assert spec.nodes[1].type == "hitl"


# =============================================================================
# JSON Serialization Tests
# =============================================================================


def test_hitl_node_json_roundtrip():
    """Test HITL node serialization to JSON and back."""
    node = HITLNode(
        id="approval",
        title="Manager Approval",
        description="Review the report",
        display=[
            HITLDisplayItem(label="Total", value="${calc.total}"),
        ],
        inputs=[
            HITLInputField(
                id="decision",
                question="Approve?",
                input_type=HITLInputType.single_choice,
                options=[
                    HITLInputOption(value="approve", label="Approve"),
                    HITLInputOption(value="reject", label="Reject"),
                    HITLInputOption(value="revise", label="Revise", requires_text=True),
                ],
            ),
        ],
        timeout_seconds=86400,
    )

    # Serialize
    json_data = node.model_dump()

    # Verify structure
    assert json_data["type"] == "hitl"
    assert json_data["title"] == "Manager Approval"
    assert len(json_data["inputs"]) == 1
    assert json_data["inputs"][0]["options"][2]["requires_text"] is True

    # Deserialize
    restored = HITLNode.model_validate(json_data)
    assert restored.id == "approval"
    assert restored.inputs[0].options[2].requires_text is True


def test_hitl_node_from_workflow_json():
    """Test parsing HITL node from workflow JSON format."""
    workflow_json = {
        "version": "2",
        "nodes": [
            {
                "id": "approval_step",
                "type": "hitl",
                "title": "Manager Approval Required",
                "description": "Review the generated report before sending",
                "display": [
                    {"label": "Report Summary", "value": "${generate_report.summary}"},
                    {"label": "Total Amount", "value": "${calculate_totals.amount}"},
                ],
                "inputs": [
                    {
                        "id": "approval_decision",
                        "question": "Do you approve this report?",
                        "input_type": "single_choice",
                        "options": [
                            {"value": "approve", "label": "Approve"},
                            {"value": "reject", "label": "Reject"},
                            {"value": "revise", "label": "Request Revision", "requires_text": True},
                        ],
                        "required": True,
                    },
                    {
                        "id": "comments",
                        "question": "Additional comments (optional)",
                        "input_type": "text",
                        "required": False,
                    },
                ],
                "timeout_seconds": 86400,
                "ui": {},
            }
        ],
        "edges": [],
    }

    spec = WorkflowSpec.model_validate(workflow_json)

    assert len(spec.nodes) == 1
    node = spec.nodes[0]
    assert isinstance(node, HITLNode)
    assert node.title == "Manager Approval Required"
    assert len(node.display) == 2
    assert len(node.inputs) == 2
    assert node.inputs[0].input_type == HITLInputType.single_choice
    assert node.inputs[1].input_type == HITLInputType.text


# =============================================================================
# Input Field Evaluation Tests
# =============================================================================


def test_evaluate_input_fields_question():
    """Test evaluation of template expressions in question field."""
    inputs = [
        HITLInputField(
            id="approval",
            question="Do you approve ${order.total} for ${customer.name}?",
            input_type=HITLInputType.boolean,
        ),
    ]

    eval_ctx = EvaluationContext(
        state={
            "order": {"total": "$500"},
            "customer": {"name": "John Doe"},
        },
        locals={},
        config={},
        trigger=None,
    )

    result = _evaluate_input_fields(eval_ctx, inputs)

    assert len(result) == 1
    assert result[0]["question"] == "Do you approve $500 for John Doe?"


def test_evaluate_input_fields_placeholder():
    """Test evaluation of template expressions in placeholder field."""
    inputs = [
        HITLInputField(
            id="amount",
            question="Enter amount",
            input_type=HITLInputType.number,
            placeholder="Previous amount was ${previous.amount}",
        ),
    ]

    eval_ctx = EvaluationContext(
        state={"previous": {"amount": "100"}},
        locals={},
        config={},
        trigger=None,
    )

    result = _evaluate_input_fields(eval_ctx, inputs)

    assert result[0]["placeholder"] == "Previous amount was 100"


def test_evaluate_input_fields_default_value_string():
    """Test evaluation of template expressions in default_value (string only)."""
    inputs = [
        HITLInputField(
            id="email",
            question="Enter email",
            input_type=HITLInputType.text,
            default_value="${user.email}",
        ),
    ]

    eval_ctx = EvaluationContext(
        state={"user": {"email": "test@example.com"}},
        locals={},
        config={},
        trigger=None,
    )

    result = _evaluate_input_fields(eval_ctx, inputs)

    assert result[0]["default_value"] == "test@example.com"


def test_evaluate_input_fields_default_value_non_string():
    """Test that non-string default_value is preserved as-is."""
    inputs = [
        HITLInputField(
            id="count",
            question="Enter count",
            input_type=HITLInputType.number,
            default_value=42,
        ),
    ]

    eval_ctx = EvaluationContext(
        state={},
        locals={},
        config={},
        trigger=None,
    )

    result = _evaluate_input_fields(eval_ctx, inputs)

    assert result[0]["default_value"] == 42


def test_evaluate_input_fields_option_labels():
    """Test evaluation of template expressions in option labels."""
    inputs = [
        HITLInputField(
            id="choice",
            question="Select an option",
            input_type=HITLInputType.single_choice,
            options=[
                HITLInputOption(value="approve", label="Approve ${order.amount}"),
                HITLInputOption(value="reject", label="Reject ${order.amount}"),
            ],
        ),
    ]

    eval_ctx = EvaluationContext(
        state={"order": {"amount": "$100"}},
        locals={},
        config={},
        trigger=None,
    )

    result = _evaluate_input_fields(eval_ctx, inputs)

    assert result[0]["options"][0]["label"] == "Approve $100"
    assert result[0]["options"][1]["label"] == "Reject $100"


def test_evaluate_input_fields_error_handling():
    """Test that evaluation errors are displayed as error messages."""
    inputs = [
        HITLInputField(
            id="test",
            question="Value is ${undefined_node.value}",
            input_type=HITLInputType.text,
        ),
    ]

    eval_ctx = EvaluationContext(
        state={},  # undefined_node is not in state
        locals={},
        config={},
        trigger=None,
    )

    result = _evaluate_input_fields(eval_ctx, inputs)

    assert "<error:" in result[0]["question"]


def test_evaluate_input_fields_mixed_expressions():
    """Test evaluation of multiple fields with different expression types."""
    inputs = [
        HITLInputField(
            id="approval",
            question="Approve ${order.id}?",
            input_type=HITLInputType.single_choice,
            placeholder="Order from ${customer.name}",
            options=[
                HITLInputOption(value="yes", label="Yes, approve ${order.amount}"),
                HITLInputOption(value="no", label="No, reject order"),
            ],
        ),
    ]

    eval_ctx = EvaluationContext(
        state={
            "order": {"id": "ORD-123", "amount": "$500"},
            "customer": {"name": "Acme Corp"},
        },
        locals={},
        config={},
        trigger=None,
    )

    result = _evaluate_input_fields(eval_ctx, inputs)

    assert result[0]["question"] == "Approve ORD-123?"
    assert result[0]["placeholder"] == "Order from Acme Corp"
    assert result[0]["options"][0]["label"] == "Yes, approve $500"
    assert result[0]["options"][1]["label"] == "No, reject order"


def test_evaluate_input_fields_no_expressions():
    """Test that fields without expressions are preserved."""
    inputs = [
        HITLInputField(
            id="static",
            question="Static question without variables",
            input_type=HITLInputType.text,
            placeholder="Static placeholder",
            default_value="Static default",
        ),
    ]

    eval_ctx = EvaluationContext(
        state={},
        locals={},
        config={},
        trigger=None,
    )

    result = _evaluate_input_fields(eval_ctx, inputs)

    assert result[0]["question"] == "Static question without variables"
    assert result[0]["placeholder"] == "Static placeholder"
    assert result[0]["default_value"] == "Static default"


# =============================================================================
# Input Field Reference Validation Tests
# =============================================================================


def test_validate_hitl_input_question_valid():
    """Test validation of valid question expressions."""
    env = TypeEnvironment()
    env.register("order", {"type": "object", "properties": {"total": {"type": "string"}}})
    env.register("order.total", {"type": "string"})

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[],
        inputs=[
            HITLInputField(
                id="approval",
                question="Approve ${order.total}?",
                input_type=HITLInputType.boolean,
            ),
        ],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 0


def test_validate_hitl_input_question_invalid():
    """Test validation catches invalid question expressions."""
    env = TypeEnvironment()
    # Don't register 'unknown_node'

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[],
        inputs=[
            HITLInputField(
                id="approval",
                question="Approve ${unknown_node.value}?",
                input_type=HITLInputType.boolean,
            ),
        ],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 1
    assert errors[0].location == "inputs[0].question"
    assert "unknown_node" in errors[0].message


def test_validate_hitl_input_placeholder_invalid():
    """Test validation catches invalid placeholder expressions."""
    env = TypeEnvironment()

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[],
        inputs=[
            HITLInputField(
                id="amount",
                question="Enter amount",
                input_type=HITLInputType.number,
                placeholder="Previous: ${undefined.value}",
            ),
        ],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 1
    assert errors[0].location == "inputs[0].placeholder"


def test_validate_hitl_input_default_value_invalid():
    """Test validation catches invalid default_value expressions (string only)."""
    env = TypeEnvironment()

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[],
        inputs=[
            HITLInputField(
                id="email",
                question="Enter email",
                input_type=HITLInputType.text,
                default_value="${undefined.email}",
            ),
        ],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 1
    assert errors[0].location == "inputs[0].default_value"


def test_validate_hitl_input_option_label_invalid():
    """Test validation catches invalid option label expressions."""
    env = TypeEnvironment()
    env.register("order", {"type": "object", "properties": {"id": {"type": "string"}}})
    env.register("order.id", {"type": "string"})

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[],
        inputs=[
            HITLInputField(
                id="choice",
                question="Select option",
                input_type=HITLInputType.single_choice,
                options=[
                    HITLInputOption(value="a", label="Valid ${order.id}"),
                    HITLInputOption(value="b", label="Invalid ${undefined.value}"),
                ],
            ),
        ],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 1
    assert errors[0].location == "inputs[0].options[1].label"


def test_validate_hitl_multiple_input_errors():
    """Test validation reports errors for multiple invalid inputs."""
    env = TypeEnvironment()

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[],
        inputs=[
            HITLInputField(
                id="q1",
                question="Question ${undefined1.value}",
                input_type=HITLInputType.text,
            ),
            HITLInputField(
                id="q2",
                question="Question ${undefined2.value}",
                input_type=HITLInputType.text,
            ),
        ],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 2
    assert errors[0].location == "inputs[0].question"
    assert errors[1].location == "inputs[1].question"


def test_validate_hitl_display_and_input_errors():
    """Test validation reports errors for both display and input fields."""
    env = TypeEnvironment()

    scope = Scope(env=env)
    errors: list[NodeError] = []

    node = HITLNode(
        id="test",
        title="Test",
        display=[
            HITLDisplayItem(label="Data", value="${undefined_display.value}"),
        ],
        inputs=[
            HITLInputField(
                id="approval",
                question="Approve ${undefined_input.value}?",
                input_type=HITLInputType.boolean,
            ),
        ],
    )

    _validate_hitl(node, scope, errors)

    assert len(errors) == 2
    assert any(e.location == "display[0].value" for e in errors)
    assert any(e.location == "inputs[0].question" for e in errors)
