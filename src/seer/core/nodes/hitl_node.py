"""
HITLNode - Human-In-The-Loop node for collecting user input.

Pauses workflow execution using LangGraph's interrupt mechanism to collect
user responses before continuing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.core.nodes.registry import register_node_type
# Import models from schema/models.py (canonical location)
from seer.core.schema.models import HITLInputType, HITLNode

if TYPE_CHECKING:
    from seer.core.expr.evaluator import EvaluationContext
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import HITLInputField, NodeBase


# =============================================================================
# Helper Functions
# =============================================================================


def _safe_evaluate(eval_ctx: "EvaluationContext", value: str) -> Any:
    """Evaluate an expression, returning an error string on failure."""
    from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    try:
        return evaluate_value(eval_ctx, value)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: Show error instead of failing
        return f"<error: {exc}>"


def _evaluate_field_options(eval_ctx: "EvaluationContext", field: "HITLInputField") -> List[Dict]:
    """Evaluate option labels for a HITL input field."""
    evaluated_options = []
    for opt in field.options or []:
        opt_dict = opt.model_dump()
        opt_dict["label"] = _safe_evaluate(eval_ctx, opt.label)
        evaluated_options.append(opt_dict)
    return evaluated_options


def _evaluate_single_field(eval_ctx: "EvaluationContext", field: "HITLInputField") -> Dict:
    """Evaluate all expression fields within a single HITLInputField."""
    field_dict = field.model_dump()
    field_dict["question"] = _safe_evaluate(eval_ctx, field.question)
    if field.placeholder is not None:
        field_dict["placeholder"] = _safe_evaluate(eval_ctx, field.placeholder)
    if isinstance(field.default_value, str):
        field_dict["default_value"] = _safe_evaluate(eval_ctx, field.default_value)
    if field.options:
        field_dict["options"] = _evaluate_field_options(eval_ctx, field)
    return field_dict


def _evaluate_input_fields(eval_ctx: "EvaluationContext", inputs: List["HITLInputField"]) -> List[Dict]:
    """
    Evaluate template expressions in HITL input fields.

    Evaluates expressions in:
    - question: str
    - placeholder: Optional[str]
    - default_value: Optional[JSONValue] (only when string)
    - options[].label: str

    Args:
        eval_ctx: Evaluation context with state, locals, config, and trigger.
        inputs: List of HITLInputField objects to evaluate.

    Returns:
        List of dicts with evaluated field values.
    """
    return [_evaluate_single_field(eval_ctx, field) for field in inputs]


def _build_hitl_output_schema(node: HITLNode) -> Dict:
    """
    Build output schema dynamically from HITL input field definitions.

    Each input field becomes a property in the output schema with the appropriate type.
    """
    properties: Dict = {}
    required: List[str] = []

    for input_field in node.inputs:
        field_schema: Dict = {}

        if input_field.input_type == HITLInputType.text:
            field_schema = {"type": "string"}
        elif input_field.input_type == HITLInputType.number:
            field_schema = {"type": "number"}
        elif input_field.input_type == HITLInputType.boolean:
            field_schema = {"type": "boolean"}
        elif input_field.input_type == HITLInputType.single_choice:
            if input_field.options:
                field_schema = {
                    "type": "string",
                    "enum": [opt.value for opt in input_field.options],
                }
            else:
                field_schema = {"type": "string"}
        elif input_field.input_type == HITLInputType.multi_choice:
            if input_field.options:
                field_schema = {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [opt.value for opt in input_field.options],
                    },
                }
            else:
                field_schema = {"type": "array", "items": {"type": "string"}}

        properties[input_field.id] = field_schema
        if input_field.required:
            required.append(input_field.id)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# =============================================================================
# Node Type Implementation
# =============================================================================

class HITLNodeType(BaseNodeType):
    """Implementation of the HITL (Human-In-The-Loop) node type."""

    @property
    def type_literal(self) -> str:
        return "hitl"

    @property
    def model_class(self) -> type["NodeBase"]:
        return HITLNode

    async def execute_async(
        self,
        node: HITLNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """
        Execute HITL node - pause workflow for user input collection.

        Uses LangGraph's interrupt() to pause execution. The workflow resumes
        when user provides responses via the resume API.
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from langgraph.types import interrupt
        from seer.core.expr.evaluator import EvaluationContext, evaluate_value
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX

        # Build eval context
        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        eval_ctx = EvaluationContext(
            state=visible_state,
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
        )

        # Evaluate display expressions
        display_data = []
        for item in node.display:
            try:
                evaluated_value = evaluate_value(eval_ctx, item.value)
            except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: Show error in display instead of failing
                evaluated_value = f"<error: {exc}>"
            display_data.append({
                "label": item.label,
                "value": evaluated_value,
            })

        # Evaluate input field expressions (question, placeholder, default_value, option labels)
        evaluated_inputs = _evaluate_input_fields(eval_ctx, node.inputs)

        # Build interrupt payload
        interrupt_payload = {
            "type": "hitl",
            "node_id": node.id,
            "title": node.title,
            "description": node.description,
            "display": display_data,
            "inputs": evaluated_inputs,
            "timeout_seconds": node.timeout_seconds,
            "delivery_channels": [ch.model_dump() for ch in node.delivery_channels],
        }

        # Trigger interrupt - execution pauses here until resumed
        user_responses = interrupt(interrupt_payload)

        # Build output from user responses
        output: Dict[str, Any] = {node.id: user_responses or {}}

        # Store trace data (loop-aware key for nested loop support)
        trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
        output[trace_key] = {
            "node_id": node.id,
            "node_type": "hitl",
            "title": node.title,
            "display": display_data,
            "inputs": evaluated_inputs,
            "output": user_responses,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output

    def register_type_sync(
        self,
        node: HITLNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """Register HITL node's dynamically-built output schema."""
        schema = _build_hitl_output_schema(node)
        if node.id:
            env.register(node.id, schema)


# Auto-register on module import
register_node_type(HITLNodeType())
