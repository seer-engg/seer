"""
HITLNode - Human-In-The-Loop node for collecting user input.

Pauses workflow execution using LangGraph's interrupt mechanism to collect
user responses before continuing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.core.nodes.registry import register_node_type
# Import models from schema/models.py (canonical location)
from seer.core.schema.models import HITLInputType, HITLNode

if TYPE_CHECKING:
    from seer.core.expr.evaluator import EvaluationContext
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import HITLDisplayItem, HITLInputField, HITLTableColumn, NodeBase


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


def _evaluate_table_rows(eval_ctx: "EvaluationContext", input_field: "HITLInputField") -> List[Dict]:
    """Evaluate row_data_expression and row_display_fields for a table input."""
    from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: avoid circular import
    row_data = _safe_evaluate(eval_ctx, input_field.row_data_expression)
    if not isinstance(row_data, list):
        row_data = []
    rows = []
    for row_idx, row in enumerate(row_data):
        row_display = {}
        if input_field.row_display_fields:
            row_ctx = eval_ctx.with_locals({"row": row, "row_index": row_idx})
            for item in input_field.row_display_fields:
                try:
                    row_display[item.label] = evaluate_value(row_ctx, item.value)
                except Exception:  # pylint: disable=broad-exception-caught  # Reason: Show error in display instead of failing
                    row_display[item.label] = f"<error evaluating {item.value}>"
        rows.append({"display": row_display, "row_index": row_idx, "row_data": row})
    return rows


def _input_type_to_schema(input_type: HITLInputType, options: Optional[List] = None) -> Dict:
    """Map a single HITLInputType (with optional options) to a JSON Schema fragment."""
    type_map = {
        HITLInputType.text: {"type": "string"},
        HITLInputType.number: {"type": "number"},
        HITLInputType.boolean: {"type": "boolean"},
    }
    if input_type in type_map:
        return type_map[input_type]
    enum_vals = [opt.value for opt in options] if options else None
    if input_type == HITLInputType.single_choice:
        return {"type": "string", "enum": enum_vals} if enum_vals else {"type": "string"}
    if input_type == HITLInputType.multi_choice:
        items = {"type": "string", "enum": enum_vals} if enum_vals else {"type": "string"}
        return {"type": "array", "items": items}
    return {"type": "string"}


def _build_hitl_output_schema(node: HITLNode) -> Dict:
    """Build output schema dynamically from HITL input field definitions."""
    properties: Dict = {}
    required: List[str] = []

    for field in node.inputs:
        if field.input_type == HITLInputType.table:
            col_props: Dict = {}
            col_req: List[str] = []
            for col in field.columns or []:
                col_props[col.id] = _input_type_to_schema(col.input_type, col.options)
                if col.required:
                    col_req.append(col.id)
            schema = {"type": "array", "items": {"type": "object", "properties": col_props, "required": col_req}}
        else:
            schema = _input_type_to_schema(field.input_type, field.options)
        properties[field.id] = schema
        if field.required:
            required.append(field.id)

    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


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
        eval_ctx = EvaluationContext(
            state={k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)},
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
            vars=ctx.vars,
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

        # Evaluate table input rows
        for i, input_field in enumerate(node.inputs):
            if input_field.input_type == HITLInputType.table and input_field.row_data_expression:
                evaluated_inputs[i]["rows"] = _evaluate_table_rows(eval_ctx, input_field)

        # Build interrupt payload
        interrupt_payload = {
            "type": "hitl",
            "node_id": node.id,
            "title": evaluate_value(eval_ctx, node.title) if node.title else node.title,
            "description": evaluate_value(eval_ctx, node.description) if node.description else node.description,
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
