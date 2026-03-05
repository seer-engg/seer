"""
Stage 3 — Validate all `${...}` references against the computed type
environment.

V2: With explicit edges, nodes no longer have nested children. Validation
is simpler as all nodes are processed at the top level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from seer.core.errors import ErrorCode, NodeError, ValidationPhaseError
from seer.core.expr import parser
from seer.core.expr.parser import ReferenceExpr, TemplateReference
from seer.core.expr.static_validator import validate_condition_expression
from seer.core.expr.typecheck import (
    Scope,
    TypeCheckError,
    TypeEnvironment,
    ensure_references_valid,
    typecheck_reference,
)
from seer.core.files.schemas import is_static_file_ref
from seer.core.schema.models import EdgeType, ForEachNode, HITLNode, IfNode, Node, WorkflowSpec

if TYPE_CHECKING:
    from seer.database import User


def validate_references(spec: WorkflowSpec, type_env: TypeEnvironment) -> None:
    scope = Scope(env=type_env)
    errors: List[NodeError] = []

    # Check for orphaned triggers (triggers without edges)
    _validate_orphaned_triggers(spec, errors)

    # Check if workflow uses trigger references without triggers declared
    if _uses_trigger_references(spec) and not spec.triggers:
        errors.append(NodeError(
            code=ErrorCode.TRIGGER_REFERENCE_ERROR,
            message="Workflow references trigger IDs but has no triggers declared. "
                    "Add triggers to WorkflowSpec.triggers or remove trigger references.",
        ))

    # Check for bare "trigger" references in all workflows
    if spec.triggers and _uses_bare_trigger_reference(spec):
        trigger_ids = [t.id for t in spec.triggers]
        if len(trigger_ids) == 1:
            errors.append(NodeError(
                code=ErrorCode.TRIGGER_REFERENCE_ERROR,
                message=f"Cannot use ${{trigger.X}} syntax. "
                        f"Use explicit trigger ID: ${{{trigger_ids[0]}.X}}",
            ))
        else:
            errors.append(NodeError(
                code=ErrorCode.TRIGGER_REFERENCE_ERROR,
                message="Cannot use ${trigger.X} syntax in multi-trigger workflow. "
                        f"Use explicit trigger IDs: {', '.join(f'${{{tid}.X}}' for tid in trigger_ids)}",
            ))

    for node in spec.nodes:
        _validate_node(node, scope, errors)

    if errors:
        raise ValidationPhaseError(_format_error_messages(errors), errors=errors)


def _format_error_messages(errors: List[NodeError]) -> str:
    """Format NodeError list into a human-readable string for the exception message."""
    lines = []
    for err in errors:
        if err.node_id and err.location:
            lines.append(f"{err.node_id}.{err.location}: {err.message}")
        elif err.node_id:
            lines.append(f"{err.node_id}: {err.message}")
        else:
            lines.append(err.message)
    return "\n".join(lines)


def _validate_orphaned_triggers(spec: WorkflowSpec, errors: List[NodeError]) -> None:
    """Validate that every trigger has at least one edge connecting it to a node."""
    if not spec.triggers:
        return

    trigger_ids = {t.id for t in spec.triggers}
    connected_trigger_ids = {edge.source for edge in spec.edges if edge.type == EdgeType.trigger}
    orphaned_triggers = trigger_ids - connected_trigger_ids

    if orphaned_triggers:
        # Add one error per orphaned trigger for better frontend mapping
        for trigger_id in sorted(orphaned_triggers):
            errors.append(NodeError(
                code=ErrorCode.ORPHANED_TRIGGER,
                message=f"Trigger '{trigger_id}' has no edges. Add a trigger edge to connect it to a node.",
                node_id=trigger_id,  # Triggers can be highlighted too
            ))


def _uses_trigger_references(spec: WorkflowSpec) -> bool:
    """Check if any node references trigger IDs."""
    if not spec.triggers:
        return False

    trigger_ids = {t.id for t in spec.triggers}

    for node in spec.nodes:
        if _node_uses_trigger_ids(node, trigger_ids):
            return True
    return False


def _collect_hitl_input_values(node: HITLNode) -> List[Any]:
    """Collect all string values from HITL input fields that may contain ${} expressions."""
    values = []
    for input_field in node.inputs:
        values.append(input_field.question)
        if input_field.placeholder is not None:
            values.append(input_field.placeholder)
        if isinstance(input_field.default_value, str):
            values.append(input_field.default_value)
        if input_field.options:
            for opt in input_field.options:
                values.append(opt.label)
    return values


def _collect_node_expression_values(node: Node) -> list:
    """Collect all expression-bearing values from a node for reference analysis."""
    values_to_check = []

    if hasattr(node, "inputs"):
        inputs = getattr(node, "inputs")
        if isinstance(inputs, dict):
            values_to_check.extend(inputs.values())
        elif isinstance(node, HITLNode):
            values_to_check.extend(_collect_hitl_input_values(node))
    if hasattr(node, "value"):
        val = getattr(node, "value")
        if val is not None:
            values_to_check.append(val)
    if hasattr(node, "condition"):
        values_to_check.append(node.condition)
    if hasattr(node, "items"):
        values_to_check.append(node.items)

    return values_to_check


def _uses_bare_trigger_reference(spec: WorkflowSpec) -> bool:
    """Check if any node uses bare 'trigger' reference (not trigger.id)."""
    for node in spec.nodes:
        refs = parser.collect_unique_references(_collect_node_expression_values(node))
        for ref in refs:
            if ref.root == "trigger":
                return True

    return False


def _node_uses_trigger_ids(node: Node, trigger_ids: set[str]) -> bool:
    """Check if a node references any trigger IDs."""
    values_to_check = _collect_node_expression_values(node)

    # Check if any collected values reference trigger IDs
    refs = parser.collect_unique_references(values_to_check)
    for ref in refs:
        if ref.root in trigger_ids:
            return True

    return False


def _validate_condition_types(
    condition: str,
    scope: Scope,
    errors: List[NodeError],
    *,
    node_id: str,
) -> None:
    """
    Perform static type checking on an IfNode condition expression.

    This catches type errors like comparing strings to numbers at compile time
    rather than waiting for runtime failures.
    """
    validation_errors = validate_condition_expression(condition, scope)
    for err_msg in validation_errors:
        errors.append(NodeError(
            code=ErrorCode.TYPE_MISMATCH,
            message=err_msg,
            node_id=node_id,
            location="condition",
            expression=condition,
        ))


def _validate_node(node: Node, scope: Scope, errors: List[NodeError]) -> None:
    if hasattr(node, "inputs"):
        _validate_value_references(
            getattr(node, "inputs"),
            scope,
            errors,
            node_id=node.id,
            location="inputs",
        )

    if hasattr(node, "value"):
        _validate_value_references(
            getattr(node, "value"),
            scope,
            errors,
            node_id=node.id,
            location="value",
        )

    if isinstance(node, IfNode):
        _validate_value_references(node.condition, scope, errors, node_id=node.id, location="condition")
        _validate_condition_types(node.condition, scope, errors, node_id=node.id)
        return

    if isinstance(node, ForEachNode):
        _validate_for_each(node, scope, errors)
        return

    if isinstance(node, HITLNode):
        _validate_hitl(node, scope, errors)
        return


def _validate_for_each(node: ForEachNode, scope: Scope, errors: List[NodeError]) -> None:
    """
    Validate the items expression for a ForEachNode.

    With edge-based control flow, loop variables are registered as global symbols
    in the type environment. Body nodes access them via ${item}, ${index}.
    """
    try:
        ref = _single_reference(node.items)
        array_schema = typecheck_reference(ref, scope)
        if array_schema.get("type") != "array":
            raise TypeCheckError("for_each items expression must resolve to an array schema")
    except (TypeCheckError, ValidationPhaseError) as exc:
        errors.append(NodeError(
            code=ErrorCode.NON_ARRAY_ITEMS if "array" in str(exc) else ErrorCode.UNDEFINED_REFERENCE,
            message=str(exc),
            node_id=node.id,
            location="items",
            expression=node.items,
        ))


def _validate_hitl(node: HITLNode, scope: Scope, errors: List[NodeError]) -> None:
    """
    Validate expressions in an HITLNode.

    Validates ${...} expressions in:
    - display[].value
    - inputs[].question
    - inputs[].placeholder (if present)
    - inputs[].default_value (if string)
    - inputs[].options[].label
    """
    # Validate display expressions
    for idx, display_item in enumerate(node.display):
        _validate_value_references(
            display_item.value,
            scope,
            errors,
            node_id=node.id,
            location=f"display[{idx}].value",
        )

    # Validate input field expressions
    for idx, input_field in enumerate(node.inputs):
        # Validate question
        _validate_value_references(
            input_field.question,
            scope,
            errors,
            node_id=node.id,
            location=f"inputs[{idx}].question",
        )

        # Validate placeholder if present
        if input_field.placeholder is not None:
            _validate_value_references(
                input_field.placeholder,
                scope,
                errors,
                node_id=node.id,
                location=f"inputs[{idx}].placeholder",
            )

        # Validate default_value only if it's a string (may contain template)
        if isinstance(input_field.default_value, str):
            _validate_value_references(
                input_field.default_value,
                scope,
                errors,
                node_id=node.id,
                location=f"inputs[{idx}].default_value",
            )

        # Validate option labels if present
        if input_field.options:
            for opt_idx, option in enumerate(input_field.options):
                _validate_value_references(
                    option.label,
                    scope,
                    errors,
                    node_id=node.id,
                    location=f"inputs[{idx}].options[{opt_idx}].label",
                )


def _single_reference(expression: str) -> ReferenceExpr:
    tokens = parser.parse_template(expression)
    if len(tokens) != 1 or not isinstance(tokens[0], TemplateReference):
        raise ValidationPhaseError("Expression must be a bare ${...} reference")
    return tokens[0].reference


def _validate_value_references(
    value: Any,
    scope: Scope,
    errors: List[NodeError],
    *,
    node_id: str,
    location: str,
) -> None:
    refs = parser.collect_unique_references([value])
    if not refs:
        return
    try:
        ensure_references_valid(refs, scope)
    except TypeCheckError as exc:
        # Try to extract the problematic expression from the error or the refs
        expression = None
        if refs:
            # Use the first reference as the expression context
            expression = refs[0].raw if hasattr(refs[0], "raw") else str(refs[0])
        errors.append(NodeError(
            code=ErrorCode.UNDEFINED_REFERENCE,
            message=str(exc),
            node_id=node_id,
            location=location,
            expression=expression,
        ))


# =============================================================================
# Static File Reference Validation
# =============================================================================


def collect_static_file_refs(spec: WorkflowSpec) -> List[tuple[str, str, str]]:
    """
    Collect all static_file_ref inputs from a workflow specification.

    Static file references are files that exist in user storage and are
    referenced at workflow design time (as opposed to dynamic references
    from parent node outputs).

    Args:
        spec: Workflow specification to scan.

    Returns:
        List of (node_id, input_key, file_id) tuples for each static file ref found.
    """
    refs: List[tuple[str, str, str]] = []

    for node in spec.nodes:
        if hasattr(node, "inputs"):
            inputs = getattr(node, "inputs")
            _scan_value_for_static_refs(inputs, node.id, "", refs)

    return refs


def _scan_value_for_static_refs(
    value: Any,
    node_id: str,
    path: str,
    refs: List[tuple[str, str, str]],
) -> None:
    """
    Recursively scan a value for static_file_ref objects.

    Static file refs can be nested in dicts or lists (e.g., in attachment arrays).
    """
    if is_static_file_ref(value):
        file_id = value.get("file_id", "")
        refs.append((node_id, path, file_id))
    elif isinstance(value, dict):
        for key, val in value.items():
            new_path = f"{path}.{key}" if path else key
            _scan_value_for_static_refs(val, node_id, new_path, refs)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            new_path = f"{path}[{i}]"
            _scan_value_for_static_refs(item, node_id, new_path, refs)


async def validate_static_file_refs(spec: WorkflowSpec, user: "User") -> List[NodeError]:
    """
    Validate that all static_file_ref inputs reference existing files.

    This should be called during workflow compilation/validation to ensure
    that static file references point to files that exist in the user's storage.

    Args:
        spec: Workflow specification to validate.
        user: User who owns the workflow (for file access control).

    Returns:
        List of NodeError objects for missing files (empty if all valid).
    """
    # pylint: disable=import-outside-toplevel  # Avoid circular imports with database models
    from seer.database import WorkflowFile

    refs = collect_static_file_refs(spec)
    if not refs:
        return []

    errors: List[NodeError] = []

    # Batch lookup all file_ids for efficiency
    file_ids = {ref[2] for ref in refs}
    existing_files = await WorkflowFile.filter(file_id__in=file_ids, user=user).values_list("file_id", flat=True)
    existing_file_ids = set(existing_files)

    for node_id, input_key, file_id in refs:
        if file_id not in existing_file_ids:
            location = f"inputs.{input_key}" if input_key else "inputs"
            errors.append(NodeError(
                code=ErrorCode.FILE_NOT_FOUND,
                message=f"Static file '{file_id}' not found in user's storage. "
                        "Ensure the file exists or select a different file.",
                node_id=node_id,
                location=location,
            ))

    return errors
