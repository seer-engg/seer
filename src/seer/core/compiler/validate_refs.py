"""
Stage 3 — Validate all `${...}` references against the computed type
environment.

V2: With explicit edges, nodes no longer have nested children. Validation
is simpler as all nodes are processed at the top level.
"""

from __future__ import annotations

from typing import List

from seer.core.errors import ValidationPhaseError
from seer.core.expr import parser
from seer.core.expr.parser import ReferenceExpr, TemplateReference
from seer.core.expr.typecheck import (
    Scope,
    TypeCheckError,
    TypeEnvironment,
    ensure_references_valid,
    typecheck_reference,
)
from seer.core.schema.models import EdgeType, ForEachNode, HITLNode, IfNode, Node, WorkflowSpec


def validate_references(spec: WorkflowSpec, type_env: TypeEnvironment) -> None:
    scope = Scope(env=type_env)
    errors: List[str] = []

    # Check for orphaned triggers (triggers without edges)
    _validate_orphaned_triggers(spec, errors)

    # Check if workflow uses trigger references without triggers declared
    if _uses_trigger_references(spec) and not spec.triggers:
        errors.append(
            "Workflow references trigger IDs but has no triggers declared. "
            "Add triggers to WorkflowSpec.triggers or remove trigger references."
        )

    # Check for bare "trigger" references in all workflows
    if spec.triggers and _uses_bare_trigger_reference(spec):
        trigger_ids = [t.id for t in spec.triggers]
        if len(trigger_ids) == 1:
            errors.append(
                f"Cannot use ${{trigger.X}} syntax. "
                f"Use explicit trigger ID: ${{{trigger_ids[0]}.X}}"
            )
        else:
            errors.append(
                "Cannot use ${trigger.X} syntax in multi-trigger workflow. "
                f"Use explicit trigger IDs: {', '.join(f'${{{tid}.X}}' for tid in trigger_ids)}"
            )

    for node in spec.nodes:
        _validate_node(node, scope, errors)

    if errors:
        raise ValidationPhaseError("\n".join(errors))


def _validate_orphaned_triggers(spec: WorkflowSpec, errors: List[str]) -> None:
    """Validate that every trigger has at least one edge connecting it to a node."""
    if not spec.triggers:
        return

    trigger_ids = {t.id for t in spec.triggers}
    connected_trigger_ids = {edge.source for edge in spec.edges if edge.type == EdgeType.trigger}
    orphaned_triggers = trigger_ids - connected_trigger_ids

    if orphaned_triggers:
        orphan_list = ", ".join(sorted(orphaned_triggers))
        errors.append(
            f"Orphaned triggers without edges are not allowed. "
            f"The following triggers are not connected to any node: {orphan_list}. "
            f"Add trigger edges with type='trigger' to connect these triggers to nodes."
        )


def _uses_trigger_references(spec: WorkflowSpec) -> bool:
    """Check if any node references trigger IDs."""
    if not spec.triggers:
        return False

    trigger_ids = {t.id for t in spec.triggers}

    for node in spec.nodes:
        if _node_uses_trigger_ids(node, trigger_ids):
            return True
    return False


def _uses_bare_trigger_reference(spec: WorkflowSpec) -> bool:
    """Check if any node uses bare 'trigger' reference (not trigger.id)."""
    for node in spec.nodes:
        values_to_check = []

        if hasattr(node, "inputs"):
            values_to_check.extend(getattr(node, "inputs").values())
        if hasattr(node, "value"):
            val = getattr(node, "value")
            if val is not None:
                values_to_check.append(val)
        if hasattr(node, "condition"):
            values_to_check.append(node.condition)
        if hasattr(node, "items"):
            values_to_check.append(node.items)

        refs = parser.collect_unique_references(values_to_check)
        for ref in refs:
            if ref.root == "trigger":
                return True

    return False


def _node_uses_trigger_ids(node: Node, trigger_ids: set[str]) -> bool:
    """Check if a node references any trigger IDs."""
    # Collect all values that may contain expressions
    values_to_check = []

    if hasattr(node, "inputs"):
        values_to_check.extend(getattr(node, "inputs").values())
    if hasattr(node, "value"):
        val = getattr(node, "value")
        if val is not None:
            values_to_check.append(val)
    if hasattr(node, "condition"):
        values_to_check.append(node.condition)
    if hasattr(node, "items"):
        values_to_check.append(node.items)

    # Check if any collected values reference trigger IDs
    refs = parser.collect_unique_references(values_to_check)
    for ref in refs:
        if ref.root in trigger_ids:
            return True

    return False


def _validate_node(node: Node, scope: Scope, errors: List[str]) -> None:
    if hasattr(node, "inputs"):
        _validate_value_references(
            getattr(node, "inputs"),
            scope,
            errors,
            context=f"{node.id}.inputs",
        )

    if hasattr(node, "value"):
        _validate_value_references(
            getattr(node, "value"),
            scope,
            errors,
            context=f"{node.id}.value",
        )

    if isinstance(node, IfNode):
        _validate_value_references(node.condition, scope, errors, context=f"{node.id}.condition")
        return

    if isinstance(node, ForEachNode):
        _validate_for_each(node, scope, errors)
        return

    if isinstance(node, HITLNode):
        _validate_hitl(node, scope, errors)
        return


def _validate_for_each(node: ForEachNode, scope: Scope, errors: List[str]) -> None:
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
        errors.append(f"{node.id}.items: {exc}")


def _validate_hitl(node: HITLNode, scope: Scope, errors: List[str]) -> None:
    """
    Validate display expressions in an HITLNode.

    Display items have value fields that can contain ${...} expressions.
    These expressions must reference valid symbols in scope.
    """
    for idx, display_item in enumerate(node.display):
        _validate_value_references(
            display_item.value,
            scope,
            errors,
            context=f"{node.id}.display[{idx}].value",
        )


def _single_reference(expression: str) -> ReferenceExpr:
    tokens = parser.parse_template(expression)
    if len(tokens) != 1 or not isinstance(tokens[0], TemplateReference):
        raise ValidationPhaseError("Expression must be a bare ${...} reference")
    return tokens[0].reference


def _validate_value_references(value, scope: Scope, errors: List[str], *, context: str) -> None:
    refs = parser.collect_unique_references([value])
    if not refs:
        return
    try:
        ensure_references_valid(refs, scope)
    except TypeCheckError as exc:
        errors.append(f"{context}: {exc}")
