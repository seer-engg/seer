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
from seer.core.schema.models import ForEachNode, IfNode, Node, WorkflowSpec


def validate_references(spec: WorkflowSpec, type_env: TypeEnvironment) -> None:
    scope = Scope(env=type_env)
    errors: List[str] = []

    # Check if workflow uses trigger references without triggers declared
    if _uses_trigger_references(spec) and not spec.triggers:
        errors.append(
            "Workflow uses trigger references but has no triggers declared. "
            "Add triggers to WorkflowSpec.triggers or remove trigger references."
        )

    for node in spec.nodes:
        _validate_node(node, scope, errors)

    if errors:
        raise ValidationPhaseError("\n".join(errors))


def _uses_trigger_references(spec: WorkflowSpec) -> bool:
    """Check if any node references trigger titles."""
    if not spec.triggers:
        return False

    trigger_titles = {t.title for t in spec.triggers}

    for node in spec.nodes:
        if _node_uses_trigger_titles(node, trigger_titles):
            return True
    return False


def _node_uses_trigger_titles(node: Node, trigger_titles: set[str]) -> bool:
    """Check if a node references any trigger titles."""
    # Collect all values that may contain expressions
    values_to_check = []

    if hasattr(node, "in_"):
        values_to_check.extend(getattr(node, "in_").values())
    if hasattr(node, "value"):
        val = getattr(node, "value")
        if val is not None:
            values_to_check.append(val)
    if hasattr(node, "prompt"):
        values_to_check.append(node.prompt)
    if hasattr(node, "condition"):
        values_to_check.append(node.condition)
    if hasattr(node, "items"):
        values_to_check.append(node.items)

    # Check if any collected values reference trigger titles
    refs = parser.collect_unique_references(values_to_check)
    for ref in refs:
        if ref.root in trigger_titles:
            return True

    return False


def _validate_node(node: Node, scope: Scope, errors: List[str]) -> None:
    if hasattr(node, "in_"):
        _validate_value_references(
            getattr(node, "in_"),
            scope,
            errors,
            context=f"{node.id}.in",
        )

    if hasattr(node, "value"):
        _validate_value_references(
            getattr(node, "value"),
            scope,
            errors,
            context=f"{node.id}.value",
        )

    if hasattr(node, "prompt"):
        _validate_value_references(node.prompt, scope, errors, context=f"{node.id}.prompt")

    if isinstance(node, IfNode):
        _validate_value_references(node.condition, scope, errors, context=f"{node.id}.condition")
        return

    if isinstance(node, ForEachNode):
        _validate_for_each(node, scope, errors)
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
