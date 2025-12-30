"""
Stage 3 — Validate all `${...}` references against the computed type
environment.
"""

from __future__ import annotations

from typing import List

from workflow_compiler.errors import ValidationPhaseError
from workflow_compiler.expr import parser
from workflow_compiler.expr.parser import ReferenceExpr, TemplateReference
from workflow_compiler.expr.typecheck import (
    Scope,
    TypeCheckError,
    TypeEnvironment,
    ensure_references_valid,
    typecheck_reference,
)
from workflow_compiler.schema.models import ForEachNode, IfNode, Node, WorkflowSpec


def validate_references(spec: WorkflowSpec, type_env: TypeEnvironment) -> None:
    scope = Scope(env=type_env)
    errors: List[str] = []

    for node in spec.nodes:
        _validate_node(node, scope, errors)

    if spec.output is not None:
        _validate_value_references(spec.output, scope, errors, context="workflow.output")

    if errors:
        raise ValidationPhaseError("\n".join(errors))


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
        for child in node.then:
            _validate_node(child, scope, errors)
        for child in node.else_:
            _validate_node(child, scope, errors)
        return

    if isinstance(node, ForEachNode):
        _validate_for_each(node, scope, errors)
        return

    if hasattr(node, "body"):
        for child in getattr(node, "body"):
            _validate_node(child, scope, errors)


def _validate_for_each(node: ForEachNode, scope: Scope, errors: List[str]) -> None:
    try:
        ref = _single_reference(node.items)
        array_schema = typecheck_reference(ref, scope)
        items_schema = array_schema.get("items")
        if array_schema.get("type") != "array" or not isinstance(items_schema, dict):
            raise TypeCheckError("for_each items expression must resolve to an array schema")
    except (TypeCheckError, ValidationPhaseError) as exc:
        errors.append(f"{node.id}.items: {exc}")
        return

    loop_scope = scope.nested()
    loop_scope.locals[node.item_var] = items_schema
    loop_scope.locals[node.index_var] = {"type": "integer"}
    for child in node.body:
        _validate_node(child, loop_scope, errors)


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


