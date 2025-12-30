"""
Runtime evaluation helpers for `${...}` expressions embedded in workflow
definitions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping

from workflow_compiler.expr import parser
from workflow_compiler.expr.parser import IndexSegment, PropertySegment, ReferenceExpr, TemplateLiteral, TemplateReference
from workflow_compiler.schema.models import JSONValue


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationContext:
    state: Mapping[str, Any]
    inputs: Mapping[str, Any]
    locals: Mapping[str, Any]
    config: Mapping[str, Any] | None = None

    def with_locals(self, overrides: Mapping[str, Any]) -> "EvaluationContext":
        merged = dict(self.locals)
        merged.update(overrides)
        return replace(self, locals=merged)


def resolve_reference(ctx: EvaluationContext, reference: ReferenceExpr) -> Any:
    value: Any | None = None
    root = reference.root

    if root in ctx.locals:
        value = ctx.locals[root]
    elif root in ctx.state:
        value = ctx.state[root]
    elif root == "inputs":
        value = ctx.inputs
    elif ctx.config and root == "config":
        value = ctx.config
    elif ctx.config and root in ctx.config:
        value = ctx.config[root]
    else:
        raise EvaluationError(f"Unknown reference root '{root}'")

    for segment in reference.segments:
        if isinstance(segment, PropertySegment):
            if not isinstance(value, Mapping):
                raise EvaluationError(
                    f"Cannot access property '{segment.key}' on non-object value in '{reference.raw}'"
                )
            if segment.key not in value:
                raise EvaluationError(
                    f"Property '{segment.key}' not found while resolving '{reference.raw}'"
                )
            value = value[segment.key]
        elif isinstance(segment, IndexSegment):
            if isinstance(segment.index, int):
                if not isinstance(value, (list, tuple)):
                    raise EvaluationError(
                        f"Cannot use numeric index on non-list value in '{reference.raw}'"
                    )
                try:
                    value = value[segment.index]
                except IndexError as exc:
                    raise EvaluationError(
                        f"Index {segment.index} out of range in '{reference.raw}'"
                    ) from exc
            else:
                if not isinstance(value, Mapping):
                    raise EvaluationError(
                        f"Cannot use string index on non-object value in '{reference.raw}'"
                    )
                if segment.index not in value:
                    raise EvaluationError(
                        f"Key '{segment.index}' not found while resolving '{reference.raw}'"
                    )
                value = value[segment.index]
        else:
            raise EvaluationError(f"Unsupported segment type {type(segment)!r}")
    return value


def evaluate_value(ctx: EvaluationContext, value: JSONValue) -> Any:
    if isinstance(value, str):
        return render_template(ctx, value)
    if isinstance(value, list):
        return [evaluate_value(ctx, item) for item in value]
    if isinstance(value, dict):
        return {key: evaluate_value(ctx, item) for key, item in value.items()}
    return value


def render_template(ctx: EvaluationContext, text: str) -> Any:
    tokens = parser.parse_template(text)
    if len(tokens) == 1 and isinstance(tokens[0], TemplateReference):
        return resolve_reference(ctx, tokens[0].reference)

    pieces: list[str] = []
    for token in tokens:
        if isinstance(token, TemplateLiteral):
            pieces.append(token.text)
        else:
            value = resolve_reference(ctx, token.reference)
            pieces.append("" if value is None else str(value))
    return "".join(pieces)


SAFE_FUNCTIONS: Dict[str, Any] = {
    "len": len,
    "any": any,
    "all": all,
    "min": min,
    "max": max,
    "sum": sum,
    "str": str,
    "int": int,
    "float": float,
}


class _ExpressionValidator(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Subscript,
        ast.Slice,
        ast.List,
        ast.Tuple,
    )

    ALLOWED_BINOPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
    )

    ALLOWED_UNARY = (ast.Not, ast.USub, ast.UAdd)

    ALLOWED_CMPS = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.Gt,
        ast.LtE,
        ast.GtE,
        ast.In,
        ast.NotIn,
    )

    def __init__(self, allowed_names: Iterable[str]) -> None:
        self.allowed_names = set(allowed_names)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.cmpop, ast.operator, ast.boolop, ast.unaryop)):
            return
        if not isinstance(node, self.ALLOWED_NODES):
            raise EvaluationError(f"Disallowed expression node: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise EvaluationError("Only whitelisted helper functions can be used in conditions")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names and node.id not in SAFE_FUNCTIONS:
            raise EvaluationError(f"Unknown variable '{node.id}' in expression")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self.ALLOWED_BINOPS):
            raise EvaluationError(f"Operator '{type(node.op).__name__}' is not allowed")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self.ALLOWED_UNARY):
            raise EvaluationError(f"Unary op '{type(node.op).__name__}' is not allowed")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if not isinstance(op, self.ALLOWED_CMPS):
                raise EvaluationError(f"Comparator '{type(op).__name__}' is not allowed")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not isinstance(node.value, ast.Name):
            raise EvaluationError("Only placeholder variables can be subscripted")
        if node.value.id not in self.allowed_names:
            raise EvaluationError("Subscripted value is not a placeholder")
        self.generic_visit(node)


def evaluate_condition(ctx: EvaluationContext, expression: str) -> bool:
    tokens = parser.parse_template(expression)
    rendered_expr: list[str] = []
    bindings: Dict[str, Any] = {}

    for token in tokens:
        if isinstance(token, TemplateLiteral):
            rendered_expr.append(token.text)
        else:
            placeholder = f"__ref_{len(bindings)}"
            bindings[placeholder] = resolve_reference(ctx, token.reference)
            rendered_expr.append(placeholder)

    expr_str = "".join(rendered_expr).strip()
    if not expr_str:
        raise EvaluationError("Condition expression resolved to empty string")

    tree = ast.parse(expr_str, mode="eval")
    validator = _ExpressionValidator(bindings.keys())
    validator.visit(tree)
    compiled = compile(tree, "<condition>", "eval")

    safe_locals = dict(SAFE_FUNCTIONS)
    safe_locals.update(bindings)
    result = eval(compiled, {"__builtins__": {}}, safe_locals)
    return bool(result)


