"""Expression parsing, typechecking, and autocomplete support."""

from __future__ import annotations

from typing import List

from api.workflows import models as api_models
from api.workflows.services._shared import VALIDATION_PROBLEM, _raise_problem, _spec_to_dict
from api.agents.checkpointer import get_checkpointer
from shared.database.models import User
from workflow_compiler.errors import ValidationPhaseError
from workflow_compiler.expr import parser as expr_parser
from workflow_compiler.expr.typecheck import Scope, TypeEnvironment, typecheck_reference
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.schema.models import WorkflowSpec

compiler = WorkflowCompilerSingleton.instance()


def _type_env_from_compiled(compiled) -> TypeEnvironment:
    """Extract type environment from compiled workflow."""
    return compiled.workflow.runtime.services.type_env


async def _prepare_type_env(user: User, spec: WorkflowSpec) -> TypeEnvironment:
    """
    Prepare type environment for expression validation.

    Args:
        user: User context
        spec: Workflow specification

    Returns:
        Type environment for typechecking
    """
    checkpointer = await get_checkpointer()
    compiled = await compiler.compile(
        user,
        _spec_to_dict(spec),
        checkpointer=checkpointer,
    )
    return _type_env_from_compiled(compiled)


def suggest_expression(user: User, payload: api_models.ExpressionSuggestRequest) -> api_models.ExpressionSuggestResponse:
    """
    Provide autocomplete suggestions for expression references.

    Args:
        user: User context
        payload: Expression autocomplete request

    Returns:
        List of autocomplete suggestions
    """
    type_env = _prepare_type_env(user, payload.spec)
    prefix = payload.cursor_context.prefix.strip()
    if not prefix.startswith("${"):
        return api_models.ExpressionSuggestResponse()
    content = prefix[2:]
    if content.endswith("}"):
        content = content[:-1]
    if "." in content:
        base, partial = content.rsplit(".", 1)
    else:
        base, partial = content, ""
    if not base:
        return api_models.ExpressionSuggestResponse()
    try:
        reference = expr_parser.parse_reference_string(base)
        schema = typecheck_reference(reference, Scope(env=type_env))
    except Exception:
        return api_models.ExpressionSuggestResponse()

    props = schema.get("properties", {})
    suggestions: List[api_models.ExpressionSuggestion] = []
    for key, value in props.items():
        if partial and not key.startswith(partial):
            continue
        type_name = value.get("type") if isinstance(value, dict) else None
        suggestions.append(
            api_models.ExpressionSuggestion(label=key, insert=key, type=type_name)
        )
    return api_models.ExpressionSuggestResponse(suggestions=suggestions)


def typecheck_expression(user: User, payload: api_models.ExpressionTypecheckRequest) -> api_models.ExpressionTypecheckResponse:
    """
    Typecheck an expression reference.

    Args:
        user: User context
        payload: Expression typecheck request

    Returns:
        Type schema for the expression

    Raises:
        HTTPException: If expression is invalid or typechecking fails
    """
    expression = payload.expression.strip()
    if not (expression.startswith("${") and expression.endswith("}")):
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid expression",
            detail="Expression must be a ${...} reference",
            status=400,
        )
    content = expression[2:-1]
    try:
        reference = expr_parser.parse_reference_string(content)
    except ValueError as exc:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Invalid expression",
            detail=str(exc),
            status=400,
        )

    type_env = _prepare_type_env(user, payload.spec)
    try:
        schema = typecheck_reference(reference, Scope(env=type_env))
    except ValidationPhaseError as exc:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Expression validation failed",
            detail=str(exc),
            status=400,
        )
    except Exception as exc:
        _raise_problem(
            type_uri=VALIDATION_PROBLEM,
            title="Expression validation failed",
            detail=str(exc),
            status=400,
        )
    return api_models.ExpressionTypecheckResponse(type=schema)
