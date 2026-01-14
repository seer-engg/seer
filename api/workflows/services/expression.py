"""Expression parsing, typechecking, and autocomplete support."""

from __future__ import annotations

from api.agents.checkpointer import get_checkpointer
from api.workflows import models as api_models
from api.workflows.services.shared import (
    VALIDATION_PROBLEM,
    _raise_problem,
    _spec_to_dict,
)
from shared.database import User
from workflow_compiler.errors import ValidationPhaseError
from workflow_compiler.expr import parser as expr_parser
from workflow_compiler.expr.typecheck import Scope, TypeEnvironment, typecheck_reference
from workflow_compiler.runtime.global_compiler import WorkflowCompilerSingleton
from workflow_compiler.schema.models import WorkflowSpec

compiler = WorkflowCompilerSingleton.instance()


def _type_env_from_compiled(compiled) -> TypeEnvironment:
    return compiled.workflow.runtime.services.type_env


async def _prepare_type_env(user: User, spec: WorkflowSpec) -> TypeEnvironment:
    checkpointer = await get_checkpointer()
    compiled = await compiler.compile(
        user,
        _spec_to_dict(spec),
        checkpointer=checkpointer,
    )
    return _type_env_from_compiled(compiled)


def typecheck_expression(user: User, payload: api_models.ExpressionTypecheckRequest) -> api_models.ExpressionTypecheckResponse:
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
