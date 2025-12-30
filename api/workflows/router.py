from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from shared.database.models import User
from api.workflows import models as api_models
from api.workflows import services


router = APIRouter(prefix="/v1", tags=["workflows"])


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


@router.get("/builder/node-types", response_model=api_models.NodeTypeResponse)
async def get_node_types(request: Request):
    _require_user(request)
    return await services.list_node_types()


@router.get("/registries/tools", response_model=api_models.ToolRegistryResponse)
async def get_tool_registry(request: Request, include_schemas: bool = Query(False)):
    _require_user(request)
    return await services.list_tools(include_schemas=include_schemas)


@router.get("/registries/models", response_model=api_models.ModelRegistryResponse)
async def get_model_registry(request: Request):
    _require_user(request)
    return await services.list_models()


@router.get("/registries/schemas/{schema_id}", response_model=api_models.SchemaResponse)
async def get_schema(request: Request, schema_id: str):
    _require_user(request)
    return await services.resolve_schema(schema_id)


@router.post("/workflows", response_model=api_models.WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(request: Request, payload: api_models.WorkflowCreateRequest):
    user = _require_user(request)
    return await services.create_workflow(user, payload)


@router.get("/workflows", response_model=api_models.WorkflowListResponse)
async def list_workflows(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    cursor: str | None = Query(None),
):
    user = _require_user(request)
    return await services.list_workflows(user, limit=limit, cursor=cursor)


@router.get("/workflows/{workflow_id}", response_model=api_models.WorkflowResponse)
async def get_workflow(request: Request, workflow_id: str):
    user = _require_user(request)
    return await services.get_workflow(user, workflow_id)


@router.put("/workflows/{workflow_id}", response_model=api_models.WorkflowResponse)
async def update_workflow(request: Request, workflow_id: str, payload: api_models.WorkflowUpdateRequest):
    user = _require_user(request)
    return await services.update_workflow(user, workflow_id, payload)


@router.delete("/workflows/{workflow_id}", status_code=status.HTTP_200_OK)
async def delete_workflow(request: Request, workflow_id: str):
    user = _require_user(request)
    await services.delete_workflow(user, workflow_id)
    return {"ok": True}


@router.post("/workflows/validate", response_model=api_models.ValidateResponse)
async def validate_workflow(request: Request, payload: api_models.ValidateRequest):
    _require_user(request)
    return services.validate_spec(payload)


@router.post("/workflows/compile", response_model=api_models.CompileResponse)
async def compile_workflow(request: Request, payload: api_models.CompileRequest):
    user = _require_user(request)
    return services.compile_spec(user, payload)


@router.post("/expr/suggest", response_model=api_models.ExpressionSuggestResponse)
async def suggest_expression(request: Request, payload: api_models.ExpressionSuggestRequest):
    user = _require_user(request)
    return services.suggest_expression(user, payload)


@router.post("/expr/typecheck", response_model=api_models.ExpressionTypecheckResponse)
async def typecheck_expression(request: Request, payload: api_models.ExpressionTypecheckRequest):
    user = _require_user(request)
    return services.typecheck_expression(user, payload)


@router.post("/runs", response_model=api_models.RunResponse, status_code=status.HTTP_201_CREATED)
async def run_draft(request: Request, payload: api_models.RunFromSpecRequest):
    user = _require_user(request)
    return await services.run_draft_workflow(user, payload)


@router.post("/workflows/{workflow_id}/runs", response_model=api_models.RunResponse, status_code=status.HTTP_201_CREATED)
async def run_workflow(request: Request, workflow_id: str, payload: api_models.RunFromWorkflowRequest):
    user = _require_user(request)
    return await services.run_saved_workflow(user, workflow_id, payload)


@router.get("/workflows/{workflow_id}/runs", response_model=api_models.WorkflowRunListResponse)
async def list_workflow_runs(
    request: Request,
    workflow_id: str,
    limit: int = Query(50, ge=1, le=100),
):
    user = _require_user(request)
    return await services.list_workflow_runs(user, workflow_id, limit=limit)


@router.get("/runs/{run_id}", response_model=api_models.RunResponse)
async def get_run_status(request: Request, run_id: str):
    user = _require_user(request)
    return await services.get_run_status(user, run_id)


@router.get("/runs/{run_id}/result", response_model=api_models.RunResultResponse)
async def get_run_result(request: Request, run_id: str, include_state: bool = Query(False)):
    user = _require_user(request)
    return await services.get_run_result(user, run_id, include_state=include_state)



@router.get("/runs/{run_id}/history", response_model=api_models.RunHistoryResponse)
async def get_run_history(request: Request, run_id: str):
    user = _require_user(request)
    return await services.get_run_history(user, run_id)

@router.get("/runs/{run_id}/steps")
async def get_run_steps(request: Request, run_id: str):
    _require_user(request)
    await services.list_run_steps(run_id=run_id)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(request: Request, run_id: str):
    _require_user(request)
    await services.cancel_run(run_id=run_id)


__all__ = ["router"]


