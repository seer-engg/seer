from __future__ import annotations

from typing import Union

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from seer.api.workflows import models as api_models
from seer.api.workflows import services
from seer.database import User

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


@router.get("/triggers", response_model=api_models.TriggerCatalogResponse)
async def get_trigger_catalog(request: Request):
    _require_user(request)
    return await services.list_triggers()


@router.post(
    "/trigger-subscriptions/{subscription_id}/test",
    response_model=api_models.TriggerSubscriptionTestResponse,
)
async def test_trigger_subscription(
    request: Request,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionTestRequest,
):
    user = _require_user(request)
    return await services.test_trigger_subscription(user, subscription_id, payload)


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


@router.post("/schemas/generate-metadata", response_model=api_models.SchemaMetadataGenerateResponse)
async def generate_schema_metadata(
    request: Request,
    payload: api_models.SchemaMetadataGenerateRequest,
):
    """Generate schema title and description using LLM analysis."""
    _require_user(request)
    return await services.generate_schema_metadata(payload)


@router.post("/workflows", response_model=api_models.WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(request: Request, payload: api_models.WorkflowCreateRequest):
    user = _require_user(request)
    return await services.create_workflow(user, payload)


@router.post("/workflows/import", response_model=api_models.WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def import_workflow(
    request: Request,
    payload: api_models.WorkflowImportRequest,
):
    user = _require_user(request)
    return await services.import_workflow(user, payload)


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


@router.get("/workflows/{workflow_id}/export")
async def export_workflow(
    request: Request,
    workflow_id: str,
    include_triggers: bool = Query(True),
):
    user = _require_user(request)
    export_data = await services.export_workflow(user, workflow_id, include_triggers)

    # Return as downloadable JSON file
    workflow = await services.get_workflow(user, workflow_id)
    filename = f"{workflow.name.replace(' ', '_')}.seer.json"

    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get("/workflows/{workflow_id}/versions", response_model=api_models.WorkflowVersionListResponse)
async def list_workflow_versions(request: Request, workflow_id: str):
    user = _require_user(request)
    return await services.list_workflow_versions(user, workflow_id)


@router.post(
    "/workflows/{workflow_id}/versions/{version_id}/restore",
    response_model=api_models.WorkflowResponse,
)
async def restore_workflow_version(
    request: Request,
    workflow_id: str,
    version_id: int,
    payload: api_models.WorkflowVersionRestoreRequest,
):
    user = _require_user(request)
    return await services.restore_workflow_version(user, workflow_id, version_id, payload)


@router.put("/workflows/{workflow_id}", response_model=api_models.WorkflowResponse)
async def update_workflow(request: Request, workflow_id: str, payload: api_models.WorkflowUpdateRequest):
    user = _require_user(request)
    return await services.update_workflow(user, workflow_id, payload)


@router.patch("/workflows/{workflow_id}/draft", response_model=api_models.WorkflowResponse)
async def patch_workflow_draft(
    request: Request,
    workflow_id: str,
    payload: api_models.WorkflowDraftPatchRequest,
):
    user = _require_user(request)
    return await services.patch_workflow_draft(user, workflow_id, payload)


@router.post("/workflows/{workflow_id}/publish", response_model=api_models.WorkflowResponse)
async def publish_workflow(
    request: Request,
    workflow_id: str,
    payload: api_models.WorkflowPublishRequest = Body(default_factory=api_models.WorkflowPublishRequest),
):
    user = _require_user(request)
    return await services.publish_workflow(user, workflow_id, payload)


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


@router.post("/expr/typecheck", response_model=api_models.ExpressionTypecheckResponse)
async def typecheck_expression(request: Request, payload: api_models.ExpressionTypecheckRequest):
    user = _require_user(request)
    return services.typecheck_expression(user, payload)


@router.post(
    "/workflows/{workflow_id}/runs",
    response_model=Union[api_models.RunResponse, api_models.MultiRunResponse],
    status_code=status.HTTP_201_CREATED
)
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


@router.get("/runs/{run_id}/history", response_model=api_models.RunHistoryResponse)
async def get_run_history(request: Request, run_id: str):
    user = _require_user(request)
    return await services.get_run_history(user, run_id)


__all__ = ["router"]
