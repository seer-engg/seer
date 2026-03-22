from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from tortoise.exceptions import IntegrityError

from seer.api.core.middleware.organization import get_membership, get_organization
from seer.api.workflows import models as api_models
from seer.api.workflows import services
from seer.api.workflows.services.share_template import (
    GenerateTemplateDescriptionResponse,
    ShareAsTemplateRequest,
    ShareAsTemplateResponse,
    generate_template_description,
    get_workflow_template_meta,
    share_workflow_as_template,
)
from seer.database import Organization, OrganizationMembership, User
from seer.database.workflow_models import GlobalVariable

router = APIRouter(prefix="/v1", tags=["workflows"])


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


def _get_org_context(request: Request) -> tuple[Optional[Organization], Optional[OrganizationMembership]]:
    """Get optional organization context from request state."""
    try:
        org = get_organization(request)
        membership = get_membership(request)
        return org, membership
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: fallback to None if org context not available
        return None, None


@router.get("/builder/node-types", response_model=api_models.NodeTypeResponse)
async def get_node_types(request: Request):
    _require_user(request)
    return await services.list_node_types()


@router.get("/triggers", response_model=api_models.TriggerCatalogResponse)
async def get_trigger_catalog(request: Request):
    user = _require_user(request)
    return await services.list_triggers(user)


@router.get("/triggers/{trigger_key}/accounts", response_model=api_models.TriggerAccountsResponse)
async def get_trigger_accounts(request: Request, trigger_key: str):
    """Get available OAuth accounts for a specific trigger."""
    user = _require_user(request)
    return await services.get_trigger_accounts(user, trigger_key)


@router.get("/trigger-subscriptions", response_model=api_models.TriggerSubscriptionListItemsResponse)
async def list_trigger_subscriptions(
    request: Request,
    workflow_id: str | None = Query(None),
    trigger_key: str | None = Query(None),
    search: str | None = Query(None),
):
    """List all trigger subscriptions with optional filtering."""
    user = _require_user(request)
    return await services.list_trigger_subscriptions_extended(
        user, workflow_id=workflow_id, trigger_key=trigger_key, search=search
    )


@router.patch(
    "/trigger-subscriptions/{subscription_id}",
    response_model=api_models.TriggerSubscriptionResponse,
)
async def toggle_trigger_subscription(
    request: Request,
    subscription_id: int,
    payload: api_models.TriggerSubscriptionToggleRequest,
):
    """Toggle the enabled status of a trigger subscription."""
    user = _require_user(request)
    update_payload = api_models.TriggerSubscriptionUpdateRequest(enabled=payload.enabled)
    return await services.update_trigger_subscription(user, subscription_id, update_payload)


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


@router.post(
    "/workflows/{workflow_id}/triggers/{trigger_id}/start-listening",
    response_model=api_models.StartListeningResponse,
)
async def start_listening(
    request: Request,
    workflow_id: str,
    trigger_id: str,
):
    user = _require_user(request)
    return await services.start_listening_for_trigger(user, workflow_id, trigger_id)


@router.get(
    "/workflows/{workflow_id}/triggers/{trigger_id}/pending-events",
    response_model=api_models.PendingEventsResponse,
)
async def get_pending_events(
    request: Request,
    workflow_id: str,
    trigger_id: str,
    since: int | None = Query(None),
):
    user = _require_user(request)
    return await services.get_pending_events(user, workflow_id, trigger_id, since=since)


@router.get(
    "/subscriptions/{subscription_id}/event-count",
    response_model=api_models.SubscriptionEventCountResponse,
)
async def get_subscription_event_count(
    request: Request,
    subscription_id: int,
):
    """Get the count of stored events for a trigger subscription."""
    user = _require_user(request)
    return await services.get_subscription_event_count(user, subscription_id)


@router.post(
    "/triggers/{trigger_key}/generate-event",
    response_model=api_models.TriggerEventGenerateResponse,
)
async def generate_trigger_event(
    request: Request,
    trigger_key: str,
    payload: api_models.TriggerEventGenerateRequest,
):
    """Generate a synthetic trigger event for immediate workflow execution."""
    _require_user(request)

    if trigger_key != "schedule.cron":
        raise HTTPException(
            status_code=400,
            detail=f"Trigger '{trigger_key}' does not support instant triggering"
        )

    envelope = services.generate_cron_event(payload.trigger_id, payload.provider_config)
    return api_models.TriggerEventGenerateResponse(
        envelope=envelope,
        display_title="Manual Trigger"
    )


@router.get("/registries/tools", response_model=api_models.ToolRegistryResponse)
async def get_tool_registry(request: Request, include_schemas: bool = Query(False)):
    _require_user(request)
    return await services.list_tools(include_schemas=include_schemas)


@router.get("/registries/models", response_model=api_models.ModelRegistryResponse)
async def get_model_registry(request: Request):
    _require_user(request)
    return await services.list_models()


@router.get("/registries/browser-models", response_model=api_models.ModelRegistryResponse)
async def get_browser_model_registry(request: Request):
    _require_user(request)
    return await services.list_browser_models()


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


@router.post("/mcp/tools", response_model=api_models.McpToolsResponse)
async def list_mcp_tools(request: Request, payload: api_models.McpToolsRequest):
    _require_user(request)
    return await services.list_mcp_tools(payload)


@router.post("/workflows", response_model=api_models.WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(request: Request, payload: api_models.WorkflowCreateRequest):
    user = _require_user(request)
    org, _ = _get_org_context(request)
    return await services.create_workflow(user, payload, organization=org)


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
    org, membership = _get_org_context(request)
    return await services.list_workflows(user, limit=limit, cursor=cursor, organization=org, membership=membership)


@router.get("/workflows/{workflow_id}", response_model=api_models.WorkflowResponse)
async def get_workflow(request: Request, workflow_id: str):
    user = _require_user(request)
    org, membership = _get_org_context(request)
    return await services.get_workflow(user, workflow_id, organization=org, membership=membership)


@router.get("/workflows/{workflow_id}/export")
async def export_workflow(
    request: Request,
    workflow_id: str,
    include_triggers: bool = Query(True),
):
    user = _require_user(request)
    org, membership = _get_org_context(request)
    export_data = await services.export_workflow(user, workflow_id, include_triggers, organization=org, membership=membership)

    # Return as downloadable JSON file
    workflow = await services.get_workflow(user, workflow_id, organization=org, membership=membership)
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
    org, membership = _get_org_context(request)
    return await services.list_workflow_versions(user, workflow_id, organization=org, membership=membership)


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
    org, membership = _get_org_context(request)
    return await services.restore_workflow_version(user, workflow_id, version_id, payload, organization=org, membership=membership)


@router.put("/workflows/{workflow_id}", response_model=api_models.WorkflowResponse)
async def update_workflow(request: Request, workflow_id: str, payload: api_models.WorkflowUpdateRequest):
    user = _require_user(request)
    org, membership = _get_org_context(request)
    return await services.update_workflow(user, workflow_id, payload, organization=org, membership=membership)


@router.patch("/workflows/{workflow_id}/draft", response_model=api_models.WorkflowResponse)
async def patch_workflow_draft(
    request: Request,
    workflow_id: str,
    payload: api_models.WorkflowDraftPatchRequest,
):
    user = _require_user(request)
    org, membership = _get_org_context(request)
    return await services.patch_workflow_draft(user, workflow_id, payload, organization=org, membership=membership)


@router.post("/workflows/{workflow_id}/publish", response_model=api_models.WorkflowResponse)
async def publish_workflow(
    request: Request,
    workflow_id: str,
    payload: api_models.WorkflowPublishRequest = Body(default_factory=api_models.WorkflowPublishRequest),
):
    user = _require_user(request)
    org, membership = _get_org_context(request)
    return await services.publish_workflow(user, workflow_id, payload, organization=org, membership=membership)


@router.patch("/workflows/{workflow_id}/active", response_model=api_models.WorkflowSummary)
async def toggle_workflow_active(
    request: Request,
    workflow_id: str,
    payload: api_models.WorkflowActiveToggleRequest,
):
    """Toggle whether a workflow is active or paused."""
    user = _require_user(request)
    org, membership = _get_org_context(request)
    return await services.toggle_workflow_active(user, workflow_id, payload.is_active, organization=org, membership=membership)


@router.delete("/workflows/{workflow_id}", status_code=status.HTTP_200_OK)
async def delete_workflow(request: Request, workflow_id: str):
    user = _require_user(request)
    org, membership = _get_org_context(request)
    try:
        await asyncio.wait_for(
            services.delete_workflow(user, workflow_id, organization=org, membership=membership),
            timeout=30.0,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Delete operation timed out. The workflow may have active runs.") from exc
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
    response_model=api_models.RunResponse,
    status_code=status.HTTP_201_CREATED
)
async def run_workflow(request: Request, workflow_id: str, payload: api_models.RunFromWorkflowRequest):
    user = _require_user(request)
    org, membership = _get_org_context(request)
    return await services.run_saved_workflow(user, workflow_id, payload, organization=org, membership=membership)


@router.get("/workflows/{workflow_id}/runs", response_model=api_models.WorkflowRunListResponse)
async def list_workflow_runs(
    request: Request,
    workflow_id: str,
    limit: int = Query(50, ge=1, le=100),
):
    user = _require_user(request)
    org, membership = _get_org_context(request)
    return await services.list_workflow_runs(user, workflow_id, limit=limit, organization=org, membership=membership)


@router.get("/runs/{run_id}", response_model=api_models.RunResponse)
async def get_run_status(request: Request, run_id: str):
    user = _require_user(request)
    _, membership = _get_org_context(request)
    return await services.get_run_status(user, run_id, membership=membership)


@router.get("/runs/{run_id}/history", response_model=api_models.RunHistoryResponse)
async def get_run_history(request: Request, run_id: str):
    user = _require_user(request)
    _, membership = _get_org_context(request)
    return await services.get_run_history(user, run_id, membership=membership)


@router.get("/runs/{run_id}/interrupt", response_model=api_models.HITLInterruptResponse)
async def get_run_interrupt(request: Request, run_id: str):
    """Get pending HITL interrupt data for a workflow run."""
    user = _require_user(request)
    interrupt_data = await services.get_workflow_run_interrupt(user, run_id)
    if interrupt_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending interrupt for this run"
        )

    # Transform interrupt data to response model
    interrupt_payload = interrupt_data.get("interrupt_data") or {}
    return api_models.HITLInterruptResponse(
        run_id=interrupt_data.get("run_id", run_id),
        status=interrupt_data.get("status", "interrupted"),
        node_id=interrupt_data.get("node_id"),
        title=interrupt_payload.get("title"),
        description=interrupt_payload.get("description"),
        display=[
            api_models.HITLInterruptDisplayItem(**item)
            for item in interrupt_payload.get("display", [])
        ],
        inputs=[
            api_models.HITLInterruptInputField(**field)
            for field in interrupt_payload.get("inputs", [])
        ],
        timeout_seconds=interrupt_payload.get("timeout_seconds"),
        expires_at=interrupt_data.get("expires_at"),
        is_expired=interrupt_data.get("is_expired", False),
    )


@router.post("/runs/{run_id}/resume", response_model=api_models.RunResponse)
async def resume_run(request: Request, run_id: str, payload: api_models.HITLResumeRequest):
    """Resume a workflow run that is paused at an HITL interrupt."""
    user = _require_user(request)
    return await services.resume_workflow_run(user, run_id, payload.responses)



@router.get("/runs/{run_id}/files", response_model=api_models.WorkflowFileListResponse)
async def list_run_files(request: Request, run_id: str):
    """List all files created during a workflow run."""
    user = _require_user(request)
    _, membership = _get_org_context(request)
    return await services.list_run_files(user, run_id, membership=membership)


@router.get("/runs/{run_id}/files/{file_id}", response_model=api_models.WorkflowFileResponse)
async def get_run_file(request: Request, run_id: str, file_id: str):
    """Get metadata for a specific file in a workflow run."""
    user = _require_user(request)
    _, membership = _get_org_context(request)
    return await services.get_run_file(user, run_id, file_id, membership=membership)


@router.get("/runs/{run_id}/files/{file_id}/download", response_model=api_models.WorkflowFileDownloadResponse)
async def download_run_file(request: Request, run_id: str, file_id: str):
    """Get a presigned download URL for a file from a workflow run."""
    user = _require_user(request)
    _, membership = _get_org_context(request)
    return await services.get_run_file_download_url(user, run_id, file_id, membership=membership)


@router.delete("/runs/{run_id}/files/{file_id}", response_model=api_models.WorkflowFileDeleteResponse)
async def delete_run_file(request: Request, run_id: str, file_id: str):
    """Delete a file from a workflow run."""
    user = _require_user(request)
    _, membership = _get_org_context(request)
    return await services.delete_run_file(user, run_id, file_id, membership=membership)


@router.get("/workflows/{workflow_id}/template")
async def get_workflow_template(request: Request, workflow_id: str):
    """Get the template published from this workflow, if any."""
    user = _require_user(request)
    result = await get_workflow_template_meta(user, workflow_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No template found for this workflow")
    return result


@router.post(
    "/workflows/{workflow_id}/share-as-template",
    response_model=ShareAsTemplateResponse,
)
async def share_as_template(request: Request, workflow_id: str, payload: ShareAsTemplateRequest):
    user = _require_user(request)
    return await share_workflow_as_template(user, workflow_id, payload)


@router.post(
    "/workflows/{workflow_id}/generate-template-description",
    response_model=GenerateTemplateDescriptionResponse,
)
async def generate_template_desc(request: Request, workflow_id: str):
    user = _require_user(request)
    return await generate_template_description(user, workflow_id)


@router.get("/variables", response_model=api_models.GlobalVariableListResponse)
async def list_global_variables(request: Request):
    """List all global variables for the current organization."""
    _require_user(request)
    org, _ = _get_org_context(request)
    if org is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization context required")
    rows = await GlobalVariable.filter(organization=org).order_by("key")
    items = []
    for row in rows:
        items.append(api_models.GlobalVariableItem(
            id=row.id,
            key=row.key,
            value="" if row.is_secret else row.value,
            is_secret=row.is_secret,
            description=row.description,
            created_at=row.created_at,
            updated_at=row.updated_at,
        ))
    return api_models.GlobalVariableListResponse(items=items)


@router.post("/variables", response_model=api_models.GlobalVariableItem, status_code=status.HTTP_201_CREATED)
async def create_global_variable(request: Request, payload: api_models.GlobalVariableCreateRequest):
    """Create a new global variable."""
    user = _require_user(request)
    org, _ = _get_org_context(request)
    if org is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization context required")
    try:
        row = await GlobalVariable.create(
            organization=org,
            key=payload.key,
            value=payload.value,
            is_secret=payload.is_secret,
            description=payload.description,
            created_by=user,
        )
    except IntegrityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Variable '{payload.key}' already exists") from exc
    return api_models.GlobalVariableItem(
        id=row.id,
        key=row.key,
        value="" if row.is_secret else row.value,
        is_secret=row.is_secret,
        description=row.description,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.patch("/variables/{variable_id}", response_model=api_models.GlobalVariableItem)
async def update_global_variable(request: Request, variable_id: int, payload: api_models.GlobalVariableUpdateRequest):
    """Update an existing global variable."""
    _require_user(request)
    org, _ = _get_org_context(request)
    if org is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization context required")
    row = await GlobalVariable.filter(id=variable_id, organization=org).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Variable not found")
    if payload.value is not None:
        row.value = payload.value
    if payload.is_secret is not None:
        row.is_secret = payload.is_secret
    if payload.description is not None:
        row.description = payload.description
    await row.save()
    return api_models.GlobalVariableItem(
        id=row.id,
        key=row.key,
        value="" if row.is_secret else row.value,
        is_secret=row.is_secret,
        description=row.description,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.delete("/variables/{variable_id}", status_code=status.HTTP_200_OK)
async def delete_global_variable(request: Request, variable_id: int):
    """Delete a global variable."""
    _require_user(request)
    org, _ = _get_org_context(request)
    if org is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Organization context required")
    deleted = await GlobalVariable.filter(id=variable_id, organization=org).delete()
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Variable not found")
    return {"ok": True}
