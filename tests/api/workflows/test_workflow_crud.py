import pytest

from api.workflows import models as api_models
from api.workflows import services


@pytest.mark.asyncio
async def test_workflow_crud_validate_compile_flow(db_user, workflow_spec):
    """Covers full CRUD plus validate/compile path for a workflow spec."""
    create_payload = api_models.WorkflowCreateRequest(
        name="Issue triage",
        description="Filters repo issues",
        tags=["github"],
        spec=workflow_spec,
    )
    created = await services.create_workflow(db_user, create_payload)
    assert created.workflow_id.startswith("wf_")
    assert created.version == 1

    list_resp = await services.list_workflows(db_user, limit=10)
    assert len(list_resp.items) == 1

    update_payload = api_models.WorkflowUpdateRequest(
        name="Updated issue triage",
        spec=workflow_spec,
    )
    updated = await services.update_workflow(db_user, created.workflow_id, update_payload)
    assert updated.name == "Updated issue triage"
    assert updated.version == 2

    validate_resp = services.validate_spec(api_models.ValidateRequest(spec=workflow_spec))
    assert validate_resp.ok

    compile_payload = api_models.CompileRequest(
        spec=workflow_spec,
        options=api_models.CompileOptions(emit_graph_preview=True, emit_type_env=True),
    )
    compile_resp = services.compile_spec(db_user, compile_payload)
    assert compile_resp.ok
    assert compile_resp.artifacts.type_env is not None
    assert compile_resp.artifacts.graph_preview is not None

    await services.delete_workflow(db_user, created.workflow_id)
    post_delete = await services.list_workflows(db_user)
    assert post_delete.items == []


@pytest.mark.asyncio
async def test_apply_workflow_from_spec(db_user, workflow_spec):
    """Applying a new spec should bump the workflow version and persist metadata."""
    create_payload = api_models.WorkflowCreateRequest(
        name="API workflow",
        description="Original",
        tags=["demo"],
        spec=workflow_spec,
    )
    created = await services.create_workflow(db_user, create_payload)

    updated_spec_dict = workflow_spec.model_copy(deep=True).model_dump(mode="json")
    updated_spec_dict.setdefault("meta", {}).update({"notes": "refined summary"})

    applied = await services.apply_workflow_from_spec(db_user, created.workflow_id, updated_spec_dict)
    assert applied.version == 2
    assert applied.spec.meta.get("notes") == "refined summary"

