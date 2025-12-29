import pytest

from api.workflows import models as api_models
from api.workflows import services
from tests.api.workflows.shared_data import TEST_USER_ID


@pytest.mark.asyncio
async def test_run_lifecycle_from_saved_and_draft(db_user, workflow_spec):
    """Verifies saved and draft workflow runs execute and surface results."""
    created = await services.create_workflow(
        db_user,
        api_models.WorkflowCreateRequest(
            name="Runnable workflow",
            description=None,
            spec=workflow_spec,
            tags=[],
        ),
    )

    saved_run = await services.run_saved_workflow(
        db_user,
        created.workflow_id,
        api_models.RunFromWorkflowRequest(inputs={"user_id": TEST_USER_ID}),
    )
    assert saved_run.status == services.WorkflowRunStatus.SUCCEEDED.value

    status = await services.get_run_status(db_user, saved_run.run_id)
    assert status.run_id == saved_run.run_id

    result = await services.get_run_result(db_user, saved_run.run_id)
    assert result.status == services.WorkflowRunStatus.SUCCEEDED.value

    draft_run = await services.run_draft_workflow(
        db_user,
        api_models.RunFromSpecRequest(spec=workflow_spec, inputs={"user_id": TEST_USER_ID}),
    )
    assert draft_run.status == services.WorkflowRunStatus.SUCCEEDED.value

