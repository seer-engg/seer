import pytest

from api.workflows import models as api_models
from api.workflows import services
from tests.api.workflows.shared_data import TEST_USER_ID


@pytest.mark.asyncio
async def test_get_run_history_returns_snapshots(db_user, workflow_spec):
    run = await services.run_draft_workflow(
        db_user,
        api_models.RunFromSpecRequest(spec=workflow_spec, inputs={"user_id": TEST_USER_ID}),
    )
    assert run.status == services.WorkflowRunStatus.SUCCEEDED.value

    response = await services.get_run_history(db_user, run.run_id)

    assert response.run_id == run.run_id
    assert response.history, "expected history snapshots to be present"
    assert any(
        snapshot.get("config", {}).get("configurable", {}).get("thread_id") == run.run_id
        for snapshot in response.history
    ), "thread_id should match run id in at least one snapshot"

