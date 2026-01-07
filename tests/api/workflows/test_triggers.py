from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from api.triggers import services as trigger_services
from api.workflows import models as api_models
from api.workflows import services as workflow_services
from worker.tasks import triggers as worker_trigger_tasks
from worker.tasks import workflows as worker_workflow_tasks
from shared.database.models import (
    TriggerEvent,
    TriggerEventStatus,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
)


async def _create_workflow(user, workflow_spec):
    payload = api_models.WorkflowCreateRequest(
        name="Triggerable Workflow",
        description="used in trigger tests",
        spec=workflow_spec,
    )
    return await workflow_services.create_workflow(user, payload)


async def _publish_latest_version(user, workflow, monkeypatch):
    async def immediate_run(*_, **kwargs):
        await workflow_services.execute_saved_workflow_run(**kwargs)

    monkeypatch.setattr(worker_workflow_tasks.execute_saved_workflow, "kiq", immediate_run)
    run = await workflow_services.run_saved_workflow(
        user,
        workflow.workflow_id,
        api_models.RunFromWorkflowRequest(inputs={"user_id": 99}),
    )
    await workflow_services.publish_workflow(
        user,
        workflow.workflow_id,
        api_models.WorkflowPublishRequest(version_id=run.workflow_version_id),
    )


@pytest.mark.asyncio
async def test_create_trigger_subscription_generates_secret(db_user, workflow_spec):
    workflow = await _create_workflow(db_user, workflow_spec)
    payload = api_models.TriggerSubscriptionCreateRequest(
        workflow_id=workflow.workflow_id,
        trigger_key="webhook.generic",
        bindings={"user_id": 1},
    )

    response = await workflow_services.create_trigger_subscription(db_user, payload)

    assert response.secret_token
    assert response.webhook_url == f"/v1/webhooks/generic/{response.subscription_id}"


@pytest.mark.asyncio
async def test_create_trigger_subscription_rejects_unknown_input(db_user, workflow_spec):
    workflow = await _create_workflow(db_user, workflow_spec)
    payload = api_models.TriggerSubscriptionCreateRequest(
        workflow_id=workflow.workflow_id,
        trigger_key="webhook.generic",
        bindings={"missing_input": "${event.data.value}"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await workflow_services.create_trigger_subscription(db_user, payload)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_generic_webhook_ingestion_creates_triggered_run(db_user, workflow_spec, monkeypatch):
    workflow = await _create_workflow(db_user, workflow_spec)
    await _publish_latest_version(db_user, workflow, monkeypatch)
    subscription_response = await workflow_services.create_trigger_subscription(
        db_user,
        api_models.TriggerSubscriptionCreateRequest(
            workflow_id=workflow.workflow_id,
            trigger_key="webhook.generic",
            bindings={"user_id": "${event.data.owner_id}"},
        ),
    )

    async def immediate_enqueue(*_, **kwargs):
        await trigger_services.process_trigger_run_job(**kwargs)

    monkeypatch.setattr(worker_trigger_tasks.process_trigger_event, "kiq", immediate_enqueue)

    async def fake_execute(run, user, inputs, config_payload):
        return {"inputs": inputs}

    async def fake_complete(run, output):
        await WorkflowRun.filter(id=run.id).update(
            status=WorkflowRunStatus.SUCCEEDED,
            finished_at=datetime.now(timezone.utc),
            output=output,
        )
        await run.refresh_from_db()
        return run

    monkeypatch.setattr(workflow_services, "_execute_compiled_run", fake_execute)
    monkeypatch.setattr(workflow_services, "_complete_run", fake_complete)

    await trigger_services.handle_generic_webhook(
        subscription_id=subscription_response.subscription_id,
        payload={"owner_id": 42},
        headers={"content-type": "application/json"},
        secret=subscription_response.secret_token,
        provider_event_id="evt-test",
    )

    event = await TriggerEvent.get(provider_event_id="evt-test")
    assert event.status == TriggerEventStatus.PROCESSED

    run = await WorkflowRun.get(trigger_event=event)
    assert run.source == WorkflowRunSource.TRIGGER
    assert run.subscription_id == subscription_response.subscription_id
    assert run.inputs["user_id"] == 42
    assert run.output == {"inputs": {"user_id": 42}}


@pytest.mark.asyncio
async def test_create_trigger_with_bindings_on_workflow_without_inputs(db_user):
    """Test that triggers with bindings can be created for workflows without inputs.

    This tests the "late-bound" inputs feature where bindings are validated
    at runtime instead of subscription creation time.
    """
    # Create workflow WITHOUT inputs (empty inputs dict)
    workflow_spec = {
        "version": "1",
        "inputs": {},  # No inputs defined
        "nodes": [
            {
                "id": "log_data",
                "type": "task",
                "kind": "set",
                "value": {"message": "Received data"},
                "out": "result",
            }
        ],
        "output": "${result}",
    }

    workflow = await _create_workflow(db_user, workflow_spec)

    # Should SUCCEED with bindings (late-bound validation)
    payload = api_models.TriggerSubscriptionCreateRequest(
        workflow_id=workflow.workflow_id,
        trigger_key="webhook.generic",
        bindings={"user_id": "${event.data.userId}"},  # Late-bound
    )

    response = await workflow_services.create_trigger_subscription(db_user, payload)

    assert response.subscription_id is not None
    assert response.bindings == {"user_id": "${event.data.userId}"}


@pytest.mark.asyncio
async def test_create_cron_trigger_without_inputs_or_bindings(db_user):
    """Test that cron triggers work without inputs or bindings.

    This is the primary use case for workflows that don't need any input data.
    """
    # Create workflow WITHOUT inputs
    workflow_spec = {
        "version": "1",
        "inputs": {},
        "nodes": [
            {
                "id": "scheduled_task",
                "type": "task",
                "kind": "set",
                "value": {"status": "executed"},
                "out": "result",
            }
        ],
        "output": "${result}",
    }

    workflow = await _create_workflow(db_user, workflow_spec)

    # Should SUCCEED with cron trigger and empty bindings
    payload = api_models.TriggerSubscriptionCreateRequest(
        workflow_id=workflow.workflow_id,
        trigger_key="schedule.cron",
        bindings={},  # No bindings
        provider_config={
            "cron_expression": "0 9 * * *",
            "timezone": "UTC",
        },
    )

    response = await workflow_services.create_trigger_subscription(db_user, payload)

    assert response.subscription_id is not None
    assert response.bindings == {}


@pytest.mark.asyncio
async def test_existing_workflows_with_inputs_still_validate_bindings(db_user, workflow_spec):
    """Test backward compatibility: workflows WITH inputs still validate bindings.

    This ensures the change doesn't break existing validation for workflows
    that DO define inputs.
    """
    workflow = await _create_workflow(db_user, workflow_spec)

    # Should FAIL when binding references non-existent input
    payload = api_models.TriggerSubscriptionCreateRequest(
        workflow_id=workflow.workflow_id,
        trigger_key="webhook.generic",
        bindings={"missing_input": "${event.data.value}"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await workflow_services.create_trigger_subscription(db_user, payload)

    assert exc_info.value.status_code == 400

