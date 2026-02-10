"""
Integration tests for complete workflow lifecycle.

Tests the end-to-end flow:
- Create workflow → Update draft → Publish → Execute → Verify traces

These tests verify that all components work together properly,
catching integration bugs where individual components work but fail together.
"""
import hashlib
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from seer.database.workflow_models import (
    Workflow,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
    WorkflowVersion,
    WorkflowVersionStatus,
)


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    """Generate hash for workflow spec."""
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


# =============================================================================
# Workflow Lifecycle Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_lifecycle_create_to_run(db_engine, test_user):
    """
    Test complete workflow lifecycle from creation to execution.

    Verifies the user journey:
    1. Create workflow
    2. Create draft version with spec
    3. Publish version
    4. Execute workflow
    5. Verify run is created with correct status
    """
    # Step 1: Create workflow
    workflow = await Workflow.create(
        user=test_user,
        name="Lifecycle Test Workflow",
        description="Testing full lifecycle",
    )
    assert workflow.workflow_id.startswith("wf_")

    # Step 2: Create draft version with spec
    spec_dict = {
        "version": "2",
        "triggers": [
            {
                "id": "webhook_trigger",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {"type": "object"},
                "meta": {
                    "sample_event": {"data": {"message": "test"}},
                    "requires_connection": False,
                },
                "ui_meta": {"title": "Test Webhook"},
            }
        ],
        "nodes": [
            {
                "id": "process_node",
                "type": "tool",
                "tool": "test.echo",
                "inputs": {"message": "${webhook_trigger.data.message}"},
            }
        ],
        "edges": [
            {"source": "webhook_trigger", "target": "process_node", "type": "trigger"}
        ],
    }

    draft_version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
        created_by=test_user,
    )
    assert draft_version.status == WorkflowVersionStatus.DRAFT

    # Step 3: Publish version
    draft_version.status = WorkflowVersionStatus.RELEASED
    await draft_version.save()

    await draft_version.refresh_from_db()
    assert draft_version.status == WorkflowVersionStatus.RELEASED

    # Step 4: Create workflow run
    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=draft_version,
        spec=spec_dict,
        source=WorkflowRunSource.MANUAL,
        status=WorkflowRunStatus.QUEUED,
    )

    # Step 5: Verify run relationships
    assert run.run_id.startswith("run_")
    assert run.status == WorkflowRunStatus.QUEUED

    # Verify relationships
    run_workflow = await run.workflow
    assert run_workflow.id == workflow.id

    run_version = await run.workflow_version
    assert run_version.id == draft_version.id

    run_user = await run.user
    assert run_user.id == test_user.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_version_progression(db_engine, test_user):
    """
    Test workflow version progression through draft -> release cycle.

    Verifies:
    - Multiple versions can be created
    - Only released versions are "active"
    - Version numbers increment correctly
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Version Progression Test",
    )

    base_spec = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    # Create v1 draft
    v1 = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.DRAFT,
        spec={**base_spec, "description": "v1"},
        spec_hash=_hash_spec({**base_spec, "description": "v1"}),
    )

    # Release v1
    v1.status = WorkflowVersionStatus.RELEASED
    await v1.save()

    # Create v2 draft (new changes)
    v2 = await WorkflowVersion.create(
        workflow=workflow,
        version_number=2,
        status=WorkflowVersionStatus.DRAFT,
        spec={**base_spec, "description": "v2"},
        spec_hash=_hash_spec({**base_spec, "description": "v2"}),
    )

    # Verify version count and states
    versions = await WorkflowVersion.filter(workflow=workflow).order_by("version_number")
    assert len(versions) == 2

    assert versions[0].version_number == 1
    assert versions[0].status == WorkflowVersionStatus.RELEASED

    assert versions[1].version_number == 2
    assert versions[1].status == WorkflowVersionStatus.DRAFT


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_status_transitions(db_engine, test_user):
    """
    Test workflow run status transitions through execution lifecycle.

    Verifies status flow:
    QUEUED -> RUNNING -> SUCCEEDED (or FAILED)
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Status Transition Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Create run in QUEUED status
    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
    )
    assert run.status == WorkflowRunStatus.QUEUED

    # Transition to RUNNING
    run.status = WorkflowRunStatus.RUNNING
    await run.save()
    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.RUNNING

    # Transition to SUCCEEDED
    run.status = WorkflowRunStatus.SUCCEEDED
    run.output = {"result": "completed"}
    await run.save()
    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.SUCCEEDED
    assert run.output == {"result": "completed"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_with_failure(db_engine, test_user):
    """
    Test workflow run failure handling.

    Verifies:
    - Run can transition to FAILED status
    - Error message is persisted
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Failure Test Workflow",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
    )

    # Simulate execution failure
    run.status = WorkflowRunStatus.FAILED
    run.error = "Tool execution failed: Connection timeout"
    await run.save()

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.FAILED
    assert "Connection timeout" in run.error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_multiple_runs(db_engine, test_user):
    """
    Test creating multiple runs for the same workflow version.

    Verifies:
    - Multiple runs can exist for same workflow
    - Each run has unique run_id
    - Runs don't interfere with each other
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Multi-Run Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Create multiple runs
    runs = []
    for i in range(3):
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            workflow_version=version,
            spec=spec_dict,
            status=WorkflowRunStatus.QUEUED,
            inputs={"iteration": i},
        )
        runs.append(run)

    # Verify all runs exist with unique IDs
    run_ids = [r.run_id for r in runs]
    assert len(set(run_ids)) == 3  # All unique

    # Complete runs with different statuses
    runs[0].status = WorkflowRunStatus.SUCCEEDED
    await runs[0].save()

    runs[1].status = WorkflowRunStatus.FAILED
    runs[1].error = "Test error"
    await runs[1].save()

    # runs[2] stays QUEUED

    # Verify each run maintains its state
    all_runs = await WorkflowRun.filter(workflow=workflow).all()
    statuses = {r.id: r.status for r in all_runs}

    assert statuses[runs[0].id] == WorkflowRunStatus.SUCCEEDED
    assert statuses[runs[1].id] == WorkflowRunStatus.FAILED
    assert statuses[runs[2].id] == WorkflowRunStatus.QUEUED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_deletion_cascades(db_engine, test_user):
    """
    Test that deleting a workflow cascades to versions and runs.

    Verifies:
    - Deleting workflow removes all versions
    - Deleting workflow removes all runs
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Cascade Delete Test",
    )
    workflow_id = workflow.id

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )
    version_id = version.id

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
    )
    run_id = run.id

    # Delete workflow
    await workflow.delete()

    # Verify cascade
    assert await Workflow.filter(id=workflow_id).first() is None
    assert await WorkflowVersion.filter(id=version_id).first() is None
    assert await WorkflowRun.filter(id=run_id).first() is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_with_trigger_source(db_engine, test_user):
    """
    Test workflow run created from trigger event.

    Verifies:
    - Run source is correctly set to TRIGGER
    - Trigger envelope is stored in inputs
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Trigger Source Test",
    )

    spec_dict = {
        "version": "2",
        "triggers": [
            {
                "id": "gmail_trigger",
                "key": "gmail.new_email",
                "mode": "polling",
                "event_schema": {},
            }
        ],
        "nodes": [],
        "edges": [],
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    trigger_envelope = {
        "trigger_id": "gmail_trigger",
        "trigger_key": "gmail.new_email",
        "data": {
            "subject": "Test Email",
            "from": "sender@example.com",
        },
    }

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        source=WorkflowRunSource.TRIGGER,
        status=WorkflowRunStatus.QUEUED,
        inputs=trigger_envelope,
    )

    await run.refresh_from_db()
    assert run.source == WorkflowRunSource.TRIGGER
    assert run.inputs["trigger_id"] == "gmail_trigger"
    assert run.inputs["data"]["subject"] == "Test Email"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_with_node_traces(db_engine, test_user):
    """
    Test storing node execution traces in workflow run.

    Verifies:
    - Node traces can be stored
    - Trace data is persisted correctly
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Node Trace Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.SUCCEEDED,
    )

    # Simulate node traces from execution
    node_traces = [
        {
            "node_id": "process_node",
            "started_at": "2024-01-01T00:00:00Z",
            "finished_at": "2024-01-01T00:00:01Z",
            "status": "succeeded",
            "output": {"result": "processed"},
        },
        {
            "node_id": "output_node",
            "started_at": "2024-01-01T00:00:01Z",
            "finished_at": "2024-01-01T00:00:02Z",
            "status": "succeeded",
            "output": {"final": True},
        },
    ]

    run.node_traces = node_traces
    await run.save()

    await run.refresh_from_db()
    assert len(run.node_traces) == 2
    assert run.node_traces[0]["node_id"] == "process_node"
    assert run.node_traces[1]["node_id"] == "output_node"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_execution_timestamps(db_engine, test_user):
    """
    Test that execution timestamps are tracked correctly.

    Verifies:
    - started_at is set when run starts
    - finished_at is set when run completes
    """
    from datetime import datetime, timezone

    workflow = await Workflow.create(
        user=test_user,
        name="Timestamp Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
    )

    # Initially no timestamps
    assert run.started_at is None
    assert run.finished_at is None

    # Start execution
    start_time = datetime.now(timezone.utc)
    run.status = WorkflowRunStatus.RUNNING
    run.started_at = start_time
    await run.save()

    await run.refresh_from_db()
    assert run.started_at is not None

    # Complete execution
    end_time = datetime.now(timezone.utc)
    run.status = WorkflowRunStatus.SUCCEEDED
    run.finished_at = end_time
    await run.save()

    await run.refresh_from_db()
    assert run.finished_at is not None
    assert run.finished_at >= run.started_at


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_user_isolation(db_engine, test_user):
    """
    Test that workflows are isolated by user.

    Verifies:
    - Workflows belong to their creator
    - Different users have separate workflow lists
    """
    from datetime import datetime, timezone
    from seer.database.models import User

    # Create second user
    user2 = await User.create(
        user_id="test_user_456",
        email="test2@example.com",
        first_name="Test2",
        last_name="User2",
        created_at=datetime.now(timezone.utc),
    )

    # Create workflow for each user
    wf1 = await Workflow.create(
        user=test_user,
        name="User 1 Workflow",
    )

    wf2 = await Workflow.create(
        user=user2,
        name="User 2 Workflow",
    )

    # Verify isolation
    user1_workflows = await Workflow.filter(user=test_user).all()
    user2_workflows = await Workflow.filter(user=user2).all()

    assert len(user1_workflows) == 1
    assert user1_workflows[0].name == "User 1 Workflow"

    assert len(user2_workflows) == 1
    assert user2_workflows[0].name == "User 2 Workflow"
