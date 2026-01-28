"""
Integration tests for workflow database models.

Tests database operations including:
- CRUD operations
- Foreign key relationships
- Unique constraints
- Cascade behavior
- Model methods and properties
"""
import pytest
from tortoise.exceptions import IntegrityError

from seer.database.workflow_models import (
    Workflow,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowRunStatus,
    WorkflowVersion,
    WorkflowVersionStatus,
    make_workflow_public_id,
    parse_workflow_public_id,
)


# =============================================================================
# Workflow Model Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_workflow(db_engine, test_user):
    """Test creating a workflow with valid data."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
    )

    assert workflow.id is not None
    assert workflow.name == "Test Workflow"
    assert workflow.user_id == test_user.id
    assert workflow.created_at is not None
    assert workflow.updated_at is not None
    assert workflow.workflow_id.startswith("wf_")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_public_id_generation(db_engine, test_user):
    """Test workflow_id property generates correct format."""
    workflow = await Workflow.create(
        user=test_user,
        name="ID Test Workflow",
    )

    workflow_id = workflow.workflow_id
    assert workflow_id == make_workflow_public_id(workflow.id)
    assert parse_workflow_public_id(workflow_id) == workflow.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_user_relationship(db_engine, test_user):
    """Test foreign key relationship between Workflow and User."""
    workflow = await Workflow.create(
        user=test_user,
        name="Relationship Test",
    )

    # Fetch user and check reverse relationship
    user = await workflow.user
    assert user.id == test_user.id
    assert user.user_id == test_user.user_id

    # Check reverse relationship
    user_workflows = await test_user.workflows.all()
    assert len(user_workflows) == 1
    assert user_workflows[0].id == workflow.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_ordering(db_engine, test_user):
    """Test workflows are ordered by updated_at descending."""
    wf1 = await Workflow.create(user=test_user, name="First")
    wf2 = await Workflow.create(user=test_user, name="Second")
    wf3 = await Workflow.create(user=test_user, name="Third")

    # Fetch all workflows
    workflows = await Workflow.all()

    # Should be ordered by most recently updated
    assert workflows[0].id == wf3.id
    assert workflows[1].id == wf2.id
    assert workflows[2].id == wf1.id


# =============================================================================
# WorkflowVersion Model Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_workflow_version(db_engine, test_user):
    """Test creating a workflow version with spec."""
    workflow = await Workflow.create(user=test_user, name="Version Test")

    spec = {
        "version": "2",
        "triggers": [],
        "nodes": [],
        "edges": [],
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec,
        spec_hash="abc123",
        version_number=1,
        created_by=test_user,
    )

    assert version.id is not None
    assert version.workflow_id == workflow.id
    assert version.status == WorkflowVersionStatus.DRAFT
    assert version.spec == spec
    assert version.spec_hash == "abc123"
    assert version.version_number == 1
    assert version.created_by_id == test_user.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_version_unique_constraint(db_engine, test_user):
    """Test unique constraint on (workflow_id, version_number)."""
    workflow = await Workflow.create(user=test_user, name="Constraint Test")

    spec = {"version": "2"}

    # Create first version
    await WorkflowVersion.create(
        workflow=workflow,
        spec=spec,
        spec_hash="hash1",
        version_number=1,
    )

    # Attempt to create duplicate version number
    with pytest.raises(IntegrityError):
        await WorkflowVersion.create(
            workflow=workflow,
            spec=spec,
            spec_hash="hash2",
            version_number=1,  # Duplicate!
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_version_cascade_delete(db_engine, test_user):
    """Test that deleting workflow cascades to versions."""
    workflow = await Workflow.create(user=test_user, name="Cascade Test")

    version = await WorkflowVersion.create(
        workflow=workflow,
        spec={"version": "2"},
        spec_hash="hash",
        version_number=1,
    )

    version_id = version.id

    # Delete workflow
    await workflow.delete()

    # Version should be deleted
    deleted_version = await WorkflowVersion.filter(id=version_id).first()
    assert deleted_version is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_multiple_versions(db_engine, test_user):
    """Test creating multiple versions for same workflow."""
    workflow = await Workflow.create(user=test_user, name="Multi-Version Test")

    # Create multiple versions
    v1 = await WorkflowVersion.create(
        workflow=workflow,
        spec={"version": "2", "data": "v1"},
        spec_hash="hash1",
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
    )

    v2 = await WorkflowVersion.create(
        workflow=workflow,
        spec={"version": "2", "data": "v2"},
        spec_hash="hash2",
        version_number=2,
        status=WorkflowVersionStatus.RELEASED,
    )

    v3 = await WorkflowVersion.create(
        workflow=workflow,
        spec={"version": "2", "data": "v3"},
        spec_hash="hash3",
        version_number=3,
        status=WorkflowVersionStatus.DRAFT,
    )

    # Fetch all versions
    versions = await WorkflowVersion.filter(workflow=workflow).order_by("version_number")

    assert len(versions) == 3
    assert versions[0].id == v1.id
    assert versions[1].id == v2.id
    assert versions[2].id == v3.id


# =============================================================================
# WorkflowRun Model Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_workflow_run(db_engine, test_user):
    """Test creating a workflow run."""
    workflow = await Workflow.create(user=test_user, name="Run Test")

    version = await WorkflowVersion.create(
        workflow=workflow,
        spec={"version": "2"},
        spec_hash="hash",
        version_number=1,
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec={"version": "2"},
        source=WorkflowRunSource.MANUAL,
        status=WorkflowRunStatus.QUEUED,
    )

    assert run.id is not None
    assert run.user_id == test_user.id
    assert run.workflow_id == workflow.id
    assert run.workflow_version_id == version.id
    assert run.source == WorkflowRunSource.MANUAL
    assert run.status == WorkflowRunStatus.QUEUED
    assert run.created_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_status_transitions(db_engine, test_user):
    """Test workflow run status transitions."""
    workflow = await Workflow.create(user=test_user, name="Status Test")

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        spec={"version": "2"},
        status=WorkflowRunStatus.QUEUED,
    )

    # Update to RUNNING
    run.status = WorkflowRunStatus.RUNNING
    await run.save()
    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.RUNNING

    # Update to SUCCEEDED
    run.status = WorkflowRunStatus.SUCCEEDED
    run.output = {"result": "success"}
    await run.save()
    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.SUCCEEDED
    assert run.output == {"result": "success"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_with_error(db_engine, test_user):
    """Test storing error information in failed run."""
    workflow = await Workflow.create(user=test_user, name="Error Test")

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        spec={"version": "2"},
        status=WorkflowRunStatus.FAILED,
        error="Test error message",
    )

    await run.refresh_from_db()
    assert run.status == WorkflowRunStatus.FAILED
    assert run.error == "Test error message"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_relationships(db_engine, test_user):
    """Test relationships between WorkflowRun and other models."""
    workflow = await Workflow.create(user=test_user, name="Relationship Test")

    version = await WorkflowVersion.create(
        workflow=workflow,
        spec={"version": "2"},
        spec_hash="hash",
        version_number=1,
    )

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec={"version": "2"},
    )

    # Test user relationship
    run_user = await run.user
    assert run_user.id == test_user.id

    # Test workflow relationship
    run_workflow = await run.workflow
    assert run_workflow.id == workflow.id

    # Test version relationship
    run_version = await run.workflow_version
    assert run_version.id == version.id

    # Test reverse relationships
    user_runs = await test_user.workflow_runs.all()
    assert len(user_runs) == 1
    assert user_runs[0].id == run.id

    workflow_runs = await workflow.runs.all()
    assert len(workflow_runs) == 1
    assert workflow_runs[0].id == run.id

    version_runs = await version.runs.all()
    assert len(version_runs) == 1
    assert version_runs[0].id == run.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_nullable_fields(db_engine, test_user):
    """Test that nullable fields can be None."""
    workflow = await Workflow.create(user=test_user, name="Nullable Test")

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        spec={"version": "2"},
    )

    assert run.inputs is None
    assert run.config is None
    assert run.output is None
    assert run.error is None
    assert run.thread_id is None
    assert run.started_at is None
    assert run.finished_at is None
    assert run.metrics is None
    assert run.subscription_id is None
    assert run.trigger_event_id is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_runs_for_workflow(db_engine, test_user):
    """Test creating multiple runs for the same workflow."""
    workflow = await Workflow.create(user=test_user, name="Multi-Run Test")

    runs = []
    for i in range(5):
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            spec={"version": "2"},
            status=WorkflowRunStatus.QUEUED,
        )
        runs.append(run)

    # Fetch all runs for workflow
    workflow_runs = await WorkflowRun.filter(workflow=workflow).all()
    assert len(workflow_runs) == 5

    # Verify all run IDs match
    run_ids = {r.id for r in runs}
    fetched_ids = {r.id for r in workflow_runs}
    assert run_ids == fetched_ids


# =============================================================================
# Utility Function Tests
# =============================================================================


@pytest.mark.integration
def test_make_workflow_public_id():
    """Test workflow public ID generation."""
    assert make_workflow_public_id(1) == "wf_1"
    assert make_workflow_public_id(123) == "wf_123"
    assert make_workflow_public_id(999999) == "wf_999999"


@pytest.mark.integration
def test_parse_workflow_public_id():
    """Test parsing workflow public ID."""
    assert parse_workflow_public_id("wf_1") == 1
    assert parse_workflow_public_id("wf_123") == 123
    assert parse_workflow_public_id("wf_999999") == 999999


@pytest.mark.integration
def test_parse_workflow_public_id_invalid():
    """Test parsing invalid workflow ID raises error."""
    with pytest.raises(ValueError, match="Invalid workflow_id format"):
        parse_workflow_public_id("invalid_1")

    with pytest.raises(ValueError, match="Invalid workflow_id format"):
        parse_workflow_public_id("run_123")

    with pytest.raises(ValueError):
        parse_workflow_public_id("wf_abc")  # Non-numeric
