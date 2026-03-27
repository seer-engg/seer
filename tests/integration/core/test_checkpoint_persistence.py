"""
Integration tests for workflow checkpoint persistence and recovery.

Tests:
- Checkpoint writing during workflow execution
- State recovery after interruption
- Thread ID consistency for checkpoint retrieval
- Checkpointer lifecycle management

Note: These tests mock the PostgreSQL checkpointer since the LangGraph
AsyncPostgresSaver requires its own connection pool separate from Tortoise.
Full PostgreSQL checkpointing is tested in e2e tests.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

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


def utcnow():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# Run Configuration Tests (Thread ID for Checkpoints)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_thread_id_generation(db_engine, test_user):
    """
    Test that workflow runs get thread_id for checkpoint tracking.

    Verifies:
    - run_id is available for use as thread_id
    - thread_id field can be set
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Thread ID Test",
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

    # run_id is automatically generated and can be used as thread_id
    assert run.run_id is not None
    assert run.run_id.startswith("run_")

    # thread_id can be explicitly set
    run.thread_id = run.run_id
    await run.save()

    await run.refresh_from_db()
    assert run.thread_id == run.run_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_custom_thread_id(db_engine, test_user):
    """
    Test workflow run with custom thread_id.

    Verifies:
    - Custom thread_id can be provided
    - Thread ID persists correctly
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Custom Thread ID Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    custom_thread_id = "custom_thread_abc123"

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
        thread_id=custom_thread_id,
    )

    await run.refresh_from_db()
    assert run.thread_id == custom_thread_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_thread_id_unique_per_run(db_engine, test_user):
    """
    Test that each run has a unique thread_id.

    Verifies:
    - Multiple runs have different run_ids
    - Thread IDs don't collide
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Unique Thread ID Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    runs = []
    for i in range(3):
        run = await WorkflowRun.create(
            user=test_user,
            workflow=workflow,
            workflow_version=version,
            spec=spec_dict,
            status=WorkflowRunStatus.QUEUED,
        )
        run.thread_id = run.run_id  # Use run_id as thread_id
        await run.save()
        runs.append(run)

    thread_ids = [r.thread_id for r in runs]
    assert len(set(thread_ids)) == 3  # All unique


# =============================================================================
# Checkpoint Configuration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_config_for_checkpoint(db_engine, test_user):
    """
    Test workflow run config structure for checkpointing.

    Verifies:
    - Config can store checkpoint-related settings
    - Configurable dict structure matches LangGraph expectations
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Checkpoint Config Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    run_config = {
        "configurable": {
            "thread_id": "test_thread_123",
        }
    }

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
        config=run_config,
    )

    await run.refresh_from_db()
    assert run.config["configurable"]["thread_id"] == "test_thread_123"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_build_run_config_helper():
    """
    Test the _build_run_config helper function.

    Verifies:
    - Default thread_id is set from run
    - Configurable dict is properly structured
    """
    from seer.services.workflows.execution import _build_run_config

    # Create mock run
    mock_run = MagicMock()
    mock_run.run_id = "run_test_123"
    mock_run.thread_id = None

    config = _build_run_config(mock_run, None)

    assert "configurable" in config
    assert config["configurable"]["thread_id"] == "run_test_123"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_build_run_config_uses_thread_id_if_present():
    """
    Test that _build_run_config uses thread_id from run if present.
    """
    from seer.services.workflows.execution import _build_run_config

    mock_run = MagicMock()
    mock_run.run_id = "run_test_123"
    mock_run.thread_id = "custom_thread_456"

    config = _build_run_config(mock_run, None)

    assert config["configurable"]["thread_id"] == "custom_thread_456"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_build_run_config_overrides_payload_thread_id():
    """
    Test that run's thread_id takes precedence over payload config.

    This ensures checkpoint retrieval works correctly by always using
    the run's thread_id for consistency.
    """
    from seer.services.workflows.execution import _build_run_config

    mock_run = MagicMock()
    mock_run.run_id = "run_test_123"
    mock_run.thread_id = "run_test_123"

    config_payload = {
        "configurable": {
            "thread_id": "different_thread",  # Should be overridden
        }
    }

    config = _build_run_config(mock_run, config_payload)

    assert config["configurable"]["thread_id"] == "run_test_123"


# =============================================================================
# Checkpointer Initialization Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_checkpointer_returns_none_without_db_url():
    """
    Test that get_checkpointer returns None without DATABASE_URL.
    """
    from seer.api.agents.checkpointer import get_checkpointer, _checkpointer, close_checkpointer

    # Clean up any existing checkpointer
    await close_checkpointer()

    with patch("seer.api.agents.checkpointer.config") as mock_config:
        mock_config.database_url = None

        # Reset singleton state
        import seer.api.agents.checkpointer as cp_module
        cp_module._checkpointer = None
        cp_module._checkpointer_cm = None

        checkpointer = await get_checkpointer()
        assert checkpointer is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpointer_singleton_pattern():
    """
    Test that checkpointer uses singleton pattern.

    Verifies:
    - Multiple calls return same instance
    - Lock prevents race conditions
    """
    from seer.api.agents.checkpointer import get_checkpointer, close_checkpointer

    # Clean up first
    await close_checkpointer()

    mock_saver = MagicMock()
    mock_saver.setup = AsyncMock()

    with patch("seer.api.agents.checkpointer.config") as mock_config, \
         patch("seer.api.agents.checkpointer.open_checkpointer") as mock_open:

        mock_config.database_url = "postgresql://test"

        # Setup mock context manager
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_saver)
        mock_cm.__aexit__ = AsyncMock()
        mock_open.return_value = mock_cm

        # Reset singleton
        import seer.api.agents.checkpointer as cp_module
        cp_module._checkpointer = None
        cp_module._checkpointer_cm = None

        # First call initializes
        result1 = await get_checkpointer()

        # Second call returns same instance
        result2 = await get_checkpointer()

        assert result1 is result2
        mock_open.assert_called_once()  # Only initialized once


# =============================================================================
# State Persistence Pattern Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_run_preserves_execution_state(db_engine, test_user):
    """
    Test that workflow run preserves execution state for recovery.

    Verifies:
    - Inputs are persisted
    - Config is persisted
    - State can be retrieved for resume
    """
    workflow = await Workflow.create(
        user=test_user,
        name="State Preservation Test",
    )

    spec_dict = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {"id": "node1", "type": "tool", "tool": "test.echo", "inputs": {}}
        ],
        "edges": [],
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    inputs = {"message": "Hello", "data": {"key": "value"}}
    config = {"configurable": {"thread_id": "state_test_thread"}}

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.QUEUED,
        inputs=inputs,
        config=config,
        thread_id="state_test_thread",
    )

    # Fetch fresh from database
    fetched_run = await WorkflowRun.get(id=run.id)

    assert fetched_run.inputs == inputs
    assert fetched_run.config == config
    assert fetched_run.thread_id == "state_test_thread"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interrupted_run_preserves_state(db_engine, test_user):
    """
    Test that interrupted runs preserve state for checkpoint recovery.

    Verifies:
    - HITL interrupt state is persisted
    - State can be retrieved for resume
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Interrupt State Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    interrupt_data = {
        "type": "hitl",
        "node_id": "approval_node",
        "state_snapshot": {"processed_items": 5},
    }

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        thread_id="interrupt_state_thread",
        pending_interrupt_node_id="approval_node",
        pending_interrupt_data=interrupt_data,
    )

    # Simulate resume - fetch state
    fetched_run = await WorkflowRun.get(id=run.id)

    assert fetched_run.status == WorkflowRunStatus.INTERRUPTED
    assert fetched_run.pending_interrupt_data["state_snapshot"]["processed_items"] == 5
    assert fetched_run.thread_id == "interrupt_state_thread"


# =============================================================================
# Node Traces as State History Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_node_traces_capture_execution_history(db_engine, test_user):
    """
    Test that node traces capture execution history.

    Node traces serve as a lightweight checkpoint history,
    recording which nodes executed and their outputs.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Node Traces Test",
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
        status=WorkflowRunStatus.RUNNING,
    )

    # Simulate node execution traces
    node_traces = [
        {
            "node_id": "fetch_data",
            "type": "tool",
            "started_at": "2024-01-01T00:00:00Z",
            "finished_at": "2024-01-01T00:00:01Z",
            "status": "succeeded",
            "inputs": {"url": "https://api.example.com"},
            "output": {"data": [1, 2, 3]},
        },
        {
            "node_id": "process_data",
            "type": "tool",
            "started_at": "2024-01-01T00:00:01Z",
            "finished_at": "2024-01-01T00:00:02Z",
            "status": "succeeded",
            "inputs": {"data": [1, 2, 3]},
            "output": {"result": 6},
        },
    ]

    run.node_traces = node_traces
    await run.save()

    # Fetch and verify
    fetched_run = await WorkflowRun.get(id=run.id)

    assert len(fetched_run.node_traces) == 2
    assert fetched_run.node_traces[0]["node_id"] == "fetch_data"
    assert fetched_run.node_traces[1]["output"]["result"] == 6


@pytest.mark.integration
@pytest.mark.asyncio
async def test_node_traces_on_failure(db_engine, test_user):
    """
    Test that node traces are preserved on workflow failure.

    This allows debugging failed workflows by examining
    execution history up to the failure point.
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Failure Traces Test",
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
        status=WorkflowRunStatus.RUNNING,
    )

    # Traces up to failure point
    node_traces = [
        {
            "node_id": "step_1",
            "status": "succeeded",
            "output": {"ok": True},
        },
        {
            "node_id": "step_2",
            "status": "failed",
            "error": "Connection refused",
        },
    ]

    run.status = WorkflowRunStatus.FAILED
    run.error = "Execution failed at step_2"
    run.node_traces = node_traces
    run.finished_at = utcnow()
    await run.save()

    fetched_run = await WorkflowRun.get(id=run.id)

    assert fetched_run.status == WorkflowRunStatus.FAILED
    assert len(fetched_run.node_traces) == 2
    assert fetched_run.node_traces[1]["status"] == "failed"
    assert "Connection refused" in fetched_run.node_traces[1]["error"]


# =============================================================================
# Resume State Recovery Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_restores_from_run_state(db_engine, test_user):
    """
    Test that resume operation can restore from run state.

    Verifies:
    - Thread ID is available for checkpoint lookup
    - Spec and config are available for recompilation
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Resume State Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Initial run that gets interrupted
    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.INTERRUPTED,
        thread_id="resume_thread_123",
        config={"configurable": {"thread_id": "resume_thread_123"}},
        pending_interrupt_node_id="hitl_node",
        pending_interrupt_data={"type": "hitl"},
    )

    # Fetch run for resume
    fetched_run = await WorkflowRun.get(id=run.id)
    await fetched_run.fetch_related("user", "workflow")

    # Verify all state needed for resume is available
    assert fetched_run.thread_id == "resume_thread_123"
    assert fetched_run.spec == spec_dict
    assert fetched_run.config["configurable"]["thread_id"] == "resume_thread_123"
    assert fetched_run.pending_interrupt_node_id == "hitl_node"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_interrupts_same_run(db_engine, test_user):
    """
    Test a run going through multiple interrupt/resume cycles.

    Verifies:
    - State is correctly updated at each interrupt
    - Thread ID remains consistent across cycles
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Multi-Interrupt Test",
    )

    spec_dict = {"version": "2", "triggers": [], "nodes": [], "edges": []}

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.RELEASED,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    thread_id = "multi_interrupt_thread"

    run = await WorkflowRun.create(
        user=test_user,
        workflow=workflow,
        workflow_version=version,
        spec=spec_dict,
        status=WorkflowRunStatus.RUNNING,
        thread_id=thread_id,
    )

    # First interrupt
    run.status = WorkflowRunStatus.INTERRUPTED
    run.pending_interrupt_node_id = "hitl_1"
    run.pending_interrupt_data = {"cycle": 1}
    await run.save()

    # First resume
    run.status = WorkflowRunStatus.RUNNING
    run.pending_interrupt_node_id = None
    run.pending_interrupt_data = None
    await run.save()

    # Second interrupt
    run.status = WorkflowRunStatus.INTERRUPTED
    run.pending_interrupt_node_id = "hitl_2"
    run.pending_interrupt_data = {"cycle": 2}
    await run.save()

    await run.refresh_from_db()

    # Thread ID should remain consistent
    assert run.thread_id == thread_id
    assert run.pending_interrupt_data["cycle"] == 2
