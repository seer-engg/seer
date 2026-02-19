"""
Unit tests for workflow execution operations logic.

Tests the actual execution service functions with mocked database operations.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.database import (
    WorkflowRunStatus,
    WorkflowRunSource,
    make_run_public_id,
    make_workflow_public_id,
)
from tests.unit.helpers import utcnow

# Note: mock_user, mock_workflow, mock_workflow_version fixtures are
# provided by tests/unit/conftest.py


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_workflow_run():
    """Create a mock workflow run object."""
    run = MagicMock()
    run.id = 123
    run.run_id = "run_123"
    run.workflow_id = 1
    run.workflow_version_id = 1
    run.status = WorkflowRunStatus.QUEUED
    run.source = WorkflowRunSource.MANUAL
    run.inputs = {"key": "value"}
    run.config = {}
    run.output = None
    run.error = None
    run.thread_id = "run_123"
    run.created_at = utcnow()
    run.started_at = None
    run.finished_at = None
    return run


@pytest.fixture
def mock_workflow_spec():
    """Create a mock workflow spec."""
    from seer.core.schema.models import WorkflowSpec
    return WorkflowSpec(
        version="2",
        nodes=[],
        edges=[],
        triggers=[]
    )


# =============================================================================
# WorkflowRunStatus Enum Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowRunStatus:
    """Tests for WorkflowRunStatus enum values."""

    @pytest.mark.parametrize("enum_member,expected_value", [
        (WorkflowRunStatus.QUEUED, "queued"),
        (WorkflowRunStatus.RUNNING, "running"),
        (WorkflowRunStatus.SUCCEEDED, "succeeded"),
        (WorkflowRunStatus.FAILED, "failed"),
        (WorkflowRunStatus.CANCELLED, "cancelled"),
        (WorkflowRunStatus.INTERRUPTED, "interrupted"),
    ])
    def test_status_values(self, enum_member, expected_value):
        """Test each WorkflowRunStatus enum has correct string value."""
        assert enum_member.value == expected_value

    def test_all_status_values_exist(self):
        """Test all expected status values exist."""
        expected = {"queued", "running", "succeeded", "failed", "cancelled", "interrupted"}
        actual = {s.value for s in WorkflowRunStatus}
        assert actual == expected


# =============================================================================
# WorkflowRunSource Enum Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowRunSource:
    """Tests for WorkflowRunSource enum values."""

    @pytest.mark.parametrize("enum_member,expected_value", [
        (WorkflowRunSource.MANUAL, "manual"),
        (WorkflowRunSource.TRIGGER, "trigger"),
    ])
    def test_source_values(self, enum_member, expected_value):
        """Test each WorkflowRunSource enum has correct string value."""
        assert enum_member.value == expected_value


# =============================================================================
# Public ID Generation Tests
# =============================================================================


@pytest.mark.unit
class TestPublicIdGeneration:
    """Tests for public ID generation functions."""

    def test_make_run_public_id(self):
        """Test run public ID generation."""
        assert make_run_public_id(123) == "run_123"
        assert make_run_public_id(1) == "run_1"
        assert make_run_public_id(0) == "run_0"

    def test_make_workflow_public_id(self):
        """Test workflow public ID generation."""
        assert make_workflow_public_id(1) == "wf_1"
        assert make_workflow_public_id(456) == "wf_456"


# =============================================================================
# _serialize_run Tests
# =============================================================================


@pytest.mark.unit
class TestSerializeRun:
    """Tests for _serialize_run function."""

    def test_serialize_run_basic(self, mock_workflow_run):
        """Test basic run serialization."""
        from seer.api.workflows.services.execution import _serialize_run

        result = _serialize_run(mock_workflow_run)

        assert result.run_id == "run_123"
        assert result.status == "queued"
        assert result.workflow_id == "wf_1"
        assert result.workflow_version_id == 1
        assert result.created_at == mock_workflow_run.created_at
        assert result.started_at is None
        assert result.finished_at is None
        assert result.last_error is None

    def test_serialize_run_with_error(self, mock_workflow_run):
        """Test run serialization with error."""
        from seer.api.workflows.services.execution import _serialize_run

        mock_workflow_run.status = WorkflowRunStatus.FAILED
        mock_workflow_run.error = "Something went wrong"

        result = _serialize_run(mock_workflow_run)

        assert result.status == "failed"
        assert result.last_error == "Something went wrong"

    def test_serialize_run_completed(self, mock_workflow_run):
        """Test completed run serialization."""
        from seer.api.workflows.services.execution import _serialize_run

        mock_workflow_run.status = WorkflowRunStatus.SUCCEEDED
        mock_workflow_run.started_at = utcnow()
        mock_workflow_run.finished_at = utcnow()

        result = _serialize_run(mock_workflow_run)

        assert result.status == "succeeded"
        assert result.started_at is not None
        assert result.finished_at is not None

    def test_serialize_run_no_workflow(self, mock_workflow_run):
        """Test run serialization when workflow_id is None."""
        from seer.api.workflows.services.execution import _serialize_run

        mock_workflow_run.workflow_id = None

        result = _serialize_run(mock_workflow_run)

        assert result.workflow_id is None

    def test_serialize_run_status_string(self, mock_workflow_run):
        """Test run serialization handles string status."""
        from seer.api.workflows.services.execution import _serialize_run

        mock_workflow_run.status = "running"  # String instead of enum

        result = _serialize_run(mock_workflow_run)

        assert result.status == "running"


# =============================================================================
# _serialize_run_summary Tests
# =============================================================================


@pytest.mark.unit
class TestSerializeRunSummary:
    """Tests for _serialize_run_summary function."""

    def test_serialize_run_summary_basic(self, mock_workflow_run):
        """Test basic run summary serialization."""
        from seer.api.workflows.services.execution import _serialize_run_summary

        result = _serialize_run_summary(mock_workflow_run)

        assert result.run_id == "run_123"
        assert result.status == "queued"
        assert result.workflow_version_id == 1
        assert result.inputs == {"key": "value"}
        assert result.output is None
        assert result.error is None

    def test_serialize_run_summary_with_output(self, mock_workflow_run):
        """Test run summary serialization with output."""
        from seer.api.workflows.services.execution import _serialize_run_summary

        mock_workflow_run.status = WorkflowRunStatus.SUCCEEDED
        mock_workflow_run.output = {"result": "success", "data": [1, 2, 3]}

        result = _serialize_run_summary(mock_workflow_run)

        assert result.output == {"result": "success", "data": [1, 2, 3]}

    def test_serialize_run_summary_with_error(self, mock_workflow_run):
        """Test run summary serialization with error."""
        from seer.api.workflows.services.execution import _serialize_run_summary

        mock_workflow_run.status = WorkflowRunStatus.FAILED
        mock_workflow_run.error = "Execution failed"

        result = _serialize_run_summary(mock_workflow_run)

        assert result.error == "Execution failed"

    def test_serialize_run_summary_empty_inputs(self, mock_workflow_run):
        """Test run summary serialization with None inputs."""
        from seer.api.workflows.services.execution import _serialize_run_summary

        mock_workflow_run.inputs = None

        result = _serialize_run_summary(mock_workflow_run)

        assert result.inputs == {}


# =============================================================================
# list_workflow_runs Tests
# =============================================================================


@pytest.mark.unit
class TestListWorkflowRuns:
    """Tests for list_workflow_runs function."""

    @pytest.mark.asyncio
    async def test_list_workflow_runs_returns_runs(self, mock_user, mock_workflow, mock_workflow_run):
        """Test listing runs returns expected structure."""
        from seer.api.workflows.services.execution import list_workflow_runs

        mock_runs = [mock_workflow_run]

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model:

            mock_get_wf.return_value = mock_workflow
            mock_filter = MagicMock()
            mock_filter.order_by.return_value.limit = AsyncMock(return_value=mock_runs)
            mock_run_model.filter.return_value = mock_filter

            result = await list_workflow_runs(mock_user, "wf_1", limit=50)

            assert result.workflow_id == "wf_1"
            assert len(result.runs) == 1
            assert result.runs[0].run_id == "run_123"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_limit,expected_limit", [
        (25, 25),    # Normal value passes through
        (0, 1),      # Below minimum, clamp to 1
        (500, 100),  # Above maximum, clamp to 100
        (1, 1),      # Minimum boundary
        (100, 100),  # Maximum boundary
    ])
    async def test_list_workflow_runs_limit_clamping(self, mock_user, mock_workflow, input_limit, expected_limit):
        """Test that limit is clamped to valid range [1, 100]."""
        from seer.api.workflows.services.execution import list_workflow_runs

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model:

            mock_get_wf.return_value = mock_workflow
            mock_filter = MagicMock()
            mock_order_by = MagicMock()
            mock_filter.order_by.return_value = mock_order_by
            mock_order_by.limit = AsyncMock(return_value=[])
            mock_run_model.filter.return_value = mock_filter

            await list_workflow_runs(mock_user, "wf_1", limit=input_limit)

            mock_order_by.limit.assert_called_once_with(expected_limit)

    @pytest.mark.asyncio
    async def test_list_workflow_runs_empty_result(self, mock_user, mock_workflow):
        """Test listing runs when no runs exist."""
        from seer.api.workflows.services.execution import list_workflow_runs

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model:

            mock_get_wf.return_value = mock_workflow
            mock_filter = MagicMock()
            mock_filter.order_by.return_value.limit = AsyncMock(return_value=[])
            mock_run_model.filter.return_value = mock_filter

            result = await list_workflow_runs(mock_user, "wf_1")

            assert result.runs == []


# =============================================================================
# _create_run_record Tests
# =============================================================================


@pytest.mark.unit
class TestCreateRunRecord:
    """Tests for _create_run_record function."""

    @pytest.mark.asyncio
    async def test_create_run_record_basic(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_spec):
        """Test basic run record creation."""
        from seer.api.workflows.services.execution import _create_run_record

        mock_created_run = MagicMock()
        mock_created_run.id = 123
        mock_created_run.run_id = "run_123"
        mock_created_run.thread_id = None

        with patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model, \
             patch("seer.api.workflows.services.execution._spec_to_dict") as mock_spec_to_dict:

            mock_run_model.create = AsyncMock(return_value=mock_created_run)
            mock_run_model.filter.return_value.update = AsyncMock()
            mock_spec_to_dict.return_value = {"version": "2", "nodes": [], "edges": []}

            result = await _create_run_record(
                mock_user,
                workflow=mock_workflow,
                workflow_version=mock_workflow_version,
                spec=mock_workflow_spec,
                inputs={"test": "input"},
                config_payload={"config": "value"},
            )

            # Verify create was called with correct parameters
            mock_run_model.create.assert_called_once()
            call_kwargs = mock_run_model.create.call_args[1]
            assert call_kwargs["user"] == mock_user
            assert call_kwargs["workflow"] == mock_workflow
            assert call_kwargs["status"] == WorkflowRunStatus.QUEUED
            assert call_kwargs["inputs"] == {"test": "input"}

    @pytest.mark.asyncio
    async def test_create_run_record_sets_thread_id(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_spec):
        """Test that thread_id is set to run_id after creation."""
        from seer.api.workflows.services.execution import _create_run_record

        mock_created_run = MagicMock()
        mock_created_run.id = 456
        mock_created_run.run_id = "run_456"
        mock_created_run.thread_id = None

        with patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model, \
             patch("seer.api.workflows.services.execution._spec_to_dict") as mock_spec_to_dict:

            mock_run_model.create = AsyncMock(return_value=mock_created_run)
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_run_model.filter.return_value = mock_filter
            mock_spec_to_dict.return_value = {}

            result = await _create_run_record(
                mock_user,
                workflow=mock_workflow,
                workflow_version=mock_workflow_version,
                spec=mock_workflow_spec,
                inputs={},
                config_payload={},
            )

            # Verify thread_id was updated
            mock_run_model.filter.assert_called_with(id=456)
            mock_filter.update.assert_called_once_with(thread_id="run_456")
            assert result.thread_id == "run_456"

    @pytest.mark.asyncio
    async def test_create_run_record_with_source(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_spec):
        """Test run record creation with specific source."""
        from seer.api.workflows.services.execution import _create_run_record

        mock_created_run = MagicMock()
        mock_created_run.id = 123
        mock_created_run.run_id = "run_123"
        mock_created_run.thread_id = None

        with patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model, \
             patch("seer.api.workflows.services.execution._spec_to_dict") as mock_spec_to_dict:

            mock_run_model.create = AsyncMock(return_value=mock_created_run)
            mock_run_model.filter.return_value.update = AsyncMock()
            mock_spec_to_dict.return_value = {}

            await _create_run_record(
                mock_user,
                workflow=mock_workflow,
                workflow_version=mock_workflow_version,
                spec=mock_workflow_spec,
                inputs={},
                config_payload={},
                source=WorkflowRunSource.TRIGGER,
            )

            call_kwargs = mock_run_model.create.call_args[1]
            assert call_kwargs["source"] == WorkflowRunSource.TRIGGER

    @pytest.mark.asyncio
    async def test_create_run_record_empty_inputs(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_spec):
        """Test run record creation with None inputs defaults to empty dict."""
        from seer.api.workflows.services.execution import _create_run_record

        mock_created_run = MagicMock()
        mock_created_run.id = 123
        mock_created_run.run_id = "run_123"
        mock_created_run.thread_id = None

        with patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model, \
             patch("seer.api.workflows.services.execution._spec_to_dict") as mock_spec_to_dict:

            mock_run_model.create = AsyncMock(return_value=mock_created_run)
            mock_run_model.filter.return_value.update = AsyncMock()
            mock_spec_to_dict.return_value = {}

            await _create_run_record(
                mock_user,
                workflow=mock_workflow,
                workflow_version=mock_workflow_version,
                spec=mock_workflow_spec,
                inputs=None,
                config_payload=None,
            )

            call_kwargs = mock_run_model.create.call_args[1]
            assert call_kwargs["inputs"] == {}
            assert call_kwargs["config"] == {}


# =============================================================================
# run_saved_workflow Tests
# =============================================================================


@pytest.mark.unit
class TestRunSavedWorkflow:
    """Tests for run_saved_workflow function."""

    @pytest.mark.asyncio
    async def test_run_saved_workflow_creates_run(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_run):
        """Test running a saved workflow creates a run record."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest

        payload = RunFromWorkflowRequest(inputs={"key": "value"}, config={})

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate:

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_create_run.return_value = mock_workflow_run
            mock_task.kiq = AsyncMock()
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[])
            mock_validate.return_value = None  # Validation passes

            result = await run_saved_workflow(mock_user, "wf_1", payload)

            mock_create_run.assert_called_once()
            mock_task.kiq.assert_called_once_with(run_id=123, user_id=1)
            # Result is a RunResponse (single run, no triggers)
            assert hasattr(result, 'run_id')
            assert result.run_id == "run_123"

    @pytest.mark.asyncio
    async def test_run_saved_workflow_with_specific_version(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_run):
        """Test running a saved workflow with specific version."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.database import WorkflowVersionStatus

        payload = RunFromWorkflowRequest(version=1, inputs={}, config={})

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution.WorkflowVersion") as mock_version_model, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate:

            mock_get_wf.return_value = mock_workflow
            mock_version_model.filter.return_value.first = AsyncMock(return_value=mock_workflow_version)
            mock_create_run.return_value = mock_workflow_run
            mock_task.kiq = AsyncMock()
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[])
            mock_validate.return_value = None  # Validation passes

            result = await run_saved_workflow(mock_user, "wf_1", payload)

            mock_version_model.filter.assert_called_once_with(
                workflow=mock_workflow,
                version_number=1,
                status=WorkflowVersionStatus.RELEASED,
            )

    @pytest.mark.asyncio
    async def test_run_saved_workflow_enqueue_failure_marks_failed(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_run):
        """Test that task enqueue failure marks run as failed."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from fastapi import HTTPException
        import asyncio

        payload = RunFromWorkflowRequest(inputs={}, config={})

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task, \
             patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.execution._raise_problem") as mock_raise, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate:

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_create_run.return_value = mock_workflow_run
            mock_task.kiq = AsyncMock(side_effect=asyncio.TimeoutError("Connection timeout"))
            mock_run_model.filter.return_value.update = AsyncMock()
            mock_workflow_run.refresh_from_db = AsyncMock()
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[])
            mock_raise.side_effect = HTTPException(status_code=500)
            mock_validate.return_value = None  # Validation passes

            with pytest.raises(HTTPException):
                await run_saved_workflow(mock_user, "wf_1", payload)

            # Verify run was marked as failed
            mock_run_model.filter.assert_called_with(id=123)

    @pytest.mark.asyncio
    async def test_run_saved_workflow_with_triggers_requires_trigger_event_override(self, mock_user, mock_workflow, mock_workflow_version):
        """Test that workflow with triggers requires trigger_event_override - returns 400."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.core.schema.models import TriggerSpec
        from fastapi import HTTPException

        # Request without trigger_event_override
        payload = RunFromWorkflowRequest(inputs={}, config={})

        mock_trigger = MagicMock(spec=TriggerSpec)
        mock_trigger.id = "trigger_1"
        mock_trigger.key = "webhook.generic"
        mock_trigger.ui_meta = {"title": "Webhook"}

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._raise_problem") as mock_raise, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions", new_callable=AsyncMock):

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[mock_trigger])
            mock_validate.return_value = None
            mock_raise.side_effect = HTTPException(status_code=400)

            with pytest.raises(HTTPException) as exc_info:
                await run_saved_workflow(mock_user, "wf_1", payload)

            assert exc_info.value.status_code == 400
            # Should raise about requiring trigger event
            mock_raise.assert_called_once()
            call_kwargs = mock_raise.call_args[1]
            assert "Trigger event required" in call_kwargs["title"]
            # Run record should NOT be created
            mock_create_run.assert_not_called()


# =============================================================================
# Trigger Envelope Structure Tests
# =============================================================================


@pytest.mark.unit
class TestTriggerEnvelopeStructure:
    """Tests for trigger envelope data structure."""

    def test_envelope_required_fields(self):
        """Test envelope has required fields."""
        envelope = {
            "trigger_id": "trigger_123",
            "trigger_key": "webhook.generic",
            "title": "My Webhook",
            "provider": "webhook",
            "payload": {"data": "value"},
            "occurred_at": utcnow().isoformat(),
        }

        required = ["trigger_id", "trigger_key", "payload"]
        for field in required:
            assert field in envelope

    def test_envelope_payload_can_be_nested(self):
        """Test envelope payload can contain nested data."""
        envelope = {
            "trigger_id": "t1",
            "trigger_key": "gmail.new_email",
            "payload": {
                "email": {
                    "from": "test@example.com",
                    "subject": "Test",
                    "body": "Hello"
                }
            }
        }

        assert envelope["payload"]["email"]["from"] == "test@example.com"


# =============================================================================
# Run Response Model Tests
# =============================================================================


@pytest.mark.unit
class TestRunResponseModel:
    """Tests for RunResponse and related models."""

    def test_run_response_serialization(self):
        """Test RunResponse can be serialized to JSON."""
        from seer.api.workflows.models import RunResponse

        response = RunResponse(
            run_id="run_123",
            status="queued",
            workflow_id="wf_1",
            workflow_version_id=1,
            created_at=utcnow(),
        )

        json_str = response.model_dump_json()
        assert "run_123" in json_str
        assert "queued" in json_str

    def test_workflow_run_list_response(self):
        """Test WorkflowRunListResponse structure."""
        from seer.api.workflows.models import WorkflowRunListResponse, WorkflowRunSummary

        summary = WorkflowRunSummary(
            run_id="run_1",
            status="completed",
            created_at=utcnow(),
            inputs={"key": "value"},
        )

        response = WorkflowRunListResponse(
            workflow_id="wf_1",
            runs=[summary]
        )

        assert response.workflow_id == "wf_1"
        assert len(response.runs) == 1
        assert response.runs[0].run_id == "run_1"

# =============================================================================
# _validate_workflow_spec Tests
# =============================================================================


@pytest.mark.unit
class TestValidateWorkflowSpec:
    """Tests for _validate_workflow_spec function."""

    @pytest.mark.asyncio
    async def test_validate_workflow_spec_valid_workflow(self, mock_user):
        """Test that valid workflow passes validation."""
        from seer.api.workflows.services.execution import _validate_workflow_spec
        from seer.core.schema.models import WorkflowSpec

        valid_spec = WorkflowSpec(
            version="2",
            nodes=[],
            edges=[],
            triggers=[]
        )

        with patch("seer.api.workflows.services.execution.WorkflowCompilerSingleton") as mock_singleton, \
             patch("seer.api.workflows.services.execution.get_checkpointer", new_callable=AsyncMock) as mock_checkpointer:

            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(return_value=MagicMock())
            mock_singleton.instance.return_value = mock_compiler
            mock_checkpointer.return_value = None

            # Should not raise any exception
            await _validate_workflow_spec(mock_user, valid_spec)

            mock_compiler.compile.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_workflow_spec_invalid_expression_raises_400(self, mock_user):
        """Test that invalid expression reference raises HTTP 400."""
        from seer.api.workflows.services.execution import _validate_workflow_spec
        from seer.core.schema.models import WorkflowSpec
        from seer.core.errors import WorkflowCompilerError
        from fastapi import HTTPException

        invalid_spec = WorkflowSpec(
            version="2",
            nodes=[],
            edges=[],
            triggers=[]
        )

        with patch("seer.api.workflows.services.execution.WorkflowCompilerSingleton") as mock_singleton, \
             patch("seer.api.workflows.services.execution.get_checkpointer", new_callable=AsyncMock) as mock_checkpointer, \
             patch("seer.api.workflows.services.execution.raise_compiler_error") as mock_raise_compiler_error:

            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(
                side_effect=WorkflowCompilerError("llm-2.inputs: Reference 'llm-1.data' is invalid: Cannot access property 'data' on string")
            )
            mock_singleton.instance.return_value = mock_compiler
            mock_checkpointer.return_value = None
            mock_raise_compiler_error.side_effect = HTTPException(status_code=400, detail="Validation failed")

            with pytest.raises(HTTPException) as exc_info:
                await _validate_workflow_spec(mock_user, invalid_spec)

            assert exc_info.value.status_code == 400
            mock_raise_compiler_error.assert_called_once()
            # Check the error was passed correctly
            call_args = mock_raise_compiler_error.call_args[0]
            assert isinstance(call_args[0], WorkflowCompilerError)
            assert "Cannot access property 'data' on string" in str(call_args[0])

    @pytest.mark.asyncio
    async def test_validate_workflow_spec_type_environment_error_raises_400(self, mock_user):
        """Test that type environment error raises HTTP 400."""
        from seer.api.workflows.services.execution import _validate_workflow_spec
        from seer.core.schema.models import WorkflowSpec
        from seer.core.errors import TypeEnvironmentError
        from fastapi import HTTPException

        spec = WorkflowSpec(
            version="2",
            nodes=[],
            edges=[],
            triggers=[]
        )

        with patch("seer.api.workflows.services.execution.WorkflowCompilerSingleton") as mock_singleton, \
             patch("seer.api.workflows.services.execution.get_checkpointer", new_callable=AsyncMock) as mock_checkpointer, \
             patch("seer.api.workflows.services.execution.raise_compiler_error") as mock_raise_compiler_error:

            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(
                side_effect=TypeEnvironmentError("Unknown model: fake-model")
            )
            mock_singleton.instance.return_value = mock_compiler
            mock_checkpointer.return_value = None
            mock_raise_compiler_error.side_effect = HTTPException(status_code=400, detail="Validation failed")

            with pytest.raises(HTTPException) as exc_info:
                await _validate_workflow_spec(mock_user, spec)

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_workflow_spec_validation_phase_error_raises_400(self, mock_user):
        """Test that validation phase error raises HTTP 400."""
        from seer.api.workflows.services.execution import _validate_workflow_spec
        from seer.core.schema.models import WorkflowSpec
        from seer.core.errors import ValidationPhaseError
        from fastapi import HTTPException

        spec = WorkflowSpec(
            version="2",
            nodes=[],
            edges=[],
            triggers=[]
        )

        with patch("seer.api.workflows.services.execution.WorkflowCompilerSingleton") as mock_singleton, \
             patch("seer.api.workflows.services.execution.get_checkpointer", new_callable=AsyncMock) as mock_checkpointer, \
             patch("seer.api.workflows.services.execution.raise_compiler_error") as mock_raise_compiler_error:

            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(
                side_effect=ValidationPhaseError("Workflow spec validation failed")
            )
            mock_singleton.instance.return_value = mock_compiler
            mock_checkpointer.return_value = None
            mock_raise_compiler_error.side_effect = HTTPException(status_code=400, detail="Validation failed")

            with pytest.raises(HTTPException) as exc_info:
                await _validate_workflow_spec(mock_user, spec)

            assert exc_info.value.status_code == 400


# =============================================================================
# Early Validation in run_saved_workflow Tests
# =============================================================================


@pytest.mark.unit
class TestRunSavedWorkflowValidation:
    """Tests for early validation in run_saved_workflow function."""

    @pytest.mark.asyncio
    async def test_run_saved_workflow_validates_before_creating_run(self, mock_user, mock_workflow, mock_workflow_version):
        """Test that validation happens before run record is created."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.core.errors import WorkflowCompilerError
        from fastapi import HTTPException

        payload = RunFromWorkflowRequest(inputs={}, config={})

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class:

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[])
            mock_validate.side_effect = HTTPException(status_code=400, detail="Validation failed")

            with pytest.raises(HTTPException) as exc_info:
                await run_saved_workflow(mock_user, "wf_1", payload)

            assert exc_info.value.status_code == 400
            # Verify validation was called
            mock_validate.assert_called_once()
            # Verify run record was NOT created (validation failed first)
            mock_create_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_saved_workflow_invalid_expression_returns_400(self, mock_user, mock_workflow, mock_workflow_version):
        """Test that invalid expression in workflow returns 400 without creating a run."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from fastapi import HTTPException

        payload = RunFromWorkflowRequest(inputs={}, config={})

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class:

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[])
            mock_validate.side_effect = HTTPException(
                status_code=400,
                detail="llm-2.inputs: Reference 'llm-1.data' is invalid: Cannot access property 'data' on string"
            )

            with pytest.raises(HTTPException) as exc_info:
                await run_saved_workflow(mock_user, "wf_1", payload)

            assert exc_info.value.status_code == 400
            assert "Cannot access property 'data' on string" in str(exc_info.value.detail)
            # No run should be created
            mock_create_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_saved_workflow_valid_workflow_creates_run(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_run):
        """Test that valid workflow passes validation and creates run."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest

        payload = RunFromWorkflowRequest(inputs={}, config={})

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class:

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[])
            mock_validate.return_value = None  # Validation passes
            mock_create_run.return_value = mock_workflow_run
            mock_task.kiq = AsyncMock()

            result = await run_saved_workflow(mock_user, "wf_1", payload)

            # Verify validation was called
            mock_validate.assert_called_once()
            # Verify run was created after validation passed
            mock_create_run.assert_called_once()
            assert result.run_id == "run_123"

    @pytest.mark.asyncio
    async def test_run_saved_workflow_validation_before_triggers(self, mock_user, mock_workflow, mock_workflow_version):
        """Test that validation happens before trigger event required error."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.core.schema.models import TriggerSpec
        from fastapi import HTTPException

        payload = RunFromWorkflowRequest(inputs={}, config={})

        mock_trigger = MagicMock(spec=TriggerSpec)
        mock_trigger.id = "trigger_1"
        mock_trigger.key = "webhook.generic"
        mock_trigger.ui_meta = {"title": "Webhook"}

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions", new_callable=AsyncMock):

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[mock_trigger])
            mock_validate.side_effect = HTTPException(status_code=400, detail="Invalid workflow")

            with pytest.raises(HTTPException) as exc_info:
                await run_saved_workflow(mock_user, "wf_1", payload)

            assert exc_info.value.status_code == 400
            # Validation should be called
            mock_validate.assert_called_once()
            # Run record should NOT be created (validation failed first)
            mock_create_run.assert_not_called()


# =============================================================================
# _validate_trigger_envelope Tests
# =============================================================================


@pytest.mark.unit
class TestValidateTriggerEnvelope:
    """Tests for _validate_trigger_envelope function."""

    def test_validate_trigger_envelope_valid(self):
        """Test valid envelope passes validation."""
        from seer.api.workflows.services.execution import _validate_trigger_envelope

        envelope = {
            "trigger_key": "poll.gmail.email_received",
            "data": {"message_id": "123", "subject": "Test"}
        }

        # Should not raise
        _validate_trigger_envelope(envelope)

    def test_validate_trigger_envelope_missing_trigger_key(self):
        """Test envelope missing trigger_key raises error."""
        from seer.api.workflows.services.execution import _validate_trigger_envelope
        from fastapi import HTTPException

        envelope = {
            "data": {"message_id": "123"}
        }

        with patch("seer.api.workflows.services.execution._raise_problem") as mock_raise:
            mock_raise.side_effect = HTTPException(status_code=400)

            with pytest.raises(HTTPException):
                _validate_trigger_envelope(envelope)

            mock_raise.assert_called_once()
            call_kwargs = mock_raise.call_args[1]
            assert "trigger_key" in call_kwargs["detail"]

    def test_validate_trigger_envelope_missing_data(self):
        """Test envelope missing data raises error."""
        from seer.api.workflows.services.execution import _validate_trigger_envelope
        from fastapi import HTTPException

        envelope = {
            "trigger_key": "poll.gmail.email_received"
        }

        with patch("seer.api.workflows.services.execution._raise_problem") as mock_raise:
            mock_raise.side_effect = HTTPException(status_code=400)

            with pytest.raises(HTTPException):
                _validate_trigger_envelope(envelope)

            mock_raise.assert_called_once()
            call_kwargs = mock_raise.call_args[1]
            assert "data" in call_kwargs["detail"]

    def test_validate_trigger_envelope_data_not_dict(self):
        """Test envelope with non-dict data raises error."""
        from seer.api.workflows.services.execution import _validate_trigger_envelope
        from fastapi import HTTPException

        envelope = {
            "trigger_key": "poll.gmail.email_received",
            "data": "not a dict"
        }

        with patch("seer.api.workflows.services.execution._raise_problem") as mock_raise:
            mock_raise.side_effect = HTTPException(status_code=400)

            with pytest.raises(HTTPException):
                _validate_trigger_envelope(envelope)

            mock_raise.assert_called_once()
            call_kwargs = mock_raise.call_args[1]
            assert "object" in call_kwargs["detail"]


# =============================================================================
# run_saved_workflow with trigger_event_override Tests
# =============================================================================


@pytest.mark.unit
class TestRunSavedWorkflowWithTriggerOverride:
    """Tests for run_saved_workflow with trigger_event_override parameter."""

    @pytest.mark.asyncio
    async def test_run_with_trigger_override_single_trigger(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_run):
        """Test running workflow with trigger_event_override and single trigger."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.core.schema.models import TriggerSpec

        trigger_override = {
            "trigger_key": "poll.gmail.email_received",
            "data": {"message_id": "real_123", "subject": "Real Email"}
        }

        payload = RunFromWorkflowRequest(
            inputs={},
            config={},
            trigger_event_override=trigger_override
        )

        mock_trigger = MagicMock(spec=TriggerSpec)
        mock_trigger.id = "trigger_gmail"
        mock_trigger.key = "poll.gmail.email_received"
        mock_trigger.ui_meta = {"title": "Gmail Inbox"}

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions", new_callable=AsyncMock):

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[mock_trigger])
            mock_validate.return_value = None
            mock_create_run.return_value = mock_workflow_run
            mock_task.kiq = AsyncMock()

            result = await run_saved_workflow(mock_user, "wf_1", payload)

            # Should create a single run with the override envelope (with updated trigger_id)
            mock_create_run.assert_called_once()
            mock_task.kiq.assert_called_once()
            call_kwargs = mock_task.kiq.call_args[1]
            # The envelope should have the trigger_id updated to match the target trigger
            assert call_kwargs["trigger_envelope"]["trigger_id"] == "trigger_gmail"
            assert call_kwargs["trigger_envelope"]["data"] == trigger_override["data"]
            assert result.run_id == "run_123"

    @pytest.mark.asyncio
    async def test_run_with_trigger_override_multiple_triggers_requires_trigger_id(self, mock_user, mock_workflow, mock_workflow_version):
        """Test that multiple triggers require trigger_id when using override."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.core.schema.models import TriggerSpec
        from fastapi import HTTPException

        trigger_override = {
            "trigger_key": "poll.gmail.email_received",
            "data": {"message_id": "real_123"}
        }

        payload = RunFromWorkflowRequest(
            inputs={},
            config={},
            trigger_event_override=trigger_override
            # No trigger_id specified
        )

        mock_trigger1 = MagicMock(spec=TriggerSpec)
        mock_trigger1.id = "trigger_1"
        mock_trigger2 = MagicMock(spec=TriggerSpec)
        mock_trigger2.id = "trigger_2"

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._raise_problem") as mock_raise, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions", new_callable=AsyncMock):

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[mock_trigger1, mock_trigger2])
            mock_validate.return_value = None
            mock_raise.side_effect = HTTPException(status_code=400)

            with pytest.raises(HTTPException):
                await run_saved_workflow(mock_user, "wf_1", payload)

            # Should raise about ambiguous trigger
            mock_raise.assert_called_once()
            call_kwargs = mock_raise.call_args[1]
            assert "Ambiguous trigger" in call_kwargs["title"]

    @pytest.mark.asyncio
    async def test_run_with_trigger_override_with_trigger_id(self, mock_user, mock_workflow, mock_workflow_version, mock_workflow_run):
        """Test running workflow with trigger_event_override and explicit trigger_id."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.core.schema.models import TriggerSpec

        trigger_override = {
            "trigger_key": "poll.gmail.email_received",
            "data": {"message_id": "real_123"}
        }

        payload = RunFromWorkflowRequest(
            inputs={},
            config={},
            trigger_event_override=trigger_override,
            trigger_id="trigger_2"
        )

        mock_trigger1 = MagicMock(spec=TriggerSpec)
        mock_trigger1.id = "trigger_1"
        mock_trigger1.key = "poll.gmail.email_received"
        mock_trigger1.ui_meta = {"title": "Trigger 1"}
        mock_trigger2 = MagicMock(spec=TriggerSpec)
        mock_trigger2.id = "trigger_2"
        mock_trigger2.key = "poll.gmail.email_received"
        mock_trigger2.ui_meta = {"title": "Trigger 2"}

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions", new_callable=AsyncMock):

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[mock_trigger1, mock_trigger2])
            mock_validate.return_value = None
            mock_create_run.return_value = mock_workflow_run
            mock_task.kiq = AsyncMock()

            result = await run_saved_workflow(mock_user, "wf_1", payload)

            # Should create run with the override (trigger_id updated to trigger_2)
            mock_create_run.assert_called_once()
            mock_task.kiq.assert_called_once()
            call_kwargs = mock_task.kiq.call_args[1]
            assert call_kwargs["trigger_envelope"]["trigger_id"] == "trigger_2"
            assert result.run_id == "run_123"

    @pytest.mark.asyncio
    async def test_run_with_trigger_override_invalid_trigger_id(self, mock_user, mock_workflow, mock_workflow_version):
        """Test that invalid trigger_id raises error."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from seer.core.schema.models import TriggerSpec
        from fastapi import HTTPException

        trigger_override = {
            "trigger_key": "poll.gmail.email_received",
            "data": {"message_id": "real_123"}
        }

        payload = RunFromWorkflowRequest(
            inputs={},
            config={},
            trigger_event_override=trigger_override,
            trigger_id="nonexistent_trigger"
        )

        mock_trigger = MagicMock(spec=TriggerSpec)
        mock_trigger.id = "trigger_1"

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._raise_problem") as mock_raise, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions", new_callable=AsyncMock):

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[mock_trigger])
            mock_validate.return_value = None
            mock_raise.side_effect = HTTPException(status_code=404)

            with pytest.raises(HTTPException):
                await run_saved_workflow(mock_user, "wf_1", payload)

            mock_raise.assert_called_once()
            call_kwargs = mock_raise.call_args[1]
            assert "Trigger not found" in call_kwargs["title"]

    @pytest.mark.asyncio
    async def test_run_with_invalid_trigger_override_envelope(self, mock_user, mock_workflow, mock_workflow_version):
        """Test that invalid envelope structure raises error."""
        from seer.api.workflows.services.execution import run_saved_workflow
        from seer.api.workflows.models import RunFromWorkflowRequest
        from fastapi import HTTPException

        # Missing required 'data' field
        trigger_override = {
            "trigger_key": "poll.gmail.email_received"
        }

        payload = RunFromWorkflowRequest(
            inputs={},
            config={},
            trigger_event_override=trigger_override
        )

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution._get_draft_version", new_callable=AsyncMock) as mock_get_draft, \
             patch("seer.api.workflows.services.execution._validate_workflow_spec", new_callable=AsyncMock) as mock_validate, \
             patch("seer.api.workflows.services.execution._raise_problem") as mock_raise, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class:

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[])
            mock_validate.return_value = None
            mock_raise.side_effect = HTTPException(status_code=400)

            with pytest.raises(HTTPException):
                await run_saved_workflow(mock_user, "wf_1", payload)

            mock_raise.assert_called_once()
            call_kwargs = mock_raise.call_args[1]
            assert "Invalid trigger envelope" in call_kwargs["title"]
