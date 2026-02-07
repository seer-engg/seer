"""
Unit tests for workflow execution operations logic.

Tests the actual execution service functions with mocked database operations.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.database import (
    WorkflowRunStatus,
    WorkflowRunSource,
    make_run_public_id,
    make_workflow_public_id,
)


def utcnow():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock()
    user.id = 1
    return user


@pytest.fixture
def mock_workflow():
    """Create a mock workflow object."""
    workflow = MagicMock()
    workflow.id = 1
    workflow.workflow_id = "wf_1"
    workflow.name = "Test Workflow"
    return workflow


@pytest.fixture
def mock_workflow_version():
    """Create a mock workflow version object."""
    version = MagicMock()
    version.id = 1
    version.version_number = 1
    version.spec = {
        "version": "2",
        "nodes": [{"id": "n1", "type": "tool"}],
        "edges": [],
        "triggers": []
    }
    return version


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

    def test_queued_status_value(self):
        """Test QUEUED status value."""
        assert WorkflowRunStatus.QUEUED.value == "queued"

    def test_running_status_value(self):
        """Test RUNNING status value."""
        assert WorkflowRunStatus.RUNNING.value == "running"

    def test_succeeded_status_value(self):
        """Test SUCCEEDED status value."""
        assert WorkflowRunStatus.SUCCEEDED.value == "succeeded"

    def test_failed_status_value(self):
        """Test FAILED status value."""
        assert WorkflowRunStatus.FAILED.value == "failed"

    def test_cancelled_status_value(self):
        """Test CANCELLED status value."""
        assert WorkflowRunStatus.CANCELLED.value == "cancelled"

    def test_interrupted_status_value(self):
        """Test INTERRUPTED status value."""
        assert WorkflowRunStatus.INTERRUPTED.value == "interrupted"

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

    def test_manual_source_value(self):
        """Test MANUAL source value."""
        assert WorkflowRunSource.MANUAL.value == "manual"

    def test_trigger_source_value(self):
        """Test TRIGGER source value."""
        assert WorkflowRunSource.TRIGGER.value == "trigger"


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
    async def test_list_workflow_runs_respects_limit(self, mock_user, mock_workflow, mock_workflow_run):
        """Test that limit is applied correctly."""
        from seer.api.workflows.services.execution import list_workflow_runs

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model:

            mock_get_wf.return_value = mock_workflow
            mock_filter = MagicMock()
            mock_order_by = MagicMock()
            mock_filter.order_by.return_value = mock_order_by
            mock_order_by.limit = AsyncMock(return_value=[])
            mock_run_model.filter.return_value = mock_filter

            await list_workflow_runs(mock_user, "wf_1", limit=25)

            mock_order_by.limit.assert_called_once_with(25)

    @pytest.mark.asyncio
    async def test_list_workflow_runs_limit_clamped_min(self, mock_user, mock_workflow):
        """Test limit is clamped to minimum of 1."""
        from seer.api.workflows.services.execution import list_workflow_runs

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model:

            mock_get_wf.return_value = mock_workflow
            mock_filter = MagicMock()
            mock_order_by = MagicMock()
            mock_filter.order_by.return_value = mock_order_by
            mock_order_by.limit = AsyncMock(return_value=[])
            mock_run_model.filter.return_value = mock_filter

            await list_workflow_runs(mock_user, "wf_1", limit=0)

            # Limit should be clamped to 1
            mock_order_by.limit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_list_workflow_runs_limit_clamped_max(self, mock_user, mock_workflow):
        """Test limit is clamped to maximum of 100."""
        from seer.api.workflows.services.execution import list_workflow_runs

        with patch("seer.api.workflows.services.execution._get_workflow", new_callable=AsyncMock) as mock_get_wf, \
             patch("seer.api.workflows.services.execution.WorkflowRun") as mock_run_model:

            mock_get_wf.return_value = mock_workflow
            mock_filter = MagicMock()
            mock_order_by = MagicMock()
            mock_filter.order_by.return_value = mock_order_by
            mock_order_by.limit = AsyncMock(return_value=[])
            mock_run_model.filter.return_value = mock_filter

            await list_workflow_runs(mock_user, "wf_1", limit=500)

            # Limit should be clamped to 100
            mock_order_by.limit.assert_called_once_with(100)

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
# _generate_sample_trigger_envelope Tests
# =============================================================================


@pytest.mark.unit
class TestGenerateSampleTriggerEnvelope:
    """Tests for _generate_sample_trigger_envelope function."""

    @pytest.mark.asyncio
    async def test_generate_envelope_from_trigger_spec(self):
        """Test generating envelope from TriggerSpec."""
        from seer.api.workflows.services.execution import _generate_sample_trigger_envelope
        from seer.core.schema.models import TriggerSpec

        trigger_spec = TriggerSpec(
            id="trigger_1",
            key="webhook.generic",
            mode="webhook",
            ui_meta={"title": "My Webhook"},
            provider_config={},
            meta={"sample_event": {"data": {"message": "test"}}}
        )

        mock_definition = MagicMock()
        mock_definition.provider = "webhook"
        mock_definition.meta.sample_event = {"data": {"message": "test"}}

        with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry, \
             patch("seer.core.triggers.events.build_event_envelope") as mock_build:

            mock_registry.maybe_get.return_value = mock_definition
            mock_build.return_value = {
                "trigger_id": "trigger_1",
                "trigger_key": "webhook.generic",
                "payload": {"message": "test"}
            }

            result = await _generate_sample_trigger_envelope(trigger_spec)

            assert result is not None
            assert result["trigger_id"] == "trigger_1"
            mock_registry.maybe_get.assert_called_once_with("webhook.generic")

    @pytest.mark.asyncio
    async def test_generate_envelope_unknown_trigger_key(self):
        """Test generating envelope returns None for unknown trigger key."""
        from seer.api.workflows.services.execution import _generate_sample_trigger_envelope
        from seer.core.schema.models import TriggerSpec

        trigger_spec = TriggerSpec(
            id="trigger_1",
            key="unknown.trigger",
            mode="webhook",
            ui_meta={},
            provider_config={},
            meta={}
        )

        with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = None

            result = await _generate_sample_trigger_envelope(trigger_spec)

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_envelope_no_sample_event(self):
        """Test generating envelope returns None when no sample event."""
        from seer.api.workflows.services.execution import _generate_sample_trigger_envelope
        from seer.core.schema.models import TriggerSpec

        trigger_spec = TriggerSpec(
            id="trigger_1",
            key="webhook.generic",
            mode="webhook",
            ui_meta={},
            provider_config={},
            meta={}  # No sample_event
        )

        mock_definition = MagicMock()
        mock_definition.provider = "webhook"
        mock_definition.meta.sample_event = None

        with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_definition

            result = await _generate_sample_trigger_envelope(trigger_spec)

            assert result is None

    @pytest.mark.asyncio
    async def test_generate_envelope_fallback_to_registry_sample_event(self):
        """Test TriggerSpec falls back to registry's sample_event when spec has none."""
        from seer.api.workflows.services.execution import _generate_sample_trigger_envelope
        from seer.core.schema.models import TriggerSpec

        # TriggerSpec with NO sample_event in meta
        trigger_spec = TriggerSpec(
            id="trigger_gmail",
            key="poll.gmail.email_received",
            mode="polling",
            ui_meta={"title": "Gmail Inbox"},
            provider_config={"provider_connection_id": 123},
            meta={}  # No sample_event in spec
        )

        # Registry definition HAS sample_event
        mock_definition = MagicMock()
        mock_definition.provider = "gmail"
        mock_definition.meta.sample_event = {"data": {"subject": "Test Email", "from": "test@example.com"}}

        with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry, \
             patch("seer.core.triggers.events.build_event_envelope") as mock_build:

            mock_registry.maybe_get.return_value = mock_definition
            mock_build.return_value = {
                "trigger_id": "trigger_gmail",
                "trigger_key": "poll.gmail.email_received",
                "payload": {"subject": "Test Email", "from": "test@example.com"}
            }

            result = await _generate_sample_trigger_envelope(trigger_spec)

            # Should succeed by falling back to registry's sample_event
            assert result is not None
            assert result["trigger_id"] == "trigger_gmail"
            mock_registry.maybe_get.assert_called_once_with("poll.gmail.email_received")
            mock_build.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_envelope_from_subscription(self):
        """Test generating envelope from TriggerSubscription."""
        from seer.api.workflows.services.execution import _generate_sample_trigger_envelope
        from seer.database import TriggerSubscription

        # Create a mock that passes isinstance check for TriggerSubscription
        mock_subscription = MagicMock(spec=TriggerSubscription)
        mock_subscription.trigger_id = "sub_trigger_1"
        mock_subscription.trigger_key = "gmail.new_email"
        mock_subscription.title = "Gmail Inbox"
        mock_subscription.provider_connection_id = 123

        mock_definition = MagicMock()
        mock_definition.provider = "gmail"
        mock_definition.meta.sample_event = {"data": {"subject": "Test Email"}}

        with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry, \
             patch("seer.core.triggers.events.build_event_envelope") as mock_build:

            mock_registry.maybe_get.return_value = mock_definition
            mock_build.return_value = {"trigger_id": "sub_trigger_1"}

            result = await _generate_sample_trigger_envelope(mock_subscription)

            assert result is not None
            assert result["trigger_id"] == "sub_trigger_1"


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

    def test_multi_run_response(self):
        """Test MultiRunResponse for trigger-based runs."""
        from seer.api.workflows.models import MultiRunResponse, RunWithTrigger

        run = RunWithTrigger(
            run_id="run_1",
            status="queued",
            created_at=utcnow(),
            trigger_title="Gmail Inbox"
        )

        response = MultiRunResponse(runs=[run])

        assert len(response.runs) == 1
        assert response.runs[0].trigger_title == "Gmail Inbox"


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
             patch("seer.api.workflows.services.execution.raise_problem") as mock_raise_problem:

            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(
                side_effect=WorkflowCompilerError("llm-2.inputs: Reference 'llm-1.data' is invalid: Cannot access property 'data' on string")
            )
            mock_singleton.instance.return_value = mock_compiler
            mock_checkpointer.return_value = None
            mock_raise_problem.side_effect = HTTPException(status_code=400, detail="Validation failed")

            with pytest.raises(HTTPException) as exc_info:
                await _validate_workflow_spec(mock_user, invalid_spec)

            assert exc_info.value.status_code == 400
            mock_raise_problem.assert_called_once()
            call_kwargs = mock_raise_problem.call_args[1]
            assert call_kwargs["status"] == 400
            assert "Cannot access property 'data' on string" in call_kwargs["detail"]

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
             patch("seer.api.workflows.services.execution.raise_problem") as mock_raise_problem:

            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(
                side_effect=TypeEnvironmentError("Unknown model: fake-model")
            )
            mock_singleton.instance.return_value = mock_compiler
            mock_checkpointer.return_value = None
            mock_raise_problem.side_effect = HTTPException(status_code=400, detail="Validation failed")

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
             patch("seer.api.workflows.services.execution.raise_problem") as mock_raise_problem:

            mock_compiler = MagicMock()
            mock_compiler.compile = AsyncMock(
                side_effect=ValidationPhaseError("Workflow spec validation failed")
            )
            mock_singleton.instance.return_value = mock_compiler
            mock_checkpointer.return_value = None
            mock_raise_problem.side_effect = HTTPException(status_code=400, detail="Validation failed")

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
        """Test that validation happens before trigger processing."""
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
             patch("seer.api.workflows.services.execution._generate_sample_trigger_envelope", new_callable=AsyncMock) as mock_gen_envelope, \
             patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock) as mock_create_run, \
             patch("seer.api.workflows.services.execution.WorkflowSpec") as mock_spec_class, \
             patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions", new_callable=AsyncMock) as mock_sync_triggers:

            mock_get_wf.return_value = mock_workflow
            mock_get_draft.return_value = mock_workflow_version
            mock_spec_class.model_validate.return_value = MagicMock(triggers=[mock_trigger])
            mock_validate.side_effect = HTTPException(status_code=400, detail="Invalid workflow")

            with pytest.raises(HTTPException) as exc_info:
                await run_saved_workflow(mock_user, "wf_1", payload)

            assert exc_info.value.status_code == 400
            # Validation should be called
            mock_validate.assert_called_once()
            # Trigger envelope generation should NOT be called (validation failed first)
            mock_gen_envelope.assert_not_called()
            # Run record should NOT be created
            mock_create_run.assert_not_called()
