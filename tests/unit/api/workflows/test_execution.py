"""
Unit tests for workflow execution API pure logic.

Tests serialization, response models, and data structure validation.
Heavy mock tests for API orchestration have been moved to E2E tests.
"""
from unittest.mock import MagicMock, patch

import pytest

from seer.database import WorkflowRunStatus
from tests.unit.helpers import utcnow


pytestmark = pytest.mark.unit


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
    run.inputs = {"key": "value"}
    run.config = {}
    run.output = None
    run.error = None
    run.thread_id = "run_123"
    run.created_at = utcnow()
    run.started_at = None
    run.finished_at = None
    return run


# =============================================================================
# _serialize_run Tests
# =============================================================================


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
# Trigger Envelope Structure Tests
# =============================================================================


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
# _validate_trigger_envelope Tests
# =============================================================================


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
