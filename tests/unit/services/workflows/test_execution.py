"""
Unit tests for workflow execution pure logic.

Tests HITL interrupt extraction, timeout calculation, and config building.
Heavy mock tests for execution orchestration have been moved to E2E tests.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def frozen_time():
    """
    Fixture to freeze _now() for deterministic time testing.

    Yields the fixed datetime that _now() will return.
    """
    fixed_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    with patch("seer.services.workflows.execution._now", return_value=fixed_time):
        yield fixed_time


@pytest.fixture
def mock_hitl_interrupt():
    """Standard HITL interrupt data structure for testing."""
    return {
        "type": "hitl",
        "node_id": "approval_node",
        "title": "Approval Required",
        "description": "Please approve this action",
        "inputs": [{"id": "decision", "question": "Approve?", "input_type": "boolean"}],
        "timeout_seconds": 3600,
        "delivery_channels": [{"type": "platform"}],
    }


@pytest.fixture
def mock_interrupt_object():
    """Factory for creating LangGraph Interrupt object mocks."""
    def _create(value: dict):
        interrupt = MagicMock()
        interrupt.value = value
        return interrupt

    return _create


@pytest.fixture
def mock_workflow_run():
    """
    Factory fixture for creating minimal WorkflowRun mocks.

    Returns a factory function that creates runs with specified state.
    """
    from typing import Optional

    def _create_run(
        run_id: str = "run_1",
        thread_id: Optional[str] = None,
    ):
        run = MagicMock()
        run.run_id = run_id
        run.thread_id = thread_id
        return run

    return _create_run


# =============================================================================
# TestExtractHitlInterrupt - HITL extraction edge cases
# =============================================================================


class TestExtractHitlInterrupt:
    """Tests for HITL interrupt extraction from workflow results."""

    def test_returns_none_when_no_interrupt_key(self):
        """Result without __interrupt__ key returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        result = {"output": "success", "data": {"value": 42}}
        assert _extract_hitl_interrupt(result) is None

    def test_returns_none_when_interrupt_is_none(self):
        """Result with __interrupt__ = None returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        result = {"output": "success", "__interrupt__": None}
        assert _extract_hitl_interrupt(result) is None

    def test_returns_none_when_interrupt_is_empty_tuple(self):
        """Result with empty interrupt tuple returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        result = {"output": "success", "__interrupt__": ()}
        assert _extract_hitl_interrupt(result) is None

    def test_returns_none_when_interrupt_is_empty_list(self):
        """Result with empty interrupt list returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        result = {"output": "success", "__interrupt__": []}
        assert _extract_hitl_interrupt(result) is None

    def test_returns_none_when_interrupt_has_non_hitl_type(self, mock_interrupt_object):
        """Interrupt with type != 'hitl' returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        interrupt = mock_interrupt_object({"type": "tool_call", "data": "something"})
        result = {"__interrupt__": (interrupt,)}
        assert _extract_hitl_interrupt(result) is None

    def test_extracts_hitl_interrupt_from_single_interrupt(
        self, mock_interrupt_object, mock_hitl_interrupt
    ):
        """Single HITL interrupt in tuple is extracted correctly."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        interrupt = mock_interrupt_object(mock_hitl_interrupt)
        result = {"__interrupt__": (interrupt,)}

        extracted = _extract_hitl_interrupt(result)

        assert extracted is not None
        assert extracted["type"] == "hitl"
        assert extracted["node_id"] == "approval_node"
        assert extracted["title"] == "Approval Required"

    def test_extracts_first_hitl_from_multiple_interrupts(self, mock_interrupt_object):
        """First HITL interrupt found in multiple interrupts."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        non_hitl = mock_interrupt_object({"type": "tool_call"})
        hitl_1 = mock_interrupt_object({"type": "hitl", "node_id": "first_hitl"})
        hitl_2 = mock_interrupt_object({"type": "hitl", "node_id": "second_hitl"})

        result = {"__interrupt__": (non_hitl, hitl_1, hitl_2)}

        extracted = _extract_hitl_interrupt(result)

        assert extracted is not None
        assert extracted["node_id"] == "first_hitl"

    def test_handles_interrupt_object_without_value_attr(self):
        """Gracefully handles Interrupt objects missing value attribute."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        interrupt = MagicMock(spec=[])  # No value attribute
        del interrupt.value  # Ensure it doesn't exist
        result = {"__interrupt__": (interrupt,)}

        assert _extract_hitl_interrupt(result) is None

    def test_handles_non_dict_interrupt_value(self, mock_interrupt_object):
        """Interrupt with non-dict value returns None."""
        from seer.services.workflows.execution import _extract_hitl_interrupt

        interrupt = mock_interrupt_object("just a string")
        result = {"__interrupt__": (interrupt,)}

        assert _extract_hitl_interrupt(result) is None


# =============================================================================
# TestCalculateInterruptExpiry - Timeout calculation
# =============================================================================


class TestCalculateInterruptExpiry:
    """Tests for interrupt timeout calculation."""

    def test_returns_none_when_timeout_is_none(self):
        """None timeout returns None (indefinite wait)."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        assert _calculate_interrupt_expiry(None) is None

    def test_returns_none_when_timeout_is_zero(self):
        """Zero timeout returns None (indefinite wait)."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        assert _calculate_interrupt_expiry(0) is None

    def test_returns_none_when_timeout_is_negative(self):
        """Negative timeout returns None."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        assert _calculate_interrupt_expiry(-10) is None
        assert _calculate_interrupt_expiry(-3600) is None

    def test_calculates_future_datetime_for_positive_timeout(self, frozen_time):
        """Positive timeout returns datetime in the future."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        expiry = _calculate_interrupt_expiry(3600)  # 1 hour

        assert expiry is not None
        assert expiry == frozen_time + timedelta(seconds=3600)

    def test_expiry_uses_utc_timezone(self, frozen_time):
        """Returned datetime is UTC timezone-aware."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        expiry = _calculate_interrupt_expiry(60)

        assert expiry.tzinfo == timezone.utc

    @pytest.mark.parametrize(
        "timeout_seconds,expected_none",
        [
            (None, True),
            (0, True),
            (-1, True),
            (-100, True),
            (1, False),
            (60, False),
            (3600, False),
            (86400, False),
        ],
    )
    def test_parametrized_timeout_values(self, timeout_seconds, expected_none, frozen_time):
        """Parametrized test for various timeout values."""
        from seer.services.workflows.execution import _calculate_interrupt_expiry

        result = _calculate_interrupt_expiry(timeout_seconds)

        if expected_none:
            assert result is None
        else:
            assert result is not None
            assert result == frozen_time + timedelta(seconds=timeout_seconds)


# =============================================================================
# TestBuildRunConfig - LangGraph config construction
# =============================================================================


class TestBuildRunConfig:
    """Tests for run configuration building."""

    def test_creates_config_with_thread_id_from_run_id(self, mock_workflow_run):
        """Config uses run.run_id as thread_id when thread_id is None."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123", thread_id=None)

        config = _build_run_config(run)

        assert config["configurable"]["thread_id"] == "run_123"

    def test_uses_run_thread_id_when_present(self, mock_workflow_run):
        """Config uses run.thread_id when it's set."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123", thread_id="custom_thread_456")

        config = _build_run_config(run)

        assert config["configurable"]["thread_id"] == "custom_thread_456"

    def test_overrides_existing_thread_id_in_payload(self, mock_workflow_run):
        """Always overrides thread_id from config_payload."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123")
        config_payload = {"configurable": {"thread_id": "old_thread"}}

        config = _build_run_config(run, config_payload)

        # Should override with run.run_id
        assert config["configurable"]["thread_id"] == "run_123"

    def test_preserves_other_configurable_keys(self, mock_workflow_run):
        """Other keys in configurable section are preserved."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123")
        config_payload = {
            "configurable": {"other_key": "other_value", "recursion_limit": 50}
        }

        config = _build_run_config(run, config_payload)

        assert config["configurable"]["thread_id"] == "run_123"
        assert config["configurable"]["other_key"] == "other_value"
        assert config["configurable"]["recursion_limit"] == 50

    def test_handles_none_config_payload(self, mock_workflow_run):
        """Handles None config_payload gracefully."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123")

        config = _build_run_config(run, None)

        assert config["configurable"]["thread_id"] == "run_123"

    def test_handles_empty_config_payload(self, mock_workflow_run):
        """Handles empty dict config_payload."""
        from seer.services.workflows.execution import _build_run_config

        run = mock_workflow_run(run_id="run_123")

        config = _build_run_config(run, {})

        assert config["configurable"]["thread_id"] == "run_123"
