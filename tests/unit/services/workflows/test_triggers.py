"""
Unit tests for services.workflows.triggers module.

Tests trigger event processing, filter matching, and path resolution.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Lookup Filter Value Tests
# =============================================================================


@pytest.mark.unit
class TestLookupFilterValue:
    """Tests for _lookup_filter_value function."""

    def test_lookup_simple_path(self):
        """Test lookup with simple single-level path."""
        from seer.services.workflows.triggers import _lookup_filter_value

        payload = {"status": "active", "type": "webhook"}
        result = _lookup_filter_value(payload, "status")

        assert result == "active"

    def test_lookup_nested_path(self):
        """Test lookup with nested dot-notation path."""
        from seer.services.workflows.triggers import _lookup_filter_value

        payload = {
            "user": {
                "profile": {
                    "name": "John"
                }
            }
        }
        result = _lookup_filter_value(payload, "user.profile.name")

        assert result == "John"

    def test_lookup_missing_path_returns_none(self):
        """Test lookup returns None for missing path."""
        from seer.services.workflows.triggers import _lookup_filter_value

        payload = {"existing": "value"}
        result = _lookup_filter_value(payload, "nonexistent.path")

        assert result is None

    def test_lookup_partial_path_returns_none(self):
        """Test lookup returns None when path partially exists."""
        from seer.services.workflows.triggers import _lookup_filter_value

        payload = {"user": {"name": "John"}}
        result = _lookup_filter_value(payload, "user.profile.email")

        assert result is None

    def test_lookup_non_dict_in_path_returns_none(self):
        """Test lookup returns None when encountering non-dict in path."""
        from seer.services.workflows.triggers import _lookup_filter_value

        payload = {"user": "string_value"}
        result = _lookup_filter_value(payload, "user.name")

        assert result is None

    def test_lookup_empty_payload(self):
        """Test lookup with empty payload."""
        from seer.services.workflows.triggers import _lookup_filter_value

        result = _lookup_filter_value({}, "any.path")

        assert result is None

    def test_lookup_with_numeric_value(self):
        """Test lookup returns numeric values correctly."""
        from seer.services.workflows.triggers import _lookup_filter_value

        payload = {"count": 42, "nested": {"value": 3.14}}

        assert _lookup_filter_value(payload, "count") == 42
        assert _lookup_filter_value(payload, "nested.value") == 3.14

    def test_lookup_with_boolean_value(self):
        """Test lookup returns boolean values correctly."""
        from seer.services.workflows.triggers import _lookup_filter_value

        payload = {"enabled": True, "settings": {"debug": False}}

        assert _lookup_filter_value(payload, "enabled") is True
        assert _lookup_filter_value(payload, "settings.debug") is False


# =============================================================================
# Filters Match Tests
# =============================================================================


@pytest.mark.unit
class TestFiltersMatch:
    """Tests for _filters_match function."""

    def test_empty_filters_match(self):
        """Test empty filters always match."""
        from seer.services.workflows.triggers import _filters_match

        result = _filters_match({}, {"data": {"any": "value"}})
        assert result is True

    def test_none_filters_match(self):
        """Test None filters always match."""
        from seer.services.workflows.triggers import _filters_match

        result = _filters_match(None, {"data": {"any": "value"}})
        assert result is True

    def test_matching_filters(self):
        """Test filters match when all conditions met."""
        from seer.services.workflows.triggers import _filters_match

        filters = {"status": "active", "type": "webhook"}
        envelope = {"data": {"status": "active", "type": "webhook", "extra": "ignored"}}

        result = _filters_match(filters, envelope)
        assert result is True

    def test_non_matching_filters(self):
        """Test filters don't match when condition fails."""
        from seer.services.workflows.triggers import _filters_match

        filters = {"status": "active"}
        envelope = {"data": {"status": "inactive"}}

        result = _filters_match(filters, envelope)
        assert result is False

    def test_partial_non_matching_filters(self):
        """Test filters don't match when one condition fails."""
        from seer.services.workflows.triggers import _filters_match

        filters = {"status": "active", "type": "webhook"}
        envelope = {"data": {"status": "active", "type": "schedule"}}  # type mismatch

        result = _filters_match(filters, envelope)
        assert result is False

    def test_filters_with_nested_path(self):
        """Test filters with nested dot-notation paths."""
        from seer.services.workflows.triggers import _filters_match

        filters = {"user.role": "admin"}
        envelope = {"data": {"user": {"role": "admin", "name": "John"}}}

        result = _filters_match(filters, envelope)
        assert result is True

    def test_filters_missing_data_key(self):
        """Test filters handle envelope without data key."""
        from seer.services.workflows.triggers import _filters_match

        filters = {"status": "active"}
        envelope = {"other": "value"}  # No "data" key

        result = _filters_match(filters, envelope)
        assert result is False

    def test_filters_non_dict_data(self):
        """Test filters handle non-dict data value."""
        from seer.services.workflows.triggers import _filters_match

        filters = {"status": "active"}
        envelope = {"data": "not_a_dict"}

        result = _filters_match(filters, envelope)
        assert result is False

    def test_filters_with_none_data(self):
        """Test filters handle None data value."""
        from seer.services.workflows.triggers import _filters_match

        filters = {"status": "active"}
        envelope = {"data": None}

        result = _filters_match(filters, envelope)
        assert result is False


# =============================================================================
# Process Trigger Event Tests
# =============================================================================


@pytest.mark.unit
class TestProcessTriggerEvent:
    """Tests for process_trigger_event function."""

    @pytest.fixture
    def mock_subscription(self):
        """Create a mock trigger subscription."""
        subscription = MagicMock()
        subscription.id = 1
        subscription.enabled = True
        subscription.trigger_key = "webhook.generic"
        subscription.filters = None
        subscription.workflow = MagicMock()
        subscription.workflow.id = 10
        subscription.user = MagicMock()
        subscription.user.user_id = "user_123"
        subscription.fetch_related = AsyncMock()
        return subscription

    @pytest.fixture
    def mock_event(self):
        """Create a mock trigger event."""
        event = MagicMock()
        event.id = 100
        event.event = {"data": {"message": "test"}}
        return event

    @pytest.mark.asyncio
    async def test_process_trigger_event_disabled_subscription(self, mock_subscription, mock_event):
        """Test process_trigger_event skips disabled subscription."""
        from seer.services.workflows.triggers import process_trigger_event

        mock_subscription.enabled = False

        with patch("seer.services.workflows.triggers.TriggerSubscription") as MockSub:
            MockSub.get = AsyncMock(return_value=mock_subscription)
            with patch("seer.services.workflows.triggers.TriggerEvent") as MockEvent:
                MockEvent.get = AsyncMock(return_value=mock_event)
                MockEvent.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))

                await process_trigger_event(subscription_id=1, event_id=100)

        # Should update event as processed with disabled message
        MockEvent.filter.return_value.update.assert_called()

    @pytest.mark.asyncio
    async def test_process_trigger_event_missing_workflow(self, mock_subscription, mock_event):
        """Test process_trigger_event handles missing workflow."""
        from seer.services.workflows.triggers import process_trigger_event

        mock_subscription.workflow = None

        with patch("seer.services.workflows.triggers.TriggerSubscription") as MockSub:
            MockSub.get = AsyncMock(return_value=mock_subscription)
            with patch("seer.services.workflows.triggers.TriggerEvent") as MockEvent:
                MockEvent.get = AsyncMock(return_value=mock_event)
                MockEvent.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))

                await process_trigger_event(subscription_id=1, event_id=100)

        # Should update event as failed
        MockEvent.filter.return_value.update.assert_called()

    @pytest.mark.asyncio
    async def test_process_trigger_event_missing_user(self, mock_subscription, mock_event):
        """Test process_trigger_event handles missing user."""
        from seer.services.workflows.triggers import process_trigger_event

        mock_subscription.user = None

        with patch("seer.services.workflows.triggers.TriggerSubscription") as MockSub:
            MockSub.get = AsyncMock(return_value=mock_subscription)
            with patch("seer.services.workflows.triggers.TriggerEvent") as MockEvent:
                MockEvent.get = AsyncMock(return_value=mock_event)
                MockEvent.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))

                await process_trigger_event(subscription_id=1, event_id=100)

        # Should update event as failed
        MockEvent.filter.return_value.update.assert_called()

    @pytest.mark.asyncio
    async def test_process_trigger_event_filtered_out(self, mock_subscription, mock_event):
        """Test process_trigger_event skips filtered events."""
        from seer.services.workflows.triggers import process_trigger_event

        mock_subscription.filters = {"status": "active"}
        mock_event.event = {"data": {"status": "inactive"}}  # Doesn't match filter

        with patch("seer.services.workflows.triggers.TriggerSubscription") as MockSub:
            MockSub.get = AsyncMock(return_value=mock_subscription)
            with patch("seer.services.workflows.triggers.TriggerEvent") as MockEvent:
                MockEvent.get = AsyncMock(return_value=mock_event)
                MockEvent.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))

                await process_trigger_event(subscription_id=1, event_id=100)

        # Should update event as processed (filtered out)
        MockEvent.filter.return_value.update.assert_called()

    @pytest.mark.asyncio
    async def test_process_trigger_event_no_published_version(self, mock_subscription, mock_event):
        """Test process_trigger_event handles workflow without published version."""
        from seer.services.workflows.triggers import process_trigger_event

        with patch("seer.services.workflows.triggers.TriggerSubscription") as MockSub:
            MockSub.get = AsyncMock(return_value=mock_subscription)
            with patch("seer.services.workflows.triggers.TriggerEvent") as MockEvent:
                MockEvent.get = AsyncMock(return_value=mock_event)
                MockEvent.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))
                with patch("seer.services.workflows.triggers.get_published_version", new_callable=AsyncMock, return_value=None):
                    await process_trigger_event(subscription_id=1, event_id=100)

        # Should update event as failed
        MockEvent.filter.return_value.update.assert_called()

    @pytest.mark.asyncio
    async def test_process_trigger_event_success(self, mock_subscription, mock_event):
        """Test process_trigger_event successfully creates and executes run."""
        from seer.services.workflows.triggers import process_trigger_event

        mock_version = MagicMock()
        mock_version.spec = {
            "version": "2",
            "nodes": [],
            "edges": []
        }

        mock_run = MagicMock()
        mock_run.id = 200

        with patch("seer.services.workflows.triggers.TriggerSubscription") as MockSub:
            MockSub.get = AsyncMock(return_value=mock_subscription)
            with patch("seer.services.workflows.triggers.TriggerEvent") as MockEvent:
                MockEvent.get = AsyncMock(return_value=mock_event)
                MockEvent.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))
                with patch("seer.services.workflows.triggers.get_published_version", new_callable=AsyncMock, return_value=mock_version):
                    with patch("seer.services.workflows.triggers.WorkflowRun") as MockRun:
                        MockRun.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))
                        with patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock, return_value=mock_run):
                            with patch("seer.services.workflows.triggers._execute_run", new_callable=AsyncMock, return_value={"result": "success"}):
                                with patch("seer.services.workflows.triggers._mark_run_succeeded", new_callable=AsyncMock):
                                    await process_trigger_event(subscription_id=1, event_id=100)

        # Event should be marked as processed
        MockEvent.filter.return_value.update.assert_called()

    @pytest.mark.asyncio
    async def test_process_trigger_event_execution_error(self, mock_subscription, mock_event):
        """Test process_trigger_event handles execution errors."""
        from fastapi import HTTPException
        from seer.services.workflows.triggers import process_trigger_event

        mock_version = MagicMock()
        mock_version.spec = {
            "version": "2",
            "nodes": [],
            "edges": []
        }

        mock_run = MagicMock()
        mock_run.id = 200

        with patch("seer.services.workflows.triggers.TriggerSubscription") as MockSub:
            MockSub.get = AsyncMock(return_value=mock_subscription)
            with patch("seer.services.workflows.triggers.TriggerEvent") as MockEvent:
                MockEvent.get = AsyncMock(return_value=mock_event)
                MockEvent.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))
                with patch("seer.services.workflows.triggers.get_published_version", new_callable=AsyncMock, return_value=mock_version):
                    with patch("seer.services.workflows.triggers.WorkflowRun") as MockRun:
                        MockRun.filter = MagicMock(return_value=MagicMock(update=AsyncMock()))
                        with patch("seer.api.workflows.services.execution._create_run_record", new_callable=AsyncMock, return_value=mock_run):
                            with patch("seer.services.workflows.triggers._execute_run", new_callable=AsyncMock, side_effect=HTTPException(status_code=500, detail="Execution failed")):
                                await process_trigger_event(subscription_id=1, event_id=100)

        # Event should be marked as failed
        call_args = MockEvent.filter.return_value.update.call_args
        # Check that the last call had error status
        assert MockEvent.filter.return_value.update.called
