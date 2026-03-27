"""
Integration tests for trigger event processing service.

Tests process_trigger_event with real DB: disabled subscriptions, filter matching,
missing published version, and run creation.
"""
from unittest.mock import AsyncMock, patch

import pytest

from seer.database.workflow_models import (
    TriggerEvent,
    TriggerEventStatus,
    TriggerSubscription,
    Workflow,
    WorkflowRun,
    WorkflowRunSource,
    WorkflowVersion,
    WorkflowVersionStatus,
)
from seer.services.workflows.triggers import (
    _filters_match,
    _lookup_filter_value,
    process_trigger_event,
)


# =============================================================================
# Filter Logic (Pure)
# =============================================================================


@pytest.mark.integration
class TestFiltersMatch:
    """Tests for _filters_match."""

    def test_no_filters_matches_everything(self):
        assert _filters_match(None, {"data": {"any": "thing"}}) is True
        assert _filters_match({}, {"data": {"any": "thing"}}) is True

    def test_matching_filter(self):
        filters = {"status": "active"}
        envelope = {"data": {"status": "active"}}
        assert _filters_match(filters, envelope) is True

    def test_non_matching_filter(self):
        filters = {"status": "active"}
        envelope = {"data": {"status": "inactive"}}
        assert _filters_match(filters, envelope) is False

    def test_missing_key_does_not_match(self):
        filters = {"status": "active"}
        envelope = {"data": {}}
        assert _filters_match(filters, envelope) is False

    def test_nested_path_lookup(self):
        assert _lookup_filter_value({"a": {"b": {"c": 42}}}, "a.b.c") == 42
        assert _lookup_filter_value({"a": 1}, "a.b") is None


# =============================================================================
# Process Trigger Event (Real DB)
# =============================================================================


@pytest.mark.integration
class TestProcessTriggerEvent:
    """Tests for process_trigger_event with real database."""

    @pytest.mark.asyncio
    async def test_disabled_subscription_skips_event(self, db_engine, test_user):
        """Disabled subscription should mark event as PROCESSED and skip."""
        workflow = await Workflow.create(user=test_user, name="Disabled Sub Test")
        subscription = await TriggerSubscription.create(
            user=test_user,
            workflow=workflow,
            trigger_id="t1",
            trigger_key="test.trigger",
            enabled=False,  # Disabled
        )
        event = await TriggerEvent.create(
            trigger_key="test.trigger",
            event={"data": {"test": True}},
            status=TriggerEventStatus.RECEIVED,
        )

        await process_trigger_event(subscription.id, event.id)

        await event.refresh_from_db()
        assert event.status == TriggerEventStatus.PROCESSED
        assert "disabled" in str(event.error.get("detail", "")).lower()

    @pytest.mark.asyncio
    async def test_filtered_event_skipped(self, db_engine, test_user):
        """Event that doesn't match filters should be skipped."""
        workflow = await Workflow.create(user=test_user, name="Filter Test")
        subscription = await TriggerSubscription.create(
            user=test_user,
            workflow=workflow,
            trigger_id="t1",
            trigger_key="test.trigger",
            enabled=True,
            filters={"status": "important"},
        )
        event = await TriggerEvent.create(
            trigger_key="test.trigger",
            event={"data": {"status": "boring"}},
            status=TriggerEventStatus.RECEIVED,
        )

        await process_trigger_event(subscription.id, event.id)

        await event.refresh_from_db()
        assert event.status == TriggerEventStatus.PROCESSED
        # No run should be created
        runs = await WorkflowRun.filter(trigger_event=event).all()
        assert len(runs) == 0

    @pytest.mark.asyncio
    async def test_no_published_version_disables_subscription(self, db_engine, test_user):
        """Missing published version should fail event and disable subscription."""
        workflow = await Workflow.create(user=test_user, name="No Version Test")
        subscription = await TriggerSubscription.create(
            user=test_user,
            workflow=workflow,
            trigger_id="t1",
            trigger_key="test.trigger",
            enabled=True,
        )
        event = await TriggerEvent.create(
            trigger_key="test.trigger",
            event={"data": {}},
            status=TriggerEventStatus.RECEIVED,
        )

        await process_trigger_event(subscription.id, event.id)

        await event.refresh_from_db()
        assert event.status == TriggerEventStatus.FAILED
        assert "no published version" in event.error["detail"].lower()

        # Subscription should be disabled to prevent further failures
        await subscription.refresh_from_db()
        assert subscription.enabled is False

    @pytest.mark.asyncio
    async def test_successful_trigger_creates_run(self, db_engine, test_user):
        """Valid trigger event should create and execute a workflow run."""
        workflow = await Workflow.create(user=test_user, name="Success Trigger Test")
        spec = {"version": "2", "nodes": [], "edges": [], "triggers": []}
        await WorkflowVersion.create(
            workflow=workflow,
            version_number=1,
            status=WorkflowVersionStatus.RELEASED,
            spec=spec,
            spec_hash="abc",
        )
        subscription = await TriggerSubscription.create(
            user=test_user,
            workflow=workflow,
            trigger_id="t1",
            trigger_key="test.trigger",
            enabled=True,
        )
        event = await TriggerEvent.create(
            trigger_key="test.trigger",
            event={"data": {"payload": "test"}},
            status=TriggerEventStatus.RECEIVED,
        )

        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {"result": "triggered"}

        with (
            patch("seer.services.workflows.execution.WorkflowCompilerSingleton") as mock_compiler_cls,
            patch("seer.services.workflows.execution.get_checkpointer", new_callable=AsyncMock),
            patch("seer.services.workflows.execution.capture_workflow_run_event", new_callable=AsyncMock),
        ):
            mock_compiler_cls.instance.return_value.compile = AsyncMock(return_value=mock_compiled)
            await process_trigger_event(subscription.id, event.id)

        await event.refresh_from_db()
        assert event.status == TriggerEventStatus.PROCESSED

        # A workflow run should have been created
        runs = await WorkflowRun.filter(trigger_event=event).all()
        assert len(runs) == 1
        assert runs[0].source == WorkflowRunSource.TRIGGER
