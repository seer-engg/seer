"""
Integration tests for trigger subscription cleanup during workflow lifecycle operations.

Tests verify proper cleanup of:
1. TriggerSubscription rows during publish (when triggers are removed from spec)
2. TriggerSubscription rows during workflow deletion
3. External resources (Supabase webhooks) during cleanup operations
4. TriggerEvent orphaning behavior

TDD Approach: These tests are written to verify/expose cleanup behavior.
- Publish cleanup tests should PASS (behavior works correctly)
- Delete cleanup tests should DOCUMENT bugs (missing Supabase webhook cleanup)
"""
from unittest.mock import AsyncMock, patch

import pytest

from seer.database.workflow_models import (
    Workflow,
    TriggerSubscription,
    TriggerEvent,
    TriggerEventStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def workflow_with_subscription(db_engine, test_user):
    """
    Create a workflow with a generic trigger subscription.

    Returns:
        tuple: (workflow, subscription)
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Cleanup Test Workflow",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_id="generic_trigger",
        trigger_key="webhook.generic",
        is_polling=False,
        enabled=True,
    )

    return workflow, subscription


@pytest.fixture
async def supabase_subscription(db_engine, test_user):
    """
    Create a workflow with a Supabase webhook trigger subscription.

    This trigger type has external webhook resources that need cleanup.

    Returns:
        tuple: (workflow, subscription)
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Supabase Cleanup Test Workflow",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_id="supabase_trigger",
        trigger_key="webhook.supabase.db_changes",
        is_polling=False,
        enabled=True,
        provider_config={
            "project_ref": "test-project",
            "table": "users",
            "events": ["INSERT", "UPDATE"],
        },
    )

    return workflow, subscription


@pytest.fixture
async def subscription_with_events(db_engine, test_user):
    """
    Create a subscription with associated TriggerEvent records.

    Returns:
        tuple: (workflow, subscription, list of events)
    """
    workflow = await Workflow.create(
        user=test_user,
        name="Event Cleanup Test Workflow",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_id="event_trigger",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
    )

    # Create multiple events linked to this subscription
    events = []
    for i in range(3):
        event = await TriggerEvent.create(
            trigger_key="test.trigger",
            subscription_id=subscription.id,
            event={"index": i, "data": f"event_{i}"},
            status=TriggerEventStatus.RECEIVED,
            provider_event_id=f"evt_{i}",
        )
        events.append(event)

    return workflow, subscription, events


# =============================================================================
# Publish Workflow Cleanup Tests (verify existing behavior works)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_publish_deletes_stale_trigger_subscription(db_engine, test_user):
    """
    Verify that publishing a workflow with a trigger removed deletes the stale subscription.

    Scenario:
    1. Create workflow with trigger subscription
    2. Publish new version WITHOUT the trigger
    3. Verify subscription is deleted via _reconcile_existing_subscriptions()
    """
    from seer.api.workflows.services.triggers import sync_trigger_subscriptions
    from seer.core.schema.models import WorkflowSpec

    # Setup: Create workflow with subscription
    workflow = await Workflow.create(
        user=test_user,
        name="Stale Trigger Test",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_id="old_trigger",
        trigger_key="webhook.generic",
        is_polling=False,
        enabled=True,
    )
    subscription_id = subscription.id

    # Verify subscription exists
    assert await TriggerSubscription.filter(id=subscription_id).first() is not None

    # Create new spec WITHOUT the trigger
    new_spec = WorkflowSpec(
        version="2",
        triggers=[],  # No triggers - the old one should be deleted
        nodes=[],
        edges=[],
    )

    # Act: Sync triggers (called during publish)
    await sync_trigger_subscriptions(
        user=test_user,
        workflow=workflow,
        spec=new_spec,
        skip_validation=True,
    )

    # Assert: Subscription should be deleted
    remaining_subscription = await TriggerSubscription.filter(id=subscription_id).first()
    assert remaining_subscription is None, "Stale subscription should be deleted during sync"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_publish_calls_supabase_cleanup_for_removed_trigger(db_engine, test_user):
    """
    Verify that Supabase webhook cleanup is called when removing a Supabase trigger.

    This test confirms that _reconcile_existing_subscriptions() calls
    delete_trigger_subscription() which includes Supabase webhook cleanup.
    """
    from seer.api.workflows.services.triggers import sync_trigger_subscriptions
    from seer.core.schema.models import WorkflowSpec

    # Setup: Create workflow with Supabase subscription
    workflow = await Workflow.create(
        user=test_user,
        name="Supabase Cleanup Test",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_id="supabase_trigger",
        trigger_key="webhook.supabase.db_changes",
        is_polling=False,
        enabled=True,
        provider_config={
            "project_ref": "test-project",
            "table": "users",
        },
    )
    subscription_id = subscription.id

    # Create new spec WITHOUT the Supabase trigger
    new_spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[],
        edges=[],
    )

    # Act: Sync triggers with Supabase cleanup mocked
    with patch(
        "seer.api.workflows.services.triggers.delete_database_webhook",
        new_callable=AsyncMock,
    ) as mock_delete_webhook:
        await sync_trigger_subscriptions(
            user=test_user,
            workflow=workflow,
            spec=new_spec,
            skip_validation=True,
        )

        # Assert: Supabase webhook cleanup should be called
        mock_delete_webhook.assert_called_once()
        # Verify the subscription object was passed
        call_args = mock_delete_webhook.call_args
        called_subscription = call_args[0][0]
        assert called_subscription.trigger_key == "webhook.supabase.db_changes"

    # Verify subscription is deleted
    assert await TriggerSubscription.filter(id=subscription_id).first() is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_publish_preserves_existing_subscription_when_in_spec(db_engine, test_user):
    """
    Verify that subscriptions present in the new spec are NOT deleted.

    This ensures we only delete stale subscriptions, not active ones.
    """
    from seer.api.workflows.services.triggers import sync_trigger_subscriptions
    from seer.core.schema.models import WorkflowSpec, TriggerSpec

    # Setup: Create workflow with subscription
    workflow = await Workflow.create(
        user=test_user,
        name="Preserve Trigger Test",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_id="keep_this_trigger",
        trigger_key="webhook.generic",
        is_polling=False,
        enabled=True,
    )
    subscription_id = subscription.id

    # Create new spec WITH the same trigger
    new_spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="keep_this_trigger",
                key="webhook.generic",
                mode="webhook",
                event_schema={"type": "object"},
            )
        ],
        nodes=[],
        edges=[],
    )

    # Act: Sync triggers
    await sync_trigger_subscriptions(
        user=test_user,
        workflow=workflow,
        spec=new_spec,
        skip_validation=True,
    )

    # Assert: Subscription should still exist
    remaining_subscription = await TriggerSubscription.filter(id=subscription_id).first()
    assert remaining_subscription is not None, "Subscription in spec should be preserved"


# =============================================================================
# Delete Workflow Cleanup Tests (expose bugs)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_workflow_cascades_subscriptions(workflow_with_subscription):
    """
    Verify that deleting a workflow cascades to delete TriggerSubscription rows.

    This tests the DB cascade behavior (on_delete=CASCADE on workflow FK).
    """
    workflow, subscription = workflow_with_subscription
    workflow_id = workflow.id
    subscription_id = subscription.id

    # Verify subscription exists before delete
    assert await TriggerSubscription.filter(id=subscription_id).first() is not None

    # Act: Delete workflow
    await workflow.delete()

    # Assert: Both workflow and subscription should be deleted
    assert await Workflow.filter(id=workflow_id).first() is None
    assert await TriggerSubscription.filter(id=subscription_id).first() is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_workflow_does_not_call_supabase_cleanup(supabase_subscription):
    """
    BUG TEST: Verify that delete_workflow() does NOT call Supabase webhook cleanup.

    Current behavior (BUG):
    - delete_workflow() just calls workflow.delete()
    - This triggers DB cascade to delete subscriptions
    - BUT delete_trigger_subscription() is NOT called
    - So Supabase webhooks are NOT cleaned up

    Expected behavior:
    - delete_workflow() should iterate over subscriptions
    - Call delete_trigger_subscription() for each
    - This ensures external resources (Supabase webhooks) are cleaned up

    This test DOCUMENTS the bug - it should PASS showing the bug exists.
    When the bug is fixed, this test should be updated to verify cleanup IS called.
    """
    from seer.api.workflows.services.lifecycle import delete_workflow

    workflow, subscription = supabase_subscription
    subscription_id = subscription.id

    # Get user for the delete call
    user = await workflow.user

    # Act: Delete workflow with Supabase cleanup mocked
    with patch(
        "seer.api.workflows.services.triggers.delete_database_webhook",
        new_callable=AsyncMock,
    ) as mock_delete_webhook:
        await delete_workflow(user, workflow.workflow_id)

        # Assert (documenting the bug): Supabase cleanup should NOT be called
        # because delete_workflow() bypasses delete_trigger_subscription()
        mock_delete_webhook.assert_not_called()

    # Subscription should still be deleted via DB cascade
    assert await TriggerSubscription.filter(id=subscription_id).first() is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_workflow_orphans_trigger_events(subscription_with_events):
    """
    Verify that deleting a workflow orphans TriggerEvent records.

    TriggerEvent uses subscription_id as IntField (not ForeignKey),
    so events are NOT cascade-deleted when subscriptions are deleted.

    This test DOCUMENTS this behavior.
    """
    workflow, subscription, events = subscription_with_events
    workflow_id = workflow.id
    subscription_id = subscription.id
    event_ids = [e.id for e in events]

    # Verify events exist before delete
    for event_id in event_ids:
        assert await TriggerEvent.filter(id=event_id).first() is not None

    # Act: Delete workflow (cascades to subscription)
    await workflow.delete()

    # Assert: Workflow and subscription should be deleted
    assert await Workflow.filter(id=workflow_id).first() is None
    assert await TriggerSubscription.filter(id=subscription_id).first() is None

    # Assert: Events should STILL exist (orphaned with dangling subscription_id)
    for event_id in event_ids:
        event = await TriggerEvent.filter(id=event_id).first()
        assert event is not None, f"Event {event_id} should still exist (orphaned)"
        # The subscription_id now points to a non-existent subscription
        assert event.subscription_id == subscription_id


# =============================================================================
# Event Orphan Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_events_orphaned_on_subscription_cascade_delete(subscription_with_events):
    """
    Verify events become orphaned when subscription is cascade-deleted.

    This test directly demonstrates the orphaning behavior when a subscription
    is deleted via cascade (not via delete_trigger_subscription).
    """
    workflow, subscription, events = subscription_with_events
    subscription_id = subscription.id
    event_ids = [e.id for e in events]

    # Verify events reference the subscription
    for event in events:
        assert event.subscription_id == subscription_id

    # Act: Delete subscription directly (simulating cascade delete)
    await subscription.delete()

    # Assert: Events still exist with now-invalid subscription_id
    for event_id in event_ids:
        orphaned_event = await TriggerEvent.filter(id=event_id).first()
        assert orphaned_event is not None, "Event should still exist after subscription deletion"
        assert orphaned_event.subscription_id == subscription_id, "Event retains dangling subscription_id"

    # Verify the subscription no longer exists
    assert await TriggerSubscription.filter(id=subscription_id).first() is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_events_properly_cleaned_via_delete_trigger_subscription(db_engine, test_user):
    """
    Test that delete_trigger_subscription() itself does NOT clean up events.

    This documents that even the proper delete path doesn't clean up events,
    confirming that event cleanup is not implemented anywhere.

    Note: The proper fix would be either:
    1. Add event cleanup to delete_trigger_subscription()
    2. Make subscription_id a proper ForeignKey with CASCADE
    """
    from seer.api.workflows.services.triggers import delete_trigger_subscription

    # Setup: Create workflow with subscription and events
    workflow = await Workflow.create(
        user=test_user,
        name="Event Cleanup Test",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_id="event_cleanup_trigger",
        trigger_key="test.trigger",
        is_polling=True,
        enabled=True,
    )
    subscription_id = subscription.id

    # Create events
    event_ids = []
    for i in range(2):
        event = await TriggerEvent.create(
            trigger_key="test.trigger",
            subscription_id=subscription_id,
            event={"data": f"test_{i}"},
            status=TriggerEventStatus.RECEIVED,
            provider_event_id=f"cleanup_evt_{i}",
        )
        event_ids.append(event.id)

    # Act: Delete subscription via the proper delete function
    await delete_trigger_subscription(test_user, subscription_id)

    # Assert: Subscription should be deleted
    assert await TriggerSubscription.filter(id=subscription_id).first() is None

    # Assert: Events should still exist (orphaned)
    # This documents that delete_trigger_subscription() doesn't clean up events
    for event_id in event_ids:
        event = await TriggerEvent.filter(id=event_id).first()
        assert event is not None, "Events should still exist - no cleanup implemented"


# =============================================================================
# Additional Edge Cases
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_workflow_with_multiple_subscriptions(db_engine, test_user):
    """
    Verify behavior when deleting a workflow with multiple trigger subscriptions.

    All subscriptions should be cascade-deleted, but external cleanup is missed.
    """
    # Setup: Create workflow with multiple subscriptions
    workflow = await Workflow.create(
        user=test_user,
        name="Multi-Subscription Test",
    )

    subscriptions = []
    for i, trigger_key in enumerate(["webhook.generic", "webhook.supabase.db_changes", "poll.gmail.email_received"]):
        sub = await TriggerSubscription.create(
            user=test_user,
            workflow=workflow,
            trigger_id=f"trigger_{i}",
            trigger_key=trigger_key,
            is_polling="poll." in trigger_key,
            enabled=True,
        )
        subscriptions.append(sub)

    subscription_ids = [s.id for s in subscriptions]

    # Act: Delete workflow
    await workflow.delete()

    # Assert: All subscriptions should be cascade-deleted
    for subscription_id in subscription_ids:
        assert await TriggerSubscription.filter(id=subscription_id).first() is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_supabase_cleanup_failure_does_not_block_subscription_delete(db_engine, test_user):
    """
    Verify that Supabase webhook cleanup failure doesn't block subscription deletion.

    The delete_trigger_subscription() function catches exceptions from
    delete_database_webhook() and logs them (best-effort cleanup).
    """
    from seer.api.workflows.services.triggers import delete_trigger_subscription

    # Setup
    workflow = await Workflow.create(
        user=test_user,
        name="Cleanup Failure Test",
    )

    subscription = await TriggerSubscription.create(
        user=test_user,
        workflow=workflow,
        trigger_id="failing_supabase_trigger",
        trigger_key="webhook.supabase.db_changes",
        is_polling=False,
        enabled=True,
    )
    subscription_id = subscription.id

    # Act: Delete with Supabase cleanup raising an exception
    with patch(
        "seer.api.workflows.services.triggers.delete_database_webhook",
        new_callable=AsyncMock,
        side_effect=Exception("Supabase API error"),
    ) as mock_delete_webhook:
        # Should not raise - exception is caught and logged
        await delete_trigger_subscription(test_user, subscription_id)

        mock_delete_webhook.assert_called_once()

    # Assert: Subscription should still be deleted despite cleanup failure
    assert await TriggerSubscription.filter(id=subscription_id).first() is None
