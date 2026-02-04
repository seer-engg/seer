"""
Unit tests for core.triggers.events module.

Tests trigger event envelope creation and persistence.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# TriggerEventEnvelopeInput Tests
# =============================================================================


@pytest.mark.unit
class TestTriggerEventEnvelopeInput:
    """Tests for TriggerEventEnvelopeInput dataclass."""

    def test_envelope_input_creation(self):
        """Test creating TriggerEventEnvelopeInput with all fields."""
        from seer.core.triggers.events import TriggerEventEnvelopeInput

        input_data = TriggerEventEnvelopeInput(
            trigger_id="trigger_123",
            trigger_key="webhook.generic",
            title="Generic Webhook",
            provider="generic",
            provider_connection_id=456,
            payload={"message": "test"},
            raw={"original": "data"},
            occurred_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        assert input_data.trigger_id == "trigger_123"
        assert input_data.trigger_key == "webhook.generic"
        assert input_data.title == "Generic Webhook"
        assert input_data.provider == "generic"
        assert input_data.provider_connection_id == 456
        assert input_data.payload == {"message": "test"}
        assert input_data.raw == {"original": "data"}

    def test_envelope_input_optional_fields(self):
        """Test TriggerEventEnvelopeInput with optional fields as None."""
        from seer.core.triggers.events import TriggerEventEnvelopeInput

        input_data = TriggerEventEnvelopeInput(
            trigger_id="trigger_abc",
            trigger_key="schedule.cron",
            title="Scheduled",
            provider="cron",
            provider_connection_id=None,
            payload={},
            raw=None,
        )

        assert input_data.provider_connection_id is None
        assert input_data.raw is None
        assert input_data.occurred_at is None

    def test_envelope_input_immutable(self):
        """Test TriggerEventEnvelopeInput is immutable (frozen)."""
        from seer.core.triggers.events import TriggerEventEnvelopeInput

        input_data = TriggerEventEnvelopeInput(
            trigger_id="test",
            trigger_key="test.key",
            title="Test",
            provider="test",
            provider_connection_id=None,
            payload={},
            raw=None,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            input_data.trigger_id = "modified"


# =============================================================================
# Build Event Envelope Tests
# =============================================================================


@pytest.mark.unit
class TestBuildEventEnvelope:
    """Tests for build_event_envelope function."""

    def test_build_envelope_all_fields(self):
        """Test building envelope with all fields populated."""
        from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope

        occurred = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        input_data = TriggerEventEnvelopeInput(
            trigger_id="trg_001",
            trigger_key="github.push",
            title="GitHub Push",
            provider="github",
            provider_connection_id=789,
            payload={"branch": "main", "commits": 3},
            raw={"full": "webhook_data"},
            occurred_at=occurred,
        )

        with patch("seer.core.triggers.events._utcnow") as mock_utcnow:
            mock_utcnow.return_value = datetime(2024, 1, 15, 10, 30, 5, tzinfo=timezone.utc)
            envelope = build_event_envelope(input_data)

        assert envelope["trigger_id"] == "trg_001"
        assert envelope["trigger_key"] == "github.push"
        assert envelope["title"] == "GitHub Push"
        assert envelope["provider"] == "github"
        assert envelope["account_id"] == 789
        assert envelope["occurred_at"] == "2024-01-15T10:30:00+00:00"
        assert envelope["received_at"] == "2024-01-15T10:30:05+00:00"
        assert envelope["data"] == {"branch": "main", "commits": 3}
        assert envelope["raw"] == {"full": "webhook_data"}
        assert envelope["id"].startswith("evt_")

    def test_build_envelope_generates_unique_id(self):
        """Test each envelope gets a unique ID."""
        from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope

        input_data = TriggerEventEnvelopeInput(
            trigger_id="test",
            trigger_key="test.trigger",
            title="Test",
            provider="test",
            provider_connection_id=None,
            payload={},
            raw=None,
        )

        envelope1 = build_event_envelope(input_data)
        envelope2 = build_event_envelope(input_data)

        assert envelope1["id"] != envelope2["id"]
        assert envelope1["id"].startswith("evt_")
        assert envelope2["id"].startswith("evt_")

    def test_build_envelope_uses_current_time_when_no_occurred_at(self):
        """Test envelope uses current time when occurred_at not provided."""
        from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope

        input_data = TriggerEventEnvelopeInput(
            trigger_id="test",
            trigger_key="test.trigger",
            title="Test",
            provider="test",
            provider_connection_id=None,
            payload={},
            raw=None,
            occurred_at=None,
        )

        with patch("seer.core.triggers.events._utcnow") as mock_utcnow:
            now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
            mock_utcnow.return_value = now
            envelope = build_event_envelope(input_data)

        assert envelope["occurred_at"] == "2024-06-01T12:00:00+00:00"

    def test_build_envelope_data_field_contains_payload(self):
        """Test envelope's data field contains the payload."""
        from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope

        payload = {"user": {"id": 123, "email": "test@example.com"}, "action": "created"}
        input_data = TriggerEventEnvelopeInput(
            trigger_id="test",
            trigger_key="webhook.user",
            title="User Webhook",
            provider="app",
            provider_connection_id=None,
            payload=payload,
            raw=None,
        )

        envelope = build_event_envelope(input_data)

        assert envelope["data"] == payload
        assert envelope["data"]["user"]["email"] == "test@example.com"


# =============================================================================
# Persist Event Tests
# =============================================================================


@pytest.mark.unit
class TestPersistEvent:
    """Tests for persist_event function."""

    @pytest.fixture
    def mock_subscription(self):
        """Create a mock trigger subscription."""
        subscription = MagicMock()
        subscription.id = 100
        subscription.trigger_key = "webhook.generic"
        subscription.provider_connection_id = 50
        return subscription

    @pytest.fixture
    def sample_envelope(self):
        """Create a sample event envelope."""
        return {
            "id": "evt_abc123",
            "trigger_id": "trg_001",
            "trigger_key": "webhook.generic",
            "title": "Webhook",
            "provider": "generic",
            "account_id": 50,
            "occurred_at": "2024-01-15T10:00:00+00:00",
            "received_at": "2024-01-15T10:00:01+00:00",
            "data": {"message": "test"},
            "raw": None,
        }

    @pytest.mark.asyncio
    async def test_persist_event_creates_record(self, mock_subscription, sample_envelope):
        """Test persist_event creates a new TriggerEvent record."""
        from seer.core.triggers.events import persist_event

        mock_event = MagicMock()
        mock_event.id = 999

        with patch("seer.core.triggers.events.TriggerEvent") as MockTriggerEvent:
            MockTriggerEvent.create = AsyncMock(return_value=mock_event)

            event, created = await persist_event(
                subscription=mock_subscription,
                envelope=sample_envelope,
                provider_event_id="provider_evt_123",
                event_hash=None,
                raw={"original": "data"},
            )

        assert event == mock_event
        assert created is True
        MockTriggerEvent.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_event_deduplicates_by_provider_event_id(self, mock_subscription, sample_envelope):
        """Test persist_event handles duplicate by provider_event_id."""
        from tortoise.exceptions import IntegrityError
        from seer.core.triggers.events import persist_event

        existing_event = MagicMock()
        existing_event.id = 888

        with patch("seer.core.triggers.events.TriggerEvent") as MockTriggerEvent:
            MockTriggerEvent.create = AsyncMock(side_effect=IntegrityError("Duplicate"))
            MockTriggerEvent.get = AsyncMock(return_value=existing_event)

            event, created = await persist_event(
                subscription=mock_subscription,
                envelope=sample_envelope,
                provider_event_id="duplicate_evt_id",
                event_hash=None,
                raw=None,
            )

        assert event == existing_event
        assert created is False

    @pytest.mark.asyncio
    async def test_persist_event_deduplicates_by_event_hash(self, mock_subscription, sample_envelope):
        """Test persist_event handles duplicate by event_hash."""
        from tortoise.exceptions import IntegrityError
        from seer.core.triggers.events import persist_event

        existing_event = MagicMock()
        existing_event.id = 777

        with patch("seer.core.triggers.events.TriggerEvent") as MockTriggerEvent:
            MockTriggerEvent.create = AsyncMock(side_effect=IntegrityError("Duplicate"))
            MockTriggerEvent.get = AsyncMock(return_value=existing_event)

            event, created = await persist_event(
                subscription=mock_subscription,
                envelope=sample_envelope,
                provider_event_id=None,
                event_hash="hash_abc123",
                raw=None,
            )

        assert event == existing_event
        assert created is False

    @pytest.mark.asyncio
    async def test_persist_event_parses_occurred_at(self, mock_subscription):
        """Test persist_event parses occurred_at from envelope."""
        from seer.core.triggers.events import persist_event

        envelope = {
            "id": "evt_test",
            "occurred_at": "2024-03-20T15:30:00+00:00",
            "data": {},
        }

        mock_event = MagicMock()

        with patch("seer.core.triggers.events.TriggerEvent") as MockTriggerEvent:
            MockTriggerEvent.create = AsyncMock(return_value=mock_event)

            await persist_event(
                subscription=mock_subscription,
                envelope=envelope,
                provider_event_id=None,
                event_hash=None,
                raw=None,
            )

        call_kwargs = MockTriggerEvent.create.call_args[1]
        assert call_kwargs["occurred_at"].year == 2024
        assert call_kwargs["occurred_at"].month == 3
        assert call_kwargs["occurred_at"].day == 20

    @pytest.mark.asyncio
    async def test_persist_event_uses_current_time_for_invalid_occurred_at(self, mock_subscription):
        """Test persist_event uses current time when occurred_at is invalid."""
        from seer.core.triggers.events import persist_event

        envelope = {
            "id": "evt_test",
            "occurred_at": 12345,  # Not a string
            "data": {},
        }

        mock_event = MagicMock()

        with patch("seer.core.triggers.events.TriggerEvent") as MockTriggerEvent:
            MockTriggerEvent.create = AsyncMock(return_value=mock_event)
            with patch("seer.core.triggers.events._utcnow") as mock_utcnow:
                now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
                mock_utcnow.return_value = now

                await persist_event(
                    subscription=mock_subscription,
                    envelope=envelope,
                    provider_event_id=None,
                    event_hash=None,
                    raw=None,
                )

        call_kwargs = MockTriggerEvent.create.call_args[1]
        assert call_kwargs["occurred_at"] == now

    @pytest.mark.asyncio
    async def test_persist_event_raises_on_non_dedupe_integrity_error(self, mock_subscription, sample_envelope):
        """Test persist_event re-raises IntegrityError when no dedupe key."""
        from tortoise.exceptions import IntegrityError
        from seer.core.triggers.events import persist_event

        with patch("seer.core.triggers.events.TriggerEvent") as MockTriggerEvent:
            MockTriggerEvent.create = AsyncMock(side_effect=IntegrityError("Other constraint"))

            with pytest.raises(IntegrityError):
                await persist_event(
                    subscription=mock_subscription,
                    envelope=sample_envelope,
                    provider_event_id=None,  # No dedupe key
                    event_hash=None,  # No dedupe key
                    raw=None,
                )

    @pytest.mark.asyncio
    async def test_persist_event_allows_same_event_for_different_subscriptions(self, sample_envelope):
        """Test that multiple subscriptions can create events for the same provider_event_id.

        This verifies the fix for the multi-workflow trigger bug where only
        the first subscription to process an event would trigger.
        """
        from seer.core.triggers.events import persist_event

        # Create two different subscriptions on the same account
        subscription_a = MagicMock()
        subscription_a.id = 100
        subscription_a.trigger_key = "gmail.email_received"
        subscription_a.provider_connection_id = 50  # Same Gmail account

        subscription_b = MagicMock()
        subscription_b.id = 200  # Different subscription
        subscription_b.trigger_key = "gmail.email_received"
        subscription_b.provider_connection_id = 50  # Same Gmail account

        mock_event_a = MagicMock()
        mock_event_a.id = 1
        mock_event_b = MagicMock()
        mock_event_b.id = 2

        with patch("seer.core.triggers.events.TriggerEvent") as MockTriggerEvent:
            # Both creates should succeed (no IntegrityError)
            MockTriggerEvent.create = AsyncMock(side_effect=[mock_event_a, mock_event_b])

            # Subscription A creates event for email msg_123
            event_a, created_a = await persist_event(
                subscription=subscription_a,
                envelope=sample_envelope,
                provider_event_id="gmail_msg_123",
                event_hash=None,
                raw=None,
            )

            # Subscription B should also be able to create event for the SAME email
            event_b, created_b = await persist_event(
                subscription=subscription_b,
                envelope=sample_envelope,
                provider_event_id="gmail_msg_123",  # Same email
                event_hash=None,
                raw=None,
            )

        # Both should report created=True
        assert created_a is True
        assert created_b is True
        # Both should be different events
        assert event_a.id != event_b.id
        # TriggerEvent.create should have been called twice
        assert MockTriggerEvent.create.call_count == 2

    @pytest.mark.asyncio
    async def test_persist_event_dedup_includes_subscription_id(self, mock_subscription, sample_envelope):
        """Test that dedup lookup includes subscription_id in the filter."""
        from tortoise.exceptions import IntegrityError
        from seer.core.triggers.events import persist_event

        existing_event = MagicMock()
        existing_event.id = 555

        with patch("seer.core.triggers.events.TriggerEvent") as MockTriggerEvent:
            MockTriggerEvent.create = AsyncMock(side_effect=IntegrityError("Duplicate"))
            MockTriggerEvent.get = AsyncMock(return_value=existing_event)

            await persist_event(
                subscription=mock_subscription,
                envelope=sample_envelope,
                provider_event_id="dup_event_id",
                event_hash=None,
                raw=None,
            )

        # Verify that the get call includes subscription_id
        call_kwargs = MockTriggerEvent.get.call_args[1]
        assert call_kwargs["subscription_id"] == mock_subscription.id
        assert call_kwargs["trigger_key"] == mock_subscription.trigger_key
        assert call_kwargs["provider_connection_id"] == mock_subscription.provider_connection_id
        assert call_kwargs["provider_event_id"] == "dup_event_id"
