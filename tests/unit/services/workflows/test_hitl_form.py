"""
Unit tests for HITL Form Service.

Tests the HITLFormService which creates temporary forms for HITL response collection.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestHITLFormFieldMapping:
    """Tests for HITL input to form field mapping."""

    def test_map_text_input(self):
        """Test mapping of text input type."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()
        inputs = [
            {
                "id": "reason",
                "question": "Why is this needed?",
                "input_type": "text",
                "required": True,
                "placeholder": "Enter your reason",
            }
        ]

        fields = service._map_hitl_inputs_to_form_fields(inputs)

        assert len(fields) == 1
        assert fields[0]["name"] == "reason"
        assert fields[0]["displayLabel"] == "Why is this needed?"
        assert fields[0]["type"] == "text"
        assert fields[0]["required"] is True
        assert fields[0]["placeholder"] == "Enter your reason"

    def test_map_number_input(self):
        """Test mapping of number input type."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()
        inputs = [
            {
                "id": "amount",
                "question": "Enter the amount",
                "input_type": "number",
                "required": True,
                "default_value": 100,
            }
        ]

        fields = service._map_hitl_inputs_to_form_fields(inputs)

        assert len(fields) == 1
        assert fields[0]["name"] == "amount"
        assert fields[0]["type"] == "number"
        assert fields[0]["defaultValue"] == 100

    def test_map_boolean_input(self):
        """Test mapping of boolean input type to checkbox."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()
        inputs = [
            {
                "id": "confirm",
                "question": "Do you confirm?",
                "input_type": "boolean",
                "required": False,
            }
        ]

        fields = service._map_hitl_inputs_to_form_fields(inputs)

        assert len(fields) == 1
        assert fields[0]["type"] == "checkbox"
        assert fields[0]["required"] is False

    def test_map_single_choice_input(self):
        """Test mapping of single choice input with options."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()
        inputs = [
            {
                "id": "decision",
                "question": "Approve or reject?",
                "input_type": "single_choice",
                "required": True,
                "options": [
                    {"value": "approve", "label": "Approve"},
                    {"value": "reject", "label": "Reject"},
                ],
            }
        ]

        fields = service._map_hitl_inputs_to_form_fields(inputs)

        assert len(fields) == 1
        assert fields[0]["type"] == "select"
        assert len(fields[0]["options"]) == 2
        assert fields[0]["options"][0]["value"] == "approve"
        assert fields[0]["options"][0]["label"] == "Approve"

    def test_map_multi_choice_input(self):
        """Test mapping of multi choice input type."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()
        inputs = [
            {
                "id": "features",
                "question": "Select features to enable",
                "input_type": "multi_choice",
                "required": True,
                "options": [
                    {"value": "feature_a", "label": "Feature A"},
                    {"value": "feature_b", "label": "Feature B"},
                    {"value": "feature_c", "label": "Feature C"},
                ],
            }
        ]

        fields = service._map_hitl_inputs_to_form_fields(inputs)

        assert len(fields) == 1
        assert fields[0]["type"] == "multiselect"
        assert len(fields[0]["options"]) == 3

    def test_map_multiple_inputs(self):
        """Test mapping of multiple mixed input types."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()
        inputs = [
            {"id": "name", "question": "Your name", "input_type": "text", "required": True},
            {"id": "age", "question": "Your age", "input_type": "number", "required": False},
            {"id": "agree", "question": "Agree to terms?", "input_type": "boolean", "required": True},
        ]

        fields = service._map_hitl_inputs_to_form_fields(inputs)

        assert len(fields) == 3
        assert fields[0]["type"] == "text"
        assert fields[1]["type"] == "number"
        assert fields[2]["type"] == "checkbox"

    def test_map_unknown_type_defaults_to_text(self):
        """Test that unknown input types default to text."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()
        inputs = [
            {
                "id": "unknown",
                "question": "Unknown type",
                "input_type": "custom_type",
                "required": True,
            }
        ]

        fields = service._map_hitl_inputs_to_form_fields(inputs)

        assert fields[0]["type"] == "text"


@pytest.mark.unit
class TestHITLFormCreation:
    """Tests for HITL form creation."""

    @pytest.mark.asyncio
    async def test_create_hitl_form_generates_correct_suffix(self):
        """Test that form suffix follows hitl-{run_id}-{node_id} pattern."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()

        # Mock workflow run
        mock_run = MagicMock()
        mock_run.run_id = "run_123"
        mock_run.fetch_related = AsyncMock()
        mock_run.user = MagicMock()
        mock_run.workflow = MagicMock()

        interrupt_data = {
            "type": "hitl",
            "node_id": "approval_node",
            "title": "Approval Required",
            "inputs": [],
        }

        with patch("seer.services.workflows.hitl_form.TriggerSubscription") as mock_ts:
            mock_ts.create = AsyncMock(return_value=MagicMock(id=1))

            subscription, form_url = await service.create_hitl_form(mock_run, interrupt_data)

            # Verify suffix pattern
            create_call = mock_ts.create.call_args
            assert create_call.kwargs["form_suffix"] == "hitl-run_123-approval_node"
            assert "run_123" in form_url
            assert "approval_node" in form_url

    @pytest.mark.asyncio
    async def test_create_hitl_form_stores_metadata_in_config(self):
        """Test that HITL metadata is stored in form_config."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()

        mock_run = MagicMock()
        mock_run.run_id = "run_456"
        mock_run.fetch_related = AsyncMock()
        mock_run.user = MagicMock()
        mock_run.workflow = MagicMock()

        interrupt_data = {
            "type": "hitl",
            "node_id": "review_node",
            "title": "Review Request",
            "description": "Please review this item",
            "display": [{"label": "Item", "value": "Widget"}],
            "inputs": [],
        }

        with patch("seer.services.workflows.hitl_form.TriggerSubscription") as mock_ts:
            mock_ts.create = AsyncMock(return_value=MagicMock(id=1))

            await service.create_hitl_form(mock_run, interrupt_data)

            create_call = mock_ts.create.call_args
            form_config = create_call.kwargs["form_config"]

            assert form_config["_hitl_run_id"] == "run_456"
            assert form_config["_hitl_node_id"] == "review_node"
            assert form_config["title"] == "Review Request"
            assert form_config["description"] == "Please review this item"
            assert len(form_config["_hitl_display"]) == 1

    @pytest.mark.asyncio
    async def test_create_hitl_form_uses_form_hitl_trigger_key(self):
        """Test that HITL forms use the form.hitl trigger key."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()

        mock_run = MagicMock()
        mock_run.run_id = "run_789"
        mock_run.fetch_related = AsyncMock()
        mock_run.user = MagicMock()
        mock_run.workflow = MagicMock()

        interrupt_data = {
            "type": "hitl",
            "node_id": "test_node",
            "title": "Test",
            "inputs": [],
        }

        with patch("seer.services.workflows.hitl_form.TriggerSubscription") as mock_ts:
            mock_ts.create = AsyncMock(return_value=MagicMock(id=1))

            await service.create_hitl_form(mock_run, interrupt_data)

            create_call = mock_ts.create.call_args
            assert create_call.kwargs["trigger_key"] == "form.hitl"


@pytest.mark.unit
class TestHITLFormDisable:
    """Tests for disabling HITL forms after use."""

    @pytest.mark.asyncio
    async def test_disable_hitl_form_updates_enabled_to_false(self):
        """Test that disabling a form sets enabled=False."""
        from seer.services.workflows.hitl_form import HITLFormService

        service = HITLFormService()

        with patch("seer.services.workflows.hitl_form.TriggerSubscription") as mock_ts:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            mock_ts.filter.return_value = mock_filter

            await service.disable_hitl_form(subscription_id=42)

            mock_ts.filter.assert_called_once_with(id=42)
            mock_filter.update.assert_called_once_with(enabled=False)
