"""
Integration tests for workflow run API (run_saved_workflow).

Tests:
- Running draft workflows with triggers from spec
- Running workflows without triggers
- Trigger envelope generation from TriggerSpec
"""
import hashlib
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from seer.api.workflows.services.execution import run_saved_workflow
from seer.api.workflows import models as api_models
from seer.core.schema.models import TriggerMetadata, TriggerSpec, WorkflowSpec
from seer.database.workflow_models import (
    Workflow,
    WorkflowVersion,
    WorkflowVersionStatus,
)


def _hash_spec(spec_dict: Dict[str, Any]) -> str:
    """Generate hash for workflow spec."""
    serialized = json.dumps(spec_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


# =============================================================================
# Draft Workflow with Triggers Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_draft_workflow_with_triggers_from_spec(db_engine, test_user):
    """Test that draft workflows read triggers from spec, not TriggerSubscription."""
    # Create workflow
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    # Create draft version with triggers in spec
    sample_event = {
        "data": {"message": "Test message", "from": "test@example.com"},
        "raw": {"raw_data": "example"},
    }

    spec_dict = {
        "version": "2",
        "nodes": [],
        "edges": [],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
                    "sample_event": sample_event,
                    "requires_connection": False,
                },
                "filters": {},
                "provider_config": {},
                "ui_meta": {"title": "Test Webhook Trigger"},
            }
        ],
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    # Mock sync_trigger_subscriptions to avoid actual sync
    with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
        mock_sync.return_value = None

        # Mock workflow_execution_task to avoid actual task execution
        with patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task:
            mock_task.kiq = AsyncMock()

            # Mock trigger_registry to return a definition
            with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry:
                mock_definition = AsyncMock()
                mock_definition.provider = "webhook"
                mock_definition.meta.sample_event = sample_event
                mock_registry.maybe_get.return_value = mock_definition

                # Run workflow
                payload = api_models.RunFromWorkflowRequest(
                    inputs={},
                    config={},
                )

                result = await run_saved_workflow(test_user, workflow.workflow_id, payload)

                # Verify MultiRunResponse with one run per trigger
                assert isinstance(result, api_models.MultiRunResponse)
                assert len(result.runs) == 1
                assert result.runs[0].trigger_title == "Test Webhook Trigger"

                # Verify trigger envelope was passed to task
                assert mock_task.kiq.called
                call_kwargs = mock_task.kiq.call_args[1]
                assert "trigger_envelope" in call_kwargs
                assert call_kwargs["trigger_envelope"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_draft_workflow_with_multiple_triggers(db_engine, test_user):
    """Test running draft workflow with multiple triggers creates multiple runs."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    sample_event = {
        "data": {"test": "data"},
    }

    spec_dict = {
        "version": "2",
        "nodes": [],
        "edges": [],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
                    "sample_event": sample_event,
                    "requires_connection": False,
                },
                "filters": {},
                "provider_config": {},
                "ui_meta": {"title": "Trigger One"},
            },
            {
                "id": "trigger_2",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
                    "sample_event": sample_event,
                    "requires_connection": False,
                },
                "filters": {},
                "provider_config": {},
                "ui_meta": {"title": "Trigger Two"},
            },
        ],
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
        mock_sync.return_value = None

        with patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task:
            mock_task.kiq = AsyncMock()

            with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry:
                mock_definition = AsyncMock()
                mock_definition.provider = "webhook"
                mock_definition.meta.sample_event = sample_event
                mock_registry.maybe_get.return_value = mock_definition

                payload = api_models.RunFromWorkflowRequest(
                    inputs={},
                    config={},
                )

                result = await run_saved_workflow(test_user, workflow.workflow_id, payload)

                # Verify two runs created
                assert isinstance(result, api_models.MultiRunResponse)
                assert len(result.runs) == 2
                assert result.runs[0].trigger_title == "Trigger One"
                assert result.runs[1].trigger_title == "Trigger Two"

                # Verify task was enqueued twice
                assert mock_task.kiq.call_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_workflow_without_triggers(db_engine, test_user):
    """Test running workflow without triggers creates single manual run."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    spec_dict = {
        "version": "2",
        "nodes": [],
        "edges": [],
        "triggers": [],  # No triggers
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    with patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task:
        mock_task.kiq = AsyncMock()

        payload = api_models.RunFromWorkflowRequest(
            inputs={},
            config={},
        )

        result = await run_saved_workflow(test_user, workflow.workflow_id, payload)

        # Verify single RunResponse (not MultiRunResponse)
        assert isinstance(result, api_models.RunResponse)
        assert result.run_id is not None

        # Verify task was enqueued once without trigger_envelope
        assert mock_task.kiq.call_count == 1
        call_kwargs = mock_task.kiq.call_args[1]
        assert "trigger_envelope" not in call_kwargs or call_kwargs.get("trigger_envelope") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_workflow_skips_triggers_without_sample_event(db_engine, test_user):
    """Test that triggers without sample events are skipped."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    spec_dict = {
        "version": "2",
        "nodes": [],
        "edges": [],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
                    "sample_event": None,  # No sample event
                    "requires_connection": False,
                },
                "filters": {},
                "provider_config": {},
                "ui_meta": {"title": "No Sample Trigger"},
            }
        ],
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
        mock_sync.return_value = None

        with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry:
            mock_definition = AsyncMock()
            mock_definition.provider = "webhook"
            mock_definition.meta.sample_event = None  # No sample
            mock_registry.maybe_get.return_value = mock_definition

            payload = api_models.RunFromWorkflowRequest(
                inputs={},
                config={},
            )

            # Should raise error since no valid triggers
            with pytest.raises(Exception) as exc_info:
                await run_saved_workflow(test_user, workflow.workflow_id, payload)

            # Verify error message
            assert "No valid triggers" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_workflow_uses_trigger_key_as_fallback_title(db_engine, test_user):
    """Test that trigger key is used as fallback when title is missing."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    sample_event = {"data": {"test": "data"}}

    spec_dict = {
        "version": "2",
        "nodes": [],
        "edges": [],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
                    "sample_event": sample_event,
                    "requires_connection": False,
                },
                "filters": {},
                "provider_config": {},
                "ui_meta": {},  # No title in ui_meta
            }
        ],
    }

    version = await WorkflowVersion.create(
        workflow=workflow,
        version_number=1,
        status=WorkflowVersionStatus.DRAFT,
        spec=spec_dict,
        spec_hash=_hash_spec(spec_dict),
    )

    with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
        mock_sync.return_value = None

        with patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task:
            mock_task.kiq = AsyncMock()

            with patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry:
                mock_definition = AsyncMock()
                mock_definition.provider = "webhook"
                mock_definition.meta.sample_event = sample_event
                mock_registry.maybe_get.return_value = mock_definition

                payload = api_models.RunFromWorkflowRequest(
                    inputs={},
                    config={},
                )

                result = await run_saved_workflow(test_user, workflow.workflow_id, payload)

                # Verify trigger_key is used as fallback title
                assert isinstance(result, api_models.MultiRunResponse)
                assert result.runs[0].trigger_title == "webhook.custom"
