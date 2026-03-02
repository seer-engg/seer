"""
Integration tests for workflow run API (run_saved_workflow).

Tests:
- Running draft workflows with triggers requires trigger_event_override
- Running workflows without triggers
- Running with trigger_event_override
"""
import hashlib
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from seer.api.workflows.services.execution import run_saved_workflow
from seer.api.workflows import models as api_models
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
async def test_run_draft_workflow_with_triggers_requires_event_override(db_engine, test_user):
    """Test that draft workflows with triggers require trigger_event_override."""
    # Create workflow
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    spec_dict = {
        "version": "2",
        "nodes": [{"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {}}],
        "edges": [{"source": "trigger_1", "target": "node1", "type": "trigger"}],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
                    "sample_event": {"data": {"test": "data"}},
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

    # Mock _validate_workflow_spec to skip compiler validation (test.tool not registered)
    with patch("seer.api.workflows.services.execution.validate_workflow_spec", new_callable=AsyncMock):
        # Mock sync_trigger_subscriptions to avoid actual sync
        with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
            mock_sync.return_value = None

            # Run workflow without trigger_event_override - should fail
            payload = api_models.RunFromWorkflowRequest(
                inputs={},
                config={},
            )

            with pytest.raises(HTTPException) as exc_info:
                await run_saved_workflow(test_user, workflow.workflow_id, payload)

            # Verify 400 error about requiring trigger event
            assert exc_info.value.status_code == 400
            assert "Trigger event required" in str(exc_info.value.detail)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_draft_workflow_with_trigger_event_override(db_engine, test_user):
    """Test running draft workflow with trigger_event_override succeeds."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    spec_dict = {
        "version": "2",
        "nodes": [{"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {}}],
        "edges": [{"source": "trigger_1", "target": "node1", "type": "trigger"}],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
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

    # Mock _validate_workflow_spec to skip compiler validation (test.tool not registered)
    with patch("seer.api.workflows.services.execution.validate_workflow_spec", new_callable=AsyncMock):
        with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
            mock_sync.return_value = None

            with patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task:
                mock_task.kiq = AsyncMock()

                # Run workflow WITH trigger_event_override - should succeed
                payload = api_models.RunFromWorkflowRequest(
                    inputs={},
                    config={},
                    trigger_event_override={
                        "trigger_key": "webhook.custom",
                        "data": {"message": "Real event data"},
                    },
                )

                result = await run_saved_workflow(test_user, workflow.workflow_id, payload)

                # Verify single RunResponse
                assert isinstance(result, api_models.RunResponse)
                assert result.run_id is not None

                # Verify trigger envelope was passed to task
                assert mock_task.kiq.called
                call_kwargs = mock_task.kiq.call_args[1]
                assert "trigger_envelope" in call_kwargs
                assert call_kwargs["trigger_envelope"]["data"] == {"message": "Real event data"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_draft_workflow_with_multiple_triggers_requires_trigger_id(db_engine, test_user):
    """Test running draft workflow with multiple triggers requires trigger_id when using override."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    spec_dict = {
        "version": "2",
        "nodes": [{"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {}}],
        "edges": [
            {"source": "trigger_1", "target": "node1", "type": "trigger"},
            {"source": "trigger_2", "target": "node1", "type": "trigger"},
        ],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {"requires_connection": False},
                "filters": {},
                "provider_config": {},
                "ui_meta": {"title": "Trigger One"},
            },
            {
                "id": "trigger_2",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {"requires_connection": False},
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

    # Mock _validate_workflow_spec to skip compiler validation (test.tool not registered)
    with patch("seer.api.workflows.services.execution.validate_workflow_spec", new_callable=AsyncMock):
        with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
            mock_sync.return_value = None

            # Run without trigger_event_override - should fail requiring event
            payload = api_models.RunFromWorkflowRequest(
                inputs={},
                config={},
            )

            with pytest.raises(HTTPException) as exc_info:
                await run_saved_workflow(test_user, workflow.workflow_id, payload)

            assert exc_info.value.status_code == 400
            assert "Trigger event required" in str(exc_info.value.detail)


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

        # Verify single RunResponse
        assert isinstance(result, api_models.RunResponse)
        assert result.run_id is not None

        # Verify task was enqueued once without trigger_envelope
        assert mock_task.kiq.call_count == 1
        call_kwargs = mock_task.kiq.call_args[1]
        assert "trigger_envelope" not in call_kwargs or call_kwargs.get("trigger_envelope") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_workflow_with_triggers_requires_trigger_event(db_engine, test_user):
    """Test that workflows with triggers require trigger_event_override."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    spec_dict = {
        "version": "2",
        "nodes": [{"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {}}],
        "edges": [{"source": "trigger_1", "target": "node1", "type": "trigger"}],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
                    "requires_connection": False,
                },
                "filters": {},
                "provider_config": {},
                "ui_meta": {"title": "Test Trigger"},
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

    # Mock _validate_workflow_spec to skip compiler validation (test.tool not registered)
    with patch("seer.api.workflows.services.execution.validate_workflow_spec", new_callable=AsyncMock):
        with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
            mock_sync.return_value = None

            payload = api_models.RunFromWorkflowRequest(
                inputs={},
                config={},
            )

            # Should raise error requiring trigger event
            with pytest.raises(HTTPException) as exc_info:
                await run_saved_workflow(test_user, workflow.workflow_id, payload)

            # Verify error message
            assert exc_info.value.status_code == 400
            assert "Trigger event required" in str(exc_info.value.detail)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_workflow_with_trigger_override_uses_fallback_title(db_engine, test_user):
    """Test that trigger key is used as fallback title in envelope when ui_meta title is missing."""
    workflow = await Workflow.create(
        user=test_user,
        name="Test Workflow",
        description="Test",
    )

    spec_dict = {
        "version": "2",
        "nodes": [{"id": "node1", "type": "tool", "tool": "test.tool", "inputs": {}}],
        "edges": [{"source": "trigger_1", "target": "node1", "type": "trigger"}],
        "triggers": [
            {
                "id": "trigger_1",
                "key": "webhook.custom",
                "mode": "webhook",
                "event_schema": {},
                "meta": {
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

    # Mock _validate_workflow_spec to skip compiler validation (test.tool not registered)
    with patch("seer.api.workflows.services.execution.validate_workflow_spec", new_callable=AsyncMock):
        with patch("seer.api.workflows.services.triggers.sync_trigger_subscriptions") as mock_sync:
            mock_sync.return_value = None

            with patch("seer.api.workflows.services.execution.workflow_execution_task") as mock_task:
                mock_task.kiq = AsyncMock()

                # Provide trigger_event_override without title
                payload = api_models.RunFromWorkflowRequest(
                    inputs={},
                    config={},
                    trigger_event_override={
                        "trigger_key": "webhook.custom",
                        "data": {"test": "data"},
                    },
                )

                result = await run_saved_workflow(test_user, workflow.workflow_id, payload)

                # Verify run was created
                assert isinstance(result, api_models.RunResponse)

                # Verify trigger_key is used as fallback title in envelope
                call_kwargs = mock_task.kiq.call_args[1]
                assert call_kwargs["trigger_envelope"]["title"] == "webhook.custom"
