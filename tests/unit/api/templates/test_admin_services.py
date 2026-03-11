"""
Unit tests for template admin services.

Tests:
- Template CRUD operations
- Category validation
- Slug uniqueness handling
- Publish/unpublish toggle
- Workflow to template conversion helpers
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.templates.models import (
    TemplateCreateRequest,
    TemplateUpdateRequest,
    TemplatePublishRequest,
    RequiredIntegration,
)


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
def test_map_integration_to_provider():
    """Test integration type to provider mapping."""
    from seer.api.templates.admin_services import _map_integration_to_provider

    # Known mappings
    assert _map_integration_to_provider("gmail") == "google"
    assert _map_integration_to_provider("google_calendar") == "google"
    assert _map_integration_to_provider("slack") == "slack"
    assert _map_integration_to_provider("github") == "github"
    assert _map_integration_to_provider("notion") == "notion"

    # Unknown integration defaults to itself
    assert _map_integration_to_provider("custom_tool") == "custom_tool"


@pytest.mark.unit
def test_detect_required_integrations_from_triggers():
    """Test auto-detection of integrations from workflow triggers."""
    from seer.api.templates.admin_services import _detect_required_integrations

    spec = {
        "triggers": [
            {"key": "gmail.new_email", "title": "Gmail Inbox"},
            {"key": "slack.new_message", "title": "Slack Channel"},
        ],
        "nodes": [],
    }

    integrations = _detect_required_integrations(spec)

    assert len(integrations) == 2
    assert {"provider": "google", "integration_type": "gmail", "reason": "Required for gmail.new_email trigger"} in integrations
    assert {"provider": "slack", "integration_type": "slack", "reason": "Required for slack.new_message trigger"} in integrations


@pytest.mark.unit
def test_detect_required_integrations_from_tool_nodes():
    """Test auto-detection of integrations from tool nodes."""
    from seer.api.templates.admin_services import _detect_required_integrations

    spec = {
        "triggers": [],
        "nodes": [
            {"id": "1", "type": "tool", "tool": "gmail.send_email"},
            {"id": "2", "type": "tool", "tool": "github.create_issue"},
            {"id": "3", "type": "agent", "tool": None},  # Non-tool node
        ],
    }

    integrations = _detect_required_integrations(spec)

    assert len(integrations) == 2
    assert {"provider": "google", "integration_type": "gmail", "reason": "Required for gmail.send_email tool"} in integrations
    assert {"provider": "github", "integration_type": "github", "reason": "Required for github.create_issue tool"} in integrations


@pytest.mark.unit
def test_detect_required_integrations_deduplication():
    """Test that integrations are deduplicated by provider."""
    from seer.api.templates.admin_services import _detect_required_integrations

    spec = {
        "triggers": [
            {"key": "gmail.new_email", "title": "Gmail Inbox"},
        ],
        "nodes": [
            {"id": "1", "type": "tool", "tool": "gmail.send_email"},  # Same provider
            {"id": "2", "type": "tool", "tool": "google_calendar.create_event"},  # Same provider (google)
        ],
    }

    integrations = _detect_required_integrations(spec)

    # Should only have one google integration (first one found)
    assert len(integrations) == 1
    assert integrations[0]["provider"] == "google"


@pytest.mark.unit
def test_detect_required_integrations_empty_spec():
    """Test auto-detection with empty spec."""
    from seer.api.templates.admin_services import _detect_required_integrations

    spec = {}
    integrations = _detect_required_integrations(spec)
    assert integrations == []

    spec = {"triggers": [], "nodes": []}
    integrations = _detect_required_integrations(spec)
    assert integrations == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_workflow_for_template_success():
    """Test successful workflow retrieval."""
    from seer.api.templates.admin_services import _get_workflow_for_template

    mock_workflow = MagicMock()
    mock_workflow.id = 123
    mock_workflow.name = "Test Workflow"

    with patch("seer.api.templates.admin_services.Workflow") as MockWorkflow:
        MockWorkflow.filter.return_value.first = AsyncMock(return_value=mock_workflow)

        workflow = await _get_workflow_for_template("wf_123")

        assert workflow.id == 123
        MockWorkflow.filter.assert_called_once_with(id=123)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_workflow_for_template_invalid_format():
    """Test invalid workflow ID format raises 400."""
    from seer.api.templates.admin_services import _get_workflow_for_template
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await _get_workflow_for_template("invalid_format")

    assert exc_info.value.status_code == 400
    assert "Invalid workflow_id format" in exc_info.value.detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_workflow_for_template_not_found():
    """Test workflow not found raises 404."""
    from seer.api.templates.admin_services import _get_workflow_for_template
    from fastapi import HTTPException

    with patch("seer.api.templates.admin_services.Workflow") as MockWorkflow:
        MockWorkflow.filter.return_value.first = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await _get_workflow_for_template("wf_999")

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_workflow_spec_prefers_released():
    """Test that RELEASED version is preferred over DRAFT."""
    from seer.api.templates.admin_services import _get_workflow_spec
    from seer.database import WorkflowVersionStatus

    mock_workflow = MagicMock()
    mock_workflow.id = 1

    mock_released_version = MagicMock()
    mock_released_version.spec = {"version": "2", "nodes": [{"id": "released"}]}

    with patch("seer.api.templates.admin_services.WorkflowVersion") as MockVersion:
        # First call (RELEASED) returns the version
        MockVersion.filter.return_value.first = AsyncMock(return_value=mock_released_version)

        spec = await _get_workflow_spec(mock_workflow)

        assert spec == {"version": "2", "nodes": [{"id": "released"}]}
        # Should have queried for RELEASED status
        MockVersion.filter.assert_called_with(
            workflow=mock_workflow,
            status=WorkflowVersionStatus.RELEASED,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_workflow_spec_falls_back_to_draft():
    """Test fallback to DRAFT when no RELEASED version exists."""
    from seer.api.templates.admin_services import _get_workflow_spec
    from seer.database import WorkflowVersionStatus

    mock_workflow = MagicMock()
    mock_workflow.id = 1

    mock_draft_version = MagicMock()
    mock_draft_version.spec = {"version": "2", "nodes": [{"id": "draft"}]}

    with patch("seer.api.templates.admin_services.WorkflowVersion") as MockVersion:
        # Simulate: RELEASED returns None, DRAFT returns version
        call_count = [0]

        async def mock_filter_first():
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # No RELEASED
            return mock_draft_version  # DRAFT exists

        MockVersion.filter.return_value.first = mock_filter_first

        spec = await _get_workflow_spec(mock_workflow)

        assert spec == {"version": "2", "nodes": [{"id": "draft"}]}
        assert call_count[0] == 2  # Called twice (RELEASED, then DRAFT)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_workflow_spec_no_version_raises():
    """Test that missing versions raises 400."""
    from seer.api.templates.admin_services import _get_workflow_spec
    from fastapi import HTTPException

    mock_workflow = MagicMock()
    mock_workflow.id = 1

    with patch("seer.api.templates.admin_services.WorkflowVersion") as MockVersion:
        MockVersion.filter.return_value.first = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await _get_workflow_spec(mock_workflow)

        assert exc_info.value.status_code == 400
        assert "no spec" in exc_info.value.detail.lower()


# =============================================================================
# Admin Service Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_template_success():
    """Test successful template creation from workflow."""
    from seer.api.templates.admin_services import create_template
    from seer.database import User, TemplateCategory, TemplateSource

    user = MagicMock(spec=User)
    user.id = 1

    payload = TemplateCreateRequest(
        workflow_id="wf_123",
        slug="new-template",
        name="New Template",
        description="A brand new template",
        category="marketing",
        tags=["email", "automation"],
        required_integrations=[
            RequiredIntegration(
                provider="google",
                integration_type="gmail",
                reason="Send emails"
            )
        ],
        is_published=False,
        is_featured=False,
    )

    mock_workflow = MagicMock()
    mock_workflow.id = 123

    mock_version = MagicMock()
    mock_version.spec = {"version": "2", "nodes": [], "edges": []}

    mock_template = MagicMock()
    mock_template.id = 1
    mock_template.slug = "new-template"
    mock_template.name = "New Template"
    mock_template.description = "A brand new template"
    mock_template.category = TemplateCategory.MARKETING
    mock_template.source = TemplateSource.SYSTEM
    mock_template.tags = ["email", "automation"]
    mock_template.icon = None
    mock_template.preview_image_url = None
    mock_template.is_featured = False
    mock_template.is_published = False
    mock_template.usage_count = 0
    mock_template.required_integrations = [
        {"provider": "google", "integration_type": "gmail", "reason": "Send emails"}
    ]
    mock_template.spec = {"version": "2", "nodes": [], "edges": []}
    mock_template.visibility = "private"
    mock_template.created_at = MagicMock()
    mock_template.updated_at = MagicMock()

    with (
        patch("seer.api.templates.admin_services.Workflow") as MockWorkflowModel,
        patch("seer.api.templates.admin_services.WorkflowVersion") as MockVersion,
        patch("seer.api.templates.admin_services.WorkflowTemplate") as MockTemplate,
    ):
        MockWorkflowModel.filter.return_value.first = AsyncMock(return_value=mock_workflow)
        MockVersion.filter.return_value.first = AsyncMock(return_value=mock_version)
        MockTemplate.create = AsyncMock(return_value=mock_template)

        result = await create_template(user, payload)

        assert result.slug == "new-template"
        assert result.name == "New Template"
        MockTemplate.create.assert_called_once()
        # Verify source_workflow_id is passed
        call_kwargs = MockTemplate.create.call_args.kwargs
        assert call_kwargs["source_workflow_id"] == 123


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_template_auto_detects_integrations():
    """Test that integrations are auto-detected when not provided."""
    from seer.api.templates.admin_services import create_template
    from seer.database import User, TemplateCategory, TemplateSource

    user = MagicMock(spec=User)
    user.id = 1

    payload = TemplateCreateRequest(
        workflow_id="wf_123",
        slug="auto-detect-template",
        name="Auto Detect Template",
        description="Template with auto-detected integrations",
        category="marketing",
        # required_integrations is None - should auto-detect
    )

    mock_workflow = MagicMock()
    mock_workflow.id = 123

    mock_version = MagicMock()
    mock_version.spec = {
        "version": "2",
        "triggers": [{"key": "gmail.new_email", "title": "Gmail"}],
        "nodes": [{"id": "1", "type": "tool", "tool": "slack.send_message"}],
    }

    mock_template = MagicMock()
    mock_template.id = 2
    mock_template.slug = "auto-detect-template"
    mock_template.name = "Auto Detect Template"
    mock_template.description = "Template with auto-detected integrations"
    mock_template.category = TemplateCategory.MARKETING
    mock_template.source = TemplateSource.SYSTEM
    mock_template.tags = []
    mock_template.icon = None
    mock_template.preview_image_url = None
    mock_template.is_featured = False
    mock_template.is_published = False
    mock_template.usage_count = 0
    mock_template.required_integrations = [
        {"provider": "google", "integration_type": "gmail", "reason": "Required for gmail.new_email trigger"},
        {"provider": "slack", "integration_type": "slack", "reason": "Required for slack.send_message tool"},
    ]
    mock_template.spec = mock_version.spec
    mock_template.visibility = "private"
    mock_template.created_at = MagicMock()
    mock_template.updated_at = MagicMock()

    with (
        patch("seer.api.templates.admin_services.Workflow") as MockWorkflowModel,
        patch("seer.api.templates.admin_services.WorkflowVersion") as MockVersion,
        patch("seer.api.templates.admin_services.WorkflowTemplate") as MockTemplate,
    ):
        MockWorkflowModel.filter.return_value.first = AsyncMock(return_value=mock_workflow)
        MockVersion.filter.return_value.first = AsyncMock(return_value=mock_version)
        MockTemplate.create = AsyncMock(return_value=mock_template)

        result = await create_template(user, payload)

        # Verify auto-detected integrations were passed
        call_kwargs = MockTemplate.create.call_args.kwargs
        assert len(call_kwargs["required_integrations"]) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_template_invalid_category():
    """Test that invalid category raises error."""
    from seer.api.templates.admin_services import create_template
    from seer.database import User
    from fastapi import HTTPException

    user = MagicMock(spec=User)

    payload = TemplateCreateRequest(
        workflow_id="wf_123",
        slug="test-template",
        name="Test Template",
        description="Test",
        category="invalid_category",  # Invalid
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_template(user, payload)

    assert exc_info.value.status_code == 400
    assert "Invalid category" in exc_info.value.detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_template_duplicate_slug():
    """Test that duplicate slug raises conflict error."""
    from seer.api.templates.admin_services import create_template
    from seer.database import User
    from tortoise.exceptions import IntegrityError
    from fastapi import HTTPException

    user = MagicMock(spec=User)

    payload = TemplateCreateRequest(
        workflow_id="wf_123",
        slug="existing-template",
        name="New Template",
        description="Test",
        category="marketing",
    )

    mock_workflow = MagicMock()
    mock_workflow.id = 123

    mock_version = MagicMock()
    mock_version.spec = {"version": "2", "nodes": []}

    with (
        patch("seer.api.templates.admin_services.Workflow") as MockWorkflowModel,
        patch("seer.api.templates.admin_services.WorkflowVersion") as MockVersion,
        patch("seer.api.templates.admin_services.WorkflowTemplate") as MockTemplate,
    ):
        MockWorkflowModel.filter.return_value.first = AsyncMock(return_value=mock_workflow)
        MockVersion.filter.return_value.first = AsyncMock(return_value=mock_version)
        MockTemplate.create = AsyncMock(
            side_effect=IntegrityError("duplicate key value violates unique constraint on slug")
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_template(user, payload)

        assert exc_info.value.status_code == 409
        assert "already exists" in exc_info.value.detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_template_success():
    """Test successful template update."""
    from seer.api.templates.admin_services import update_template
    from seer.database import TemplateCategory, TemplateSource

    payload = TemplateUpdateRequest(
        name="Updated Name",
        description="Updated description",
    )

    mock_template = MagicMock()
    mock_template.id = 1
    mock_template.slug = "test-template"
    mock_template.name = "Updated Name"
    mock_template.description = "Updated description"
    mock_template.category = TemplateCategory.MARKETING
    mock_template.source = TemplateSource.SYSTEM
    mock_template.tags = []
    mock_template.icon = None
    mock_template.preview_image_url = None
    mock_template.is_featured = False
    mock_template.is_published = True
    mock_template.usage_count = 5
    mock_template.required_integrations = []
    mock_template.spec = {}
    mock_template.visibility = "private"
    mock_template.created_at = MagicMock()
    mock_template.updated_at = MagicMock()

    with patch("seer.api.templates.admin_services.WorkflowTemplate") as MockTemplate:
        MockTemplate.filter.return_value.first = AsyncMock(return_value=mock_template)
        MockTemplate.filter.return_value.update = AsyncMock()
        MockTemplate.get = AsyncMock(return_value=mock_template)

        result = await update_template("test-template", payload)

        assert result.name == "Updated Name"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_template_not_found():
    """Test update of non-existent template raises 404."""
    from seer.api.templates.admin_services import update_template
    from fastapi import HTTPException

    payload = TemplateUpdateRequest(name="New Name")

    with patch("seer.api.templates.admin_services.WorkflowTemplate") as MockTemplate:
        MockTemplate.filter.return_value.first = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await update_template("non-existent", payload)

        assert exc_info.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_template_success():
    """Test successful template deletion."""
    from seer.api.templates.admin_services import delete_template

    mock_template = MagicMock()
    mock_template.delete = AsyncMock()

    with patch("seer.api.templates.admin_services.WorkflowTemplate") as MockTemplate:
        MockTemplate.filter.return_value.first = AsyncMock(return_value=mock_template)

        await delete_template("test-template")

        mock_template.delete.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_template_not_found():
    """Test delete of non-existent template raises 404."""
    from seer.api.templates.admin_services import delete_template
    from fastapi import HTTPException

    with patch("seer.api.templates.admin_services.WorkflowTemplate") as MockTemplate:
        MockTemplate.filter.return_value.first = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await delete_template("non-existent")

        assert exc_info.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_toggle_publish():
    """Test publishing and unpublishing a template."""
    from seer.api.templates.admin_services import toggle_publish
    from seer.database import TemplateCategory, TemplateSource

    mock_template = MagicMock()
    mock_template.id = 1
    mock_template.slug = "test-template"
    mock_template.name = "Test Template"
    mock_template.description = "Test"
    mock_template.category = TemplateCategory.MARKETING
    mock_template.source = TemplateSource.SYSTEM
    mock_template.tags = []
    mock_template.icon = None
    mock_template.preview_image_url = None
    mock_template.is_featured = False
    mock_template.is_published = True  # After toggle
    mock_template.usage_count = 0
    mock_template.required_integrations = []
    mock_template.spec = {}
    mock_template.visibility = "private"
    mock_template.created_at = MagicMock()
    mock_template.updated_at = MagicMock()

    with patch("seer.api.templates.admin_services.WorkflowTemplate") as MockTemplate:
        MockTemplate.filter.return_value.first = AsyncMock(return_value=mock_template)
        MockTemplate.filter.return_value.update = AsyncMock()
        MockTemplate.get = AsyncMock(return_value=mock_template)

        payload = TemplatePublishRequest(is_published=True)
        result = await toggle_publish("test-template", payload)

        assert result.is_published is True


@pytest.mark.unit
def test_to_admin_response_helper():
    """Test _to_admin_response helper function."""
    from seer.api.templates.admin_services import _to_admin_response
    from seer.database import TemplateCategory, TemplateSource
    from datetime import datetime, timezone

    mock_template = MagicMock()
    mock_template.id = 1
    mock_template.slug = "test-template"
    mock_template.name = "Test Template"
    mock_template.description = "A test template"
    mock_template.category = TemplateCategory.MARKETING
    mock_template.source = TemplateSource.SYSTEM
    mock_template.tags = ["test"]
    mock_template.icon = "icon"
    mock_template.preview_image_url = "https://example.com/image.png"
    mock_template.is_featured = True
    mock_template.is_published = False  # Admin can see unpublished
    mock_template.usage_count = 5
    mock_template.required_integrations = [
        {"provider": "google", "integration_type": "gmail", "reason": "Send"}
    ]
    mock_template.spec = {"version": "2"}
    mock_template.visibility = "private"
    mock_template.created_at = datetime.now(timezone.utc)
    mock_template.updated_at = datetime.now(timezone.utc)

    response = _to_admin_response(mock_template)

    assert response.template_id == "tpl_1"
    assert response.slug == "test-template"
    assert response.is_published is False  # Includes publish status
    assert response.is_featured is True
    assert response.category == "marketing"
    assert len(response.required_integrations) == 1
