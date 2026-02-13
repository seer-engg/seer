"""
Unit tests for MCP workflow CRUD tools.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from seer.tools.workflow_validation import ValidationResult, ValidationError


@pytest.mark.unit
class TestValidateAndUpsertWorkflow:
    """Tests for validate_and_upsert_workflow MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    @patch("seer.mcp.tools.workflows.run_full_validation")
    async def test_upsert_validates_spec_first(self, mock_validation, mock_get_user):
        """Test that validate_and_upsert_workflow validates spec before creating."""
        mock_get_user.return_value = MagicMock()
        mock_validation.return_value = ValidationResult(
            success=False,
            error=ValidationError("schema_validation", "Invalid spec", "Check schema"),
        )

        from seer.mcp.tools.workflows import validate_and_upsert_workflow
        result = await validate_and_upsert_workflow.fn("Test Workflow", {"invalid": "spec"})
        data = json.loads(result)

        assert data["status"] == "error"
        assert data["error_type"] == "schema_validation"

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    @patch("seer.mcp.tools.workflows.run_full_validation")
    async def test_upsert_returns_reference_validation_error(self, mock_validation, mock_get_user):
        """Test that reference validation errors are returned with hints."""
        mock_get_user.return_value = MagicMock()
        mock_validation.return_value = ValidationResult(
            success=False,
            error=ValidationError(
                "reference_validation",
                "Workflow references non-existent tools or triggers",
                "Tool 'bad_tool' not found.\nUse search_tools() and list_triggers() to find valid names",
            ),
        )

        from seer.mcp.tools.workflows import validate_and_upsert_workflow
        result = await validate_and_upsert_workflow.fn("Test", {"version": "2", "nodes": []})
        data = json.loads(result)

        assert data["status"] == "error"
        assert data["error_type"] == "reference_validation"
        assert "hint" in data

    @pytest.mark.asyncio
    @patch("seer.api.workflows.models.WorkflowCreateRequest")
    @patch("seer.api.workflows.services.lifecycle.create_workflow")
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    @patch("seer.mcp.tools.workflows.run_full_validation")
    async def test_upsert_creates_workflow_on_success(self, mock_validation, mock_get_user, mock_create, mock_request_cls):
        """Test that a valid spec results in workflow creation."""
        mock_get_user.return_value = MagicMock()
        mock_spec = MagicMock()
        mock_validation.return_value = ValidationResult(
            success=True,
            validated_spec=mock_spec,
            fixed_spec_dict={"version": "2", "nodes": []},
            schema_fixes=[],
        )
        mock_response = MagicMock()
        mock_response.workflow_id = "wf_test123"
        mock_response.name = "Test Workflow"
        mock_response.spec.model_dump.return_value = {"version": "2", "nodes": []}
        mock_response.created_at.isoformat.return_value = "2025-01-01T00:00:00"
        mock_create.return_value = mock_response

        from seer.mcp.tools.workflows import validate_and_upsert_workflow
        result = await validate_and_upsert_workflow.fn("Test Workflow", {"version": "2", "nodes": []})
        data = json.loads(result)

        assert data["status"] == "ok"
        assert data["workflow_id"] == "wf_test123"

    @pytest.mark.asyncio
    @patch("seer.api.workflows.models.WorkflowCreateRequest")
    @patch("seer.api.workflows.services.lifecycle.create_workflow")
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    @patch("seer.mcp.tools.workflows.run_full_validation")
    async def test_upsert_includes_auto_fixes(self, mock_validation, mock_get_user, mock_create, mock_request_cls):
        """Test that auto-fix info is included when trigger schemas were corrected."""
        mock_get_user.return_value = MagicMock()
        mock_spec = MagicMock()
        mock_validation.return_value = ValidationResult(
            success=True,
            validated_spec=mock_spec,
            fixed_spec_dict={"version": "2", "nodes": []},
            schema_fixes=[{"trigger_id": "t1", "trigger_key": "poll.gmail.email_received", "reason": "auto-fixed"}],
        )
        mock_response = MagicMock()
        mock_response.workflow_id = "wf_test123"
        mock_response.name = "Test Workflow"
        mock_response.spec.model_dump.return_value = {"version": "2", "nodes": []}
        mock_response.created_at.isoformat.return_value = "2025-01-01T00:00:00"
        mock_create.return_value = mock_response

        from seer.mcp.tools.workflows import validate_and_upsert_workflow
        result = await validate_and_upsert_workflow.fn("Test Workflow", {"version": "2", "nodes": []})
        data = json.loads(result)

        assert data["status"] == "ok"
        assert "auto_fixes" in data


@pytest.mark.unit
class TestListWorkflows:
    """Tests for list_workflows MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    async def test_list_workflows_returns_empty_on_error(self, mock_get_user):
        """Test that list_workflows returns empty list on error."""
        mock_get_user.side_effect = Exception("Database error")

        from seer.mcp.tools.workflows import list_workflows
        result = await list_workflows.fn()
        data = json.loads(result)

        assert "workflows" in data
        assert data["workflows"] == []
        assert "error" in data


@pytest.mark.unit
class TestDeleteWorkflow:
    """Tests for delete_workflow MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.workflows._get_mcp_user")
    async def test_delete_returns_error_on_failure(self, mock_get_user):
        """Test that delete_workflow returns error on failure."""
        mock_get_user.side_effect = Exception("Not found")

        from seer.mcp.tools.workflows import delete_workflow
        result = await delete_workflow.fn("wf_nonexistent")
        data = json.loads(result)

        assert data["deleted"] is False
        assert "error" in data
