"""
Unit tests for MCP execution tools.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from seer.api.workflows.models import RunResponse


@pytest.mark.unit
class TestRunWorkflow:
    """Tests for run_workflow MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.execution._ensure_db")
    @patch("seer.mcp.tools.execution._get_mcp_user")
    async def test_run_workflow_handles_error(self, mock_get_user, mock_ensure_db):
        """Test that run_workflow handles errors gracefully."""
        mock_ensure_db.return_value = None
        mock_get_user.side_effect = Exception("Database error")

        from seer.mcp.tools.execution import run_workflow
        result = await run_workflow.fn("wf_test")
        data = json.loads(result)

        assert "error" in data
        assert data["error"] == "execution_failed"

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.execution._ensure_db")
    @patch("seer.mcp.tools.execution._get_mcp_user")
    @patch("seer.api.workflows.services.execution.run_saved_workflow")
    async def test_run_workflow_success(self, mock_run, mock_get_user, mock_ensure_db):
        """Test that run_workflow returns correct JSON for a successful run."""
        mock_ensure_db.return_value = None
        mock_get_user.return_value = MagicMock()
        mock_run.return_value = RunResponse(
            run_id="run_abc",
            status="pending",
            workflow_id="wf_test",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        from seer.mcp.tools.execution import run_workflow
        result = await run_workflow.fn("wf_test")
        data = json.loads(result)

        assert data["run_id"] == "run_abc"
        assert data["status"] == "pending"
        assert data["workflow_id"] == "wf_test"
        assert "created_at" in data


@pytest.mark.unit
class TestGetRunStatus:
    """Tests for get_run_status MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.execution._ensure_db")
    @patch("seer.mcp.tools.execution._get_mcp_user")
    async def test_get_run_status_handles_not_found(self, mock_get_user, mock_ensure_db):
        """Test that get_run_status handles not found errors."""
        mock_ensure_db.return_value = None
        mock_get_user.side_effect = Exception("Run not found")

        from seer.mcp.tools.execution import get_run_status
        result = await get_run_status.fn("wf_test", "run_nonexistent")
        data = json.loads(result)

        assert "error" in data

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.execution._ensure_db")
    @patch("seer.mcp.tools.execution._get_mcp_user")
    @patch("seer.api.workflows.services.history.get_run_status")
    async def test_get_run_status_success(self, mock_svc, mock_get_user, mock_ensure_db):
        """Test that get_run_status returns correct JSON for a known run."""
        mock_ensure_db.return_value = None
        mock_get_user.return_value = MagicMock()
        mock_svc.return_value = RunResponse(
            run_id="run_123",
            status="completed",
            workflow_id="wf_test",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        from seer.mcp.tools.execution import get_run_status
        result = await get_run_status.fn("wf_test", "run_123")
        data = json.loads(result)

        assert data["run_id"] == "run_123"
        assert data["status"] == "completed"


@pytest.mark.unit
class TestListRuns:
    """Tests for list_runs MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.execution._ensure_db")
    @patch("seer.mcp.tools.execution._get_mcp_user")
    async def test_list_runs_handles_error(self, mock_get_user, mock_ensure_db):
        """Test that list_runs handles errors gracefully."""
        mock_ensure_db.return_value = None
        mock_get_user.side_effect = Exception("Database error")

        from seer.mcp.tools.execution import list_runs
        result = await list_runs.fn("wf_test")
        data = json.loads(result)

        assert "runs" in data
        assert data["runs"] == []
        assert "error" in data

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.execution._ensure_db")
    @patch("seer.mcp.tools.execution._get_mcp_user")
    @patch("seer.api.workflows.services.execution.list_workflow_runs")
    async def test_list_runs_success(self, mock_svc, mock_get_user, mock_ensure_db):
        """Test that list_runs returns correct JSON with run list."""
        mock_ensure_db.return_value = None
        mock_get_user.return_value = MagicMock()
        run = MagicMock()
        run.run_id = "run_abc"
        run.status = "completed"
        run.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        run.started_at = None
        run.finished_at = None
        run.error = None
        mock_response = MagicMock()
        mock_response.workflow_id = "wf_test"
        mock_response.runs = [run]
        mock_svc.return_value = mock_response

        from seer.mcp.tools.execution import list_runs
        result = await list_runs.fn("wf_test")
        data = json.loads(result)

        assert data["workflow_id"] == "wf_test"
        assert len(data["runs"]) == 1
        assert data["runs"][0]["run_id"] == "run_abc"
        assert data["total"] == 1


@pytest.mark.unit
class TestGetRunHistory:
    """Tests for get_run_history MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.execution._ensure_db")
    @patch("seer.mcp.tools.execution._get_mcp_user")
    async def test_get_run_history_handles_error(self, mock_get_user, mock_ensure_db):
        """Test that get_run_history handles errors gracefully."""
        mock_ensure_db.return_value = None
        mock_get_user.side_effect = Exception("History not available")

        from seer.mcp.tools.execution import get_run_history
        result = await get_run_history.fn("wf_test", "run_123")
        data = json.loads(result)

        assert "error" in data
        assert data["history"] == []

    @pytest.mark.asyncio
    @patch("seer.mcp.tools.execution._ensure_db")
    @patch("seer.mcp.tools.execution._get_mcp_user")
    @patch("seer.api.workflows.services.history.get_run_history")
    async def test_get_run_history_success(self, mock_svc, mock_get_user, mock_ensure_db):
        """Test that get_run_history returns correct JSON with history."""
        mock_ensure_db.return_value = None
        mock_get_user.return_value = MagicMock()
        mock_response = MagicMock()
        mock_response.run_id = "run_123"
        mock_response.history = [{"node_id": "n1", "status": "completed"}]
        mock_svc.return_value = mock_response

        from seer.mcp.tools.execution import get_run_history
        result = await get_run_history.fn("wf_test", "run_123")
        data = json.loads(result)

        assert data["run_id"] == "run_123"
        assert len(data["history"]) == 1
        assert data["history"][0]["node_id"] == "n1"
