"""
Unit tests for MCP execution tools.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


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
