"""
Unit tests for workflow agent services pure logic.

Tests graph transformation and spec parsing.
Heavy mock tests for chat/proposal services have been moved to E2E tests.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_spec_dict():
    """Sample workflow spec dict."""
    return {
        "version": "2",
        "nodes": [
            {
                "id": "n1",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {},
                "meta": {"label": "Node 1", "position": {"x": 100, "y": 100}},
            },
            {
                "id": "n2",
                "type": "agent",
                "inputs": {"model": "qwen/qwen3-235b-a22b-2507", "prompt": "Test"},
                "meta": {"label": "Node 2", "position": {"x": 200, "y": 200}},
            },
        ],
        "edges": [],
    }


@pytest.fixture
def minimal_spec_dict():
    """Minimal workflow spec dict."""
    return {
        "version": "2",
        "nodes": [],
        "edges": [],
    }


# =============================================================================
# Workflow State From Spec Tests
# =============================================================================


class TestWorkflowStateFromSpec:
    """Tests for workflow_state_from_spec function."""

    def test_workflow_state_from_spec_basic(self, sample_spec_dict):
        """Test converting spec to workflow state."""
        from seer.api.agents.workflow.services import workflow_state_from_spec

        result = workflow_state_from_spec(sample_spec_dict)

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 2

    def test_workflow_state_from_spec_empty(self, minimal_spec_dict):
        """Test converting empty spec."""
        from seer.api.agents.workflow.services import workflow_state_from_spec

        result = workflow_state_from_spec(minimal_spec_dict)

        assert result["nodes"] == []
        assert result["edges"] == []

    def test_workflow_state_from_spec_invalid_input(self):
        """Test handling invalid input."""
        from seer.api.agents.workflow.services import workflow_state_from_spec

        result = workflow_state_from_spec("not a dict")

        assert result == {"nodes": [], "edges": []}

    def test_workflow_state_from_spec_none_input(self):
        """Test handling None input."""
        from seer.api.agents.workflow.services import workflow_state_from_spec

        result = workflow_state_from_spec(None)

        assert result == {"nodes": [], "edges": []}


# =============================================================================
# Internal Workflow State Tests
# =============================================================================


class TestInternalWorkflowState:
    """Tests for _workflow_state_from_spec internal function."""

    def test_node_extraction(self, sample_spec_dict):
        """Test that nodes are correctly extracted."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        result = _workflow_state_from_spec(sample_spec_dict)

        node1 = result["nodes"][0]
        assert node1["id"] == "n1"
        assert node1["type"] == "tool"
        assert node1["data"]["label"] == "Node 1"

    def test_position_extraction(self, sample_spec_dict):
        """Test that positions are extracted from meta."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        result = _workflow_state_from_spec(sample_spec_dict)

        node1 = result["nodes"][0]
        assert "position" in node1
        assert node1["position"]["x"] == 100
        assert node1["position"]["y"] == 100

    def test_edge_generation(self, sample_spec_dict):
        """Test that edges are generated between sequential nodes."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        result = _workflow_state_from_spec(sample_spec_dict)

        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert edge["source"] == "n1"
        assert edge["target"] == "n2"

    def test_node_without_meta(self):
        """Test handling nodes without meta field."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool"},
            ]
        }

        result = _workflow_state_from_spec(spec)

        node = result["nodes"][0]
        assert node["data"]["label"] == "n1"
        assert "position" not in node

    def test_node_with_invalid_meta(self):
        """Test handling nodes with invalid meta type."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool", "meta": "not a dict"},
            ]
        }

        result = _workflow_state_from_spec(spec)

        node = result["nodes"][0]
        assert node["data"]["label"] == "n1"

    def test_node_with_empty_meta(self):
        """Test handling nodes with empty meta dict."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool", "meta": {}},
            ]
        }

        result = _workflow_state_from_spec(spec)

        node = result["nodes"][0]
        assert node["data"]["label"] == "n1"
        assert "position" not in node

    def test_nodes_not_a_list(self):
        """Test handling when nodes field is not a list."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {"nodes": "not a list"}

        result = _workflow_state_from_spec(spec)

        assert result["nodes"] == []
        assert result["edges"] == []

    def test_node_not_a_dict(self):
        """Test handling when a node is not a dict."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {"nodes": ["not a dict", {"id": "n1", "type": "tool"}]}

        result = _workflow_state_from_spec(spec)

        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "n1"

    def test_position_with_missing_coordinates(self):
        """Test position defaults when x/y are missing."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool", "meta": {"position": {}}},
            ]
        }

        result = _workflow_state_from_spec(spec)

        node = result["nodes"][0]
        assert node["position"]["x"] == 0
        assert node["position"]["y"] == 0

    def test_multiple_edges_for_multiple_nodes(self):
        """Test edge generation for multiple sequential nodes."""
        from seer.api.agents.workflow.services import _workflow_state_from_spec

        spec = {
            "nodes": [
                {"id": "n1", "type": "tool"},
                {"id": "n2", "type": "tool"},
                {"id": "n3", "type": "tool"},
            ]
        }

        result = _workflow_state_from_spec(spec)

        assert len(result["edges"]) == 2
        assert result["edges"][0]["source"] == "n1"
        assert result["edges"][0]["target"] == "n2"
        assert result["edges"][1]["source"] == "n2"
        assert result["edges"][1]["target"] == "n3"


# =============================================================================
# Preview From Spec Tests
# =============================================================================


class TestPreviewFromSpec:
    """Tests for _preview_from_spec function."""

    def test_preview_from_spec_basic(self, sample_spec_dict):
        """Test building preview from spec."""
        from seer.api.agents.workflow.services import _preview_from_spec

        result = _preview_from_spec(sample_spec_dict)

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["id"] == "n1"
        assert result["nodes"][0]["type"] == "tool"

    def test_preview_from_spec_empty(self, minimal_spec_dict):
        """Test preview from empty spec."""
        from seer.api.agents.workflow.services import _preview_from_spec

        result = _preview_from_spec(minimal_spec_dict)

        assert result["nodes"] == []
        assert result["edges"] == []

    def test_preview_from_spec_generates_edges(self, sample_spec_dict):
        """Test that preview generates edges between sequential nodes."""
        from seer.api.agents.workflow.services import _preview_from_spec

        result = _preview_from_spec(sample_spec_dict)

        assert len(result["edges"]) == 1
        assert result["edges"][0]["source"] == "n1"
        assert result["edges"][0]["target"] == "n2"


# =============================================================================
# Normalize Spec Tests
# =============================================================================


class TestNormalizeSpec:
    """Tests for _normalize_spec function."""

    def test_normalize_spec_none_input(self):
        """Test that None input raises HTTPException."""
        from seer.api.agents.workflow.services import _normalize_spec

        with pytest.raises(HTTPException) as exc_info:
            _normalize_spec(None)

        assert exc_info.value.status_code == 400
        assert "required" in exc_info.value.detail

    def test_normalize_spec_empty_dict(self):
        """Test that empty dict raises HTTPException."""
        from seer.api.agents.workflow.services import _normalize_spec

        with pytest.raises(HTTPException) as exc_info:
            _normalize_spec({})

        assert exc_info.value.status_code == 400

    def test_normalize_spec_valid(self, minimal_spec_dict):
        """Test normalizing a valid spec."""
        with patch("seer.core.compiler.parse.parse_workflow_spec") as mock_parse:
            mock_spec = MagicMock()
            mock_spec.model_dump.return_value = minimal_spec_dict
            mock_parse.return_value = mock_spec

            from seer.api.agents.workflow.services import _normalize_spec

            result = _normalize_spec(minimal_spec_dict)

            assert result == minimal_spec_dict
            mock_parse.assert_called_once_with(minimal_spec_dict)

    def test_normalize_spec_invalid(self):
        """Test that invalid spec raises HTTPException."""
        with patch("seer.core.compiler.parse.parse_workflow_spec") as mock_parse:
            mock_parse.side_effect = ValueError("Invalid spec")

            from seer.api.agents.workflow.services import _normalize_spec

            with pytest.raises(HTTPException) as exc_info:
                _normalize_spec({"invalid": "spec"})

            assert exc_info.value.status_code == 400
            assert "Invalid workflow spec" in exc_info.value.detail
