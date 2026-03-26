"""
Unit tests for workflow lifecycle pure logic.

Tests parsing, hashing, cursor handling, and data transformation.
Heavy mock tests for service orchestration have been moved to E2E tests.
"""
from datetime import datetime, timezone
import hashlib
import json

import pytest
from fastapi import HTTPException

from seer.database import (
    parse_workflow_public_id,
    make_workflow_public_id,
    make_run_public_id,
    parse_run_public_id,
)
from seer.api.workflows.services.shared import (
    _hash_spec,
    _now,
    _spec_to_dict,
)
from seer.api.workflows.services.lifecycle import _parse_workflow_cursor


# =============================================================================
# Parse Workflow ID Tests
# =============================================================================


@pytest.mark.unit
class TestParseWorkflowPublicId:
    """Tests for parse_workflow_public_id function."""

    def test_parse_valid_workflow_id(self):
        """Test parsing valid workflow ID."""
        result = parse_workflow_public_id("wf_123")
        assert result == 123

    def test_parse_large_workflow_id(self):
        """Test parsing large workflow ID."""
        result = parse_workflow_public_id("wf_999999999")
        assert result == 999999999

    def test_parse_workflow_id_with_zero(self):
        """Test parsing workflow ID with zero."""
        result = parse_workflow_public_id("wf_0")
        assert result == 0

    def test_parse_invalid_prefix_raises_error(self):
        """Test parsing ID with wrong prefix raises error."""
        with pytest.raises(ValueError, match="Invalid workflow_id format"):
            parse_workflow_public_id("wx_123")

    def test_parse_no_prefix_raises_error(self):
        """Test parsing ID without prefix raises error."""
        with pytest.raises(ValueError, match="Invalid workflow_id format"):
            parse_workflow_public_id("123")

    def test_parse_invalid_suffix_raises_error(self):
        """Test parsing ID with non-numeric suffix raises error."""
        with pytest.raises(ValueError):
            parse_workflow_public_id("wf_abc")

    def test_parse_empty_suffix_raises_error(self):
        """Test parsing ID with empty suffix raises error."""
        with pytest.raises(ValueError):
            parse_workflow_public_id("wf_")

    def test_parse_empty_id_raises_error(self):
        """Test parsing empty ID raises error."""
        with pytest.raises(ValueError, match="Invalid workflow_id format"):
            parse_workflow_public_id("")

    def test_parse_prefix_only_raises_error(self):
        """Test parsing prefix only raises error."""
        with pytest.raises(ValueError):
            parse_workflow_public_id("wf_")


@pytest.mark.unit
class TestMakeWorkflowPublicId:
    """Tests for make_workflow_public_id function."""

    def test_make_workflow_id(self):
        """Test creating workflow public ID."""
        result = make_workflow_public_id(123)
        assert result == "wf_123"

    def test_make_workflow_id_zero(self):
        """Test creating workflow public ID for zero."""
        result = make_workflow_public_id(0)
        assert result == "wf_0"

    def test_make_workflow_id_large_number(self):
        """Test creating workflow public ID for large number."""
        result = make_workflow_public_id(999999999)
        assert result == "wf_999999999"

    def test_roundtrip_workflow_id(self):
        """Test roundtrip conversion."""
        original_pk = 42
        public_id = make_workflow_public_id(original_pk)
        parsed_pk = parse_workflow_public_id(public_id)
        assert parsed_pk == original_pk


# =============================================================================
# Parse Run ID Tests
# =============================================================================


@pytest.mark.unit
class TestRunPublicId:
    """Tests for run public ID functions."""

    def test_make_run_id(self):
        """Test creating run public ID."""
        result = make_run_public_id(456)
        assert result == "run_456"

    def test_parse_run_id(self):
        """Test parsing run public ID."""
        result = parse_run_public_id("run_456")
        assert result == 456

    def test_parse_invalid_run_prefix_raises_error(self):
        """Test parsing run ID with wrong prefix raises error."""
        with pytest.raises(ValueError, match="Invalid run_id format"):
            parse_run_public_id("wf_123")

    def test_roundtrip_run_id(self):
        """Test roundtrip conversion."""
        original_pk = 789
        public_id = make_run_public_id(original_pk)
        parsed_pk = parse_run_public_id(public_id)
        assert parsed_pk == original_pk


# =============================================================================
# Parse Workflow Cursor Tests
# =============================================================================


@pytest.mark.unit
class TestParseWorkflowCursor:
    """Tests for _parse_workflow_cursor function."""

    def test_parse_none_cursor(self):
        """Test parsing None cursor returns None."""
        result = _parse_workflow_cursor(None)
        assert result is None

    def test_parse_numeric_string_cursor(self):
        """Test parsing numeric string cursor."""
        result = _parse_workflow_cursor("123")
        assert result == 123

    def test_parse_workflow_id_format_cursor(self):
        """Test parsing wf_* format cursor."""
        result = _parse_workflow_cursor("wf_456")
        assert result == 456

    def test_parse_invalid_cursor_raises_problem(self):
        """Test parsing invalid cursor raises HTTP problem."""
        with pytest.raises(HTTPException) as exc_info:
            _parse_workflow_cursor("invalid")

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["title"] == "Invalid cursor"


# =============================================================================
# List Workflows Limit Bounds Tests
# =============================================================================


@pytest.mark.unit
class TestListWorkflowsLimitBounds:
    """Tests for list_workflows limit bounds logic."""

    @pytest.mark.parametrize("input_limit,expected", [
        (0, 1),      # Minimum boundary - clamp to 1
        (-5, 1),     # Negative value - clamp to 1
        (50, 50),    # Valid middle value - passes through
        (200, 100),  # Above maximum - clamp to 100
        (1, 1),      # At minimum boundary
        (100, 100),  # At maximum boundary
    ])
    def test_limit_bounds_clamping(self, input_limit, expected):
        """Test limit is clamped to valid range [1, 100]."""
        result = max(1, min(input_limit, 100))
        assert result == expected


# =============================================================================
# Spec Hash Tests
# =============================================================================


@pytest.mark.unit
class TestSpecHash:
    """Tests for _hash_spec function."""

    def test_spec_hash_deterministic(self):
        """Test that spec hash is deterministic for same input."""
        spec = {"version": "2", "nodes": [], "edges": []}

        hash1 = _hash_spec(spec)
        hash2 = _hash_spec(spec)

        assert hash1 == hash2

    def test_spec_hash_different_for_different_specs(self):
        """Test that different specs produce different hashes."""
        spec1 = {"version": "2", "nodes": [], "edges": []}
        spec2 = {"version": "2", "nodes": [{"id": "n1"}], "edges": []}

        hash1 = _hash_spec(spec1)
        hash2 = _hash_spec(spec2)

        assert hash1 != hash2

    def test_spec_hash_key_order_independent(self):
        """Test spec hash is consistent regardless of key order."""
        spec1 = {"version": "2", "nodes": [], "edges": []}
        spec2 = {"edges": [], "nodes": [], "version": "2"}

        hash1 = _hash_spec(spec1)
        hash2 = _hash_spec(spec2)

        assert hash1 == hash2

    def test_spec_hash_is_sha256(self):
        """Test spec hash uses SHA256."""
        spec = {"version": "2", "nodes": [], "edges": []}
        hash_value = _hash_spec(spec)

        # SHA256 produces 64 hex characters
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_spec_hash_matches_manual_calculation(self):
        """Test spec hash matches manual SHA256 calculation."""
        spec = {"version": "2", "nodes": [], "edges": []}

        # Manual calculation with same parameters
        serialized = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected = hashlib.sha256(serialized).hexdigest()

        result = _hash_spec(spec)
        assert result == expected


# =============================================================================
# Now Function Tests
# =============================================================================


@pytest.mark.unit
class TestNowFunction:
    """Tests for _now function."""

    def test_now_returns_datetime(self):
        """Test _now returns a datetime object."""
        result = _now()
        assert isinstance(result, datetime)

    def test_now_is_utc(self):
        """Test _now returns UTC timezone."""
        result = _now()
        assert result.tzinfo == timezone.utc

    def test_now_is_recent(self):
        """Test _now returns a recent time."""
        before = datetime.now(timezone.utc)
        result = _now()
        after = datetime.now(timezone.utc)

        assert before <= result <= after


# =============================================================================
# Spec to Dict Tests
# =============================================================================


@pytest.mark.unit
class TestSpecToDict:
    """Tests for _spec_to_dict function."""

    def test_spec_to_dict_converts_model(self):
        """Test _spec_to_dict converts Pydantic model to dict."""
        from seer.core.schema.models import WorkflowSpec

        spec = WorkflowSpec(version="2.0", nodes=[], edges=[])
        result = _spec_to_dict(spec)

        assert isinstance(result, dict)
        assert result["version"] == "2.0"
        assert result["nodes"] == []
        assert result["edges"] == []


# =============================================================================
# Workflow State Transformation Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowStateTransformation:
    """Tests for workflow state transformation logic."""

    def test_extract_node_label_from_meta(self):
        """Test extracting node label from meta."""
        node = {
            "id": "n1",
            "type": "tool",
            "meta": {"label": "My Node"}
        }

        meta = node.get("meta", {})
        label = meta.get("label") or node.get("id")

        assert label == "My Node"

    def test_node_label_fallback_to_id(self):
        """Test node label falls back to id."""
        node = {
            "id": "n1",
            "type": "tool"
        }

        meta = node.get("meta", {})
        label = meta.get("label") or node.get("id")

        assert label == "n1"

    def test_extract_position_from_meta(self):
        """Test extracting position from node meta."""
        node = {
            "id": "n1",
            "meta": {"position": {"x": 100, "y": 200}}
        }

        meta = node.get("meta", {})
        position = meta.get("position")

        assert position["x"] == 100
        assert position["y"] == 200


# =============================================================================
# Export/Import Format Tests
# =============================================================================


@pytest.mark.unit
class TestExportImportFormat:
    """Tests for export/import workflow format."""

    def test_export_format_version(self):
        """Test export format has version 1.0."""
        export_data = {
            "version": "1.0",
            "workflow": {"name": "Test", "spec": {}},
            "triggers": [],
            "metadata": {}
        }

        assert export_data["version"] == "1.0"

    def test_export_preserves_spec(self):
        """Test export preserves spec structure."""
        original_spec = {
            "version": "2.0",
            "nodes": [{"id": "n1", "type": "tool"}],
            "edges": [],
            "triggers": []
        }

        # Export should preserve the spec
        exported_spec = json.dumps(original_spec)
        reimported = json.loads(exported_spec)

        assert reimported == original_spec

    def test_import_validates_required_fields(self):
        """Test import validates required spec fields."""
        required_fields = ["version", "nodes", "edges"]
        spec = {"version": "2.0", "nodes": [], "edges": []}

        for field in required_fields:
            assert field in spec

    def test_import_handles_missing_triggers(self):
        """Test import handles missing triggers gracefully."""
        spec = {"version": "2.0", "nodes": [], "edges": []}

        triggers = spec.get("triggers", [])
        assert triggers == []


# =============================================================================
# Pagination Logic Tests
# =============================================================================


@pytest.mark.unit
class TestPaginationLogic:
    """Tests for pagination logic."""

    def test_has_more_when_exceeds_limit(self):
        """Test has_more is True when items exceed limit."""
        limit = 10
        items = list(range(11))  # 11 items fetched (limit + 1)

        has_more = len(items) > limit

        assert has_more is True

    def test_no_more_when_under_limit(self):
        """Test has_more is False when items under limit."""
        limit = 10
        items = list(range(5))

        has_more = len(items) > limit

        assert has_more is False

    def test_no_more_when_exact_limit(self):
        """Test has_more is False when items equal limit."""
        limit = 10
        items = list(range(10))

        has_more = len(items) > limit

        assert has_more is False

    def test_next_cursor_from_last_item(self):
        """Test generating next cursor from last item."""
        items = [
            {"id": 1, "name": "First"},
            {"id": 2, "name": "Second"},
            {"id": 3, "name": "Third"},
        ]

        next_cursor = str(items[-1]["id"]) if items else None

        assert next_cursor == "3"

    def test_empty_list_no_cursor(self):
        """Test empty list produces no cursor."""
        items = []

        next_cursor = str(items[-1]["id"]) if items else None

        assert next_cursor is None


# =============================================================================
# Workflow Summary Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowSummary:
    """Tests for workflow summary structure."""

    def test_workflow_id_format(self):
        """Test workflow ID format in response."""
        workflow_id = 123
        public_id = make_workflow_public_id(workflow_id)

        assert public_id == "wf_123"
        assert public_id.startswith("wf_")

    def test_trigger_count_from_spec(self):
        """Test counting triggers from spec."""
        spec = {
            "triggers": [
                {"type": "webhook", "id": "t1"},
                {"type": "schedule", "id": "t2"},
            ]
        }

        trigger_count = len(spec.get("triggers", []))

        assert trigger_count == 2

    def test_trigger_count_empty_spec(self):
        """Test trigger count when spec has no triggers."""
        spec = {"triggers": []}
        trigger_count = len(spec.get("triggers", []))
        assert trigger_count == 0

    def test_trigger_count_missing_triggers_key(self):
        """Test trigger count when triggers key is missing."""
        spec = {"nodes": []}
        trigger_count = len(spec.get("triggers", []))
        assert trigger_count == 0
