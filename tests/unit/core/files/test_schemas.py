"""
Unit tests for file I/O schemas.

Tests the schema definitions used for file inputs and outputs in workflow tools.
"""

import pytest

from seer.core.files.models import WORKFLOW_FILE_REF_TYPE
from seer.core.files.schemas import (
    FILE_INPUT_SCHEMA,
    FILE_OUTPUT_SCHEMA,
    STATIC_FILE_REF_TYPE,
    get_file_array_input_property,
    get_file_input_property,
    is_static_file_ref,
)


# =============================================================================
# is_static_file_ref Tests
# =============================================================================


@pytest.mark.unit
class TestIsStaticFileRef:
    """Tests for is_static_file_ref detection function."""

    def test_is_static_file_ref_valid(self):
        """Test is_static_file_ref returns True for valid static file ref dict."""
        data = {
            "_type": STATIC_FILE_REF_TYPE,
            "file_id": "abc-123",
        }
        assert is_static_file_ref(data) is True

    def test_is_static_file_ref_workflow_file_ref(self):
        """Test is_static_file_ref returns False for WorkflowFileRef."""
        data = {
            "_type": WORKFLOW_FILE_REF_TYPE,
            "file_id": "abc-123",
        }
        assert is_static_file_ref(data) is False

    def test_is_static_file_ref_no_type(self):
        """Test is_static_file_ref returns False when _type missing."""
        data = {"file_id": "abc-123"}
        assert is_static_file_ref(data) is False

    def test_is_static_file_ref_wrong_type(self):
        """Test is_static_file_ref returns False for wrong _type."""
        data = {
            "_type": "something_else",
            "file_id": "abc-123",
        }
        assert is_static_file_ref(data) is False

    def test_is_static_file_ref_not_dict(self):
        """Test is_static_file_ref returns False for non-dict values."""
        assert is_static_file_ref("string") is False
        assert is_static_file_ref(123) is False
        assert is_static_file_ref(None) is False
        assert is_static_file_ref([]) is False


# =============================================================================
# FILE_INPUT_SCHEMA Tests
# =============================================================================


@pytest.mark.unit
class TestFileInputSchema:
    """Tests for FILE_INPUT_SCHEMA structure."""

    def test_schema_has_oneof(self):
        """Test FILE_INPUT_SCHEMA uses oneOf for multiple input types."""
        assert "oneOf" in FILE_INPUT_SCHEMA
        assert len(FILE_INPUT_SCHEMA["oneOf"]) == 2

    def test_schema_has_workflow_file_ref_option(self):
        """Test FILE_INPUT_SCHEMA includes WorkflowFileRef option."""
        options = FILE_INPUT_SCHEMA["oneOf"]
        workflow_ref_option = next(
            (opt for opt in options if opt.get("properties", {}).get("_type", {}).get("const") == WORKFLOW_FILE_REF_TYPE),
            None,
        )
        assert workflow_ref_option is not None
        assert "file_id" in workflow_ref_option["properties"]
        assert "storage_path" in workflow_ref_option["properties"]
        assert "filename" in workflow_ref_option["properties"]
        assert "mime_type" in workflow_ref_option["properties"]

    def test_schema_has_static_file_ref_option(self):
        """Test FILE_INPUT_SCHEMA includes static_file_ref option."""
        options = FILE_INPUT_SCHEMA["oneOf"]
        static_ref_option = next(
            (opt for opt in options if opt.get("properties", {}).get("_type", {}).get("const") == STATIC_FILE_REF_TYPE),
            None,
        )
        assert static_ref_option is not None
        assert "file_id" in static_ref_option["properties"]
        assert "_type" in static_ref_option["required"]
        assert "file_id" in static_ref_option["required"]


# =============================================================================
# FILE_OUTPUT_SCHEMA Tests
# =============================================================================


@pytest.mark.unit
class TestFileOutputSchema:
    """Tests for FILE_OUTPUT_SCHEMA structure."""

    def test_schema_is_object(self):
        """Test FILE_OUTPUT_SCHEMA is an object type."""
        assert FILE_OUTPUT_SCHEMA["type"] == "object"

    def test_schema_has_required_fields(self):
        """Test FILE_OUTPUT_SCHEMA has required fields."""
        required = FILE_OUTPUT_SCHEMA.get("required", [])
        assert "_type" in required
        assert "file_id" in required
        assert "filename" in required
        assert "mime_type" in required
        assert "size_bytes" in required

    def test_schema_has_workflow_file_ref_type(self):
        """Test FILE_OUTPUT_SCHEMA uses WorkflowFileRef type."""
        assert FILE_OUTPUT_SCHEMA["properties"]["_type"]["const"] == WORKFLOW_FILE_REF_TYPE


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
class TestSchemaHelpers:
    """Tests for schema helper functions."""

    def test_get_file_input_property_default(self):
        """Test get_file_input_property with default description."""
        schema = get_file_input_property()
        assert schema["description"] == "File input"
        assert "oneOf" in schema

    def test_get_file_input_property_custom_description(self):
        """Test get_file_input_property with custom description."""
        schema = get_file_input_property("Custom file description")
        assert schema["description"] == "Custom file description"

    def test_get_file_array_input_property_basic(self):
        """Test get_file_array_input_property basic structure."""
        schema = get_file_array_input_property()
        assert schema["type"] == "array"
        assert "items" in schema
        assert "oneOf" in schema["items"]

    def test_get_file_array_input_property_with_limits(self):
        """Test get_file_array_input_property with item limits."""
        schema = get_file_array_input_property(
            description="Multiple files",
            min_items=1,
            max_items=5,
        )
        assert schema["description"] == "Multiple files"
        assert schema["minItems"] == 1
        assert schema["maxItems"] == 5

    def test_get_file_array_input_property_no_max(self):
        """Test get_file_array_input_property without max_items."""
        schema = get_file_array_input_property()
        assert "maxItems" not in schema
