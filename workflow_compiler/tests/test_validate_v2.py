"""Tests for V2 workflow schema validator"""
# pylint: disable=redefined-outer-name  # pytest fixture pattern

import json
from pathlib import Path

import pytest

from workflow_compiler.schema.validate_v2 import WorkflowValidator, validate_workflow_v2


@pytest.fixture
def v2_schema():
    """Load V2 JSON Schema"""
    schema_path = Path(__file__).parent.parent / "schema" / "v2_schema.json"
    with open(schema_path, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def validator(v2_schema):
    """Create workflow validator"""
    return WorkflowValidator(v2_schema)


class TestSchemaValidation:
    """Test JSON Schema validation"""

    def test_minimal_valid_workflow(self, validator):
        """Test minimal valid workflow passes validation"""
        workflow = {
            "name": "Test Workflow",
            "spec": {
                "nodes": [],
                "edges": []
            }
        }
        errors = validator.validate(workflow)
        assert len(errors) == 0

    def test_missing_required_fields(self, validator):
        """Test workflow missing required fields fails"""
        workflow = {}  # Missing 'name' and 'spec'
        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("required" in str(e).lower() for e in errors)

    def test_invalid_node_type(self, validator):
        """Test workflow with invalid node type fails"""
        workflow = {
            "name": "Test",
            "spec": {
                "nodes": [{"id": "test", "type": "invalid_type"}],
                "edges": []
            }
        }
        errors = validator.validate(workflow)
        assert len(errors) > 0


class TestDAGValidation:
    """Test DAG structure validation"""

    def test_simple_sequential_dag(self, validator):
        """Test simple sequential DAG is valid"""
        workflow = {
            "name": "Sequential",
            "spec": {
                "nodes": [
                    {"id": "node1", "type": "task", "kind": "set", "value": "test"},
                    {"id": "node2", "type": "task", "kind": "set", "value": "test2"}
                ],
                "edges": [
                    {"from": "_start", "to": "node1"},
                    {"from": "node1", "to": "node2"},
                    {"from": "node2", "to": "_end"}
                ]
            }
        }
        errors = validator.validate(workflow)
        assert len(errors) == 0

    def test_parallel_dag(self, validator):
        """Test parallel DAG is valid"""
        workflow = {
            "name": "Parallel",
            "spec": {
                "nodes": [
                    {"id": "node1", "type": "task", "kind": "set", "value": "test1"},
                    {"id": "node2", "type": "task", "kind": "set", "value": "test2"},
                    {"id": "node3", "type": "task", "kind": "set", "value": "test3"}
                ],
                "edges": [
                    {"from": "_start", "to": "node1"},
                    {"from": "_start", "to": "node2"},
                    {"from": "node1", "to": "node3"},
                    {"from": "node2", "to": "node3"},
                    {"from": "node3", "to": "_end"}
                ]
            }
        }
        errors = validator.validate(workflow)
        assert len(errors) == 0

    def test_cycle_detection(self, validator):
        """Test cycle in DAG is detected"""
        workflow = {
            "name": "Cycle",
            "spec": {
                "nodes": [
                    {"id": "node1", "type": "task", "kind": "set", "value": "test1"},
                    {"id": "node2", "type": "task", "kind": "set", "value": "test2"}
                ],
                "edges": [
                    {"from": "_start", "to": "node1"},
                    {"from": "node1", "to": "node2"},
                    {"from": "node2", "to": "node1"},  # Cycle!
                    {"from": "node2", "to": "_end"}
                ]
            }
        }
        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("cycle" in str(e).lower() for e in errors)

    def test_orphaned_node_detection(self, validator):
        """Test orphaned node is detected"""
        workflow = {
            "name": "Orphaned",
            "spec": {
                "nodes": [
                    {"id": "node1", "type": "task", "kind": "set", "value": "test1"},
                    {"id": "node2", "type": "task", "kind": "set", "value": "test2"}
                ],
                "edges": [
                    {"from": "_start", "to": "node1"},
                    {"from": "node1", "to": "_end"}
                    # node2 is orphaned - no edges!
                ]
            }
        }
        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("orphan" in str(e).lower() for e in errors)

    def test_unknown_node_in_edge(self, validator):
        """Test edge referencing unknown node fails"""
        workflow = {
            "name": "Unknown",
            "spec": {
                "nodes": [
                    {"id": "node1", "type": "task", "kind": "set", "value": "test1"}
                ],
                "edges": [
                    {"from": "_start", "to": "node1"},
                    {"from": "node1", "to": "unknown_node"},  # Unknown!
                    {"from": "unknown_node", "to": "_end"}
                ]
            }
        }
        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("unknown" in str(e).lower() for e in errors)

    def test_empty_edges_allowed(self, validator):
        """Test empty edges list is allowed (fallback to sequential)"""
        workflow = {
            "name": "No Edges",
            "spec": {
                "nodes": [
                    {"id": "node1", "type": "task", "kind": "set", "value": "test1"}
                ],
                "edges": []
            }
        }
        errors = validator.validate(workflow)
        # Empty edges should not cause errors (fallback behavior)
        assert len([e for e in errors if "edge" in str(e).lower()]) == 0


class TestTemplateExpressions:
    """Test template expression validation"""

    def test_valid_expressions(self, validator):
        """Test valid template expressions pass"""
        workflow = {
            "name": "Valid Expressions",
            "spec": {
                "nodes": [
                    {
                        "id": "node1",
                        "type": "tool",
                        "tool": "test_tool",
                        "in": {
                            "field1": "${trigger.data.value}",
                            "field2": "${trigger.config.user_id}"
                        }
                    }
                ],
                "edges": [
                    {"from": "_start", "to": "node1"},
                    {"from": "node1", "to": "_end"}
                ]
            }
        }
        errors = validator.validate(workflow)
        # Should not have expression errors
        assert len([e for e in errors if "brace" in str(e).lower()]) == 0

    def test_unbalanced_braces(self, validator):
        """Test unbalanced braces are detected"""
        workflow = {
            "name": "Unbalanced",
            "spec": {
                "nodes": [
                    {
                        "id": "node1",
                        "type": "tool",
                        "tool": "test_tool",
                        "in": {
                            "field1": "${trigger.data.value"  # Missing }
                        }
                    }
                ],
                "edges": [
                    {"from": "_start", "to": "node1"},
                    {"from": "node1", "to": "_end"}
                ]
            }
        }
        errors = validator.validate(workflow)
        assert len(errors) > 0
        assert any("brace" in str(e).lower() for e in errors)


class TestExampleWorkflows:
    """Test all example workflows are valid"""

    @pytest.fixture
    def examples_dir(self):
        """Get examples directory"""
        return Path(__file__).parent.parent / "schema" / "examples" / "v2"

    def test_all_examples_valid(self, examples_dir, v2_schema):
        """Test all example workflows pass validation"""
        example_files = sorted(examples_dir.glob("*.json"))
        assert len(example_files) == 7, f"Expected 7 examples, found {len(example_files)}"

        for example_file in example_files:
            with open(example_file, encoding='utf-8') as f:
                workflow = json.load(f)

            is_valid, errors = validate_workflow_v2(workflow, v2_schema)
            assert is_valid, f"{example_file.name} failed validation: {errors}"

    def test_example_01_simple_email_alert(self, examples_dir, v2_schema):
        """Test example 1: Simple email alert"""
        with open(examples_dir / "01_simple_email_alert.json", encoding='utf-8') as f:
            workflow = json.load(f)

        is_valid, errors = validate_workflow_v2(workflow, v2_schema)
        assert is_valid, f"Errors: {errors}"

    def test_example_02_parallel_data_fetch(self, examples_dir, v2_schema):
        """Test example 2: Parallel data fetching"""
        with open(examples_dir / "02_parallel_data_fetch.json", encoding='utf-8') as f:
            workflow = json.load(f)

        is_valid, errors = validate_workflow_v2(workflow, v2_schema)
        assert is_valid, f"Errors: {errors}"

        # Verify it's actually parallel (multiple nodes from _start)
        spec = workflow["spec"]
        start_edges = [e for e in spec["edges"] if e["from"] == "_start"]
        assert len(start_edges) == 2, "Should have 2 parallel branches from _start"
