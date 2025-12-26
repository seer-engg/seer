"""
Tests for workflow validation guardrails.

These tests ensure that invalid workflow proposals are caught early
and don't reach the accept/reject stage.
"""
import pytest
from fastapi import HTTPException

from agents.workflow_agent.validation import validate_block_config
from api.workflows.services import preview_patch_ops
from workflow_core.schema import BlockDefinition, BlockType, validate_workflow_graph
from workflow_core.state import WorkflowState
import workflow_core.nodes as workflow_nodes


class TestBlockConfigValidation:
    """Test block configuration validation in agent tools."""
    
    def test_validate_tool_block_without_tool_name(self):
        """Test that tool blocks without tool_name are rejected."""
        error = validate_block_config("tool", {}, "test-block")
        assert error is not None
        assert "tool_name" in error.lower()
        assert "required" in error.lower()
    
    def test_validate_tool_block_with_tool_name(self):
        """Test that tool blocks with tool_name are accepted."""
        error = validate_block_config("tool", {"tool_name": "gmail_read_emails"}, "test-block")
        assert error is None
    
    def test_validate_llm_block_without_user_prompt(self):
        """Test that LLM blocks without user_prompt are rejected."""
        error = validate_block_config("llm", {}, "test-block")
        assert error is not None
        assert "user_prompt" in error.lower()
        assert "required" in error.lower()
    
    def test_validate_llm_block_with_empty_user_prompt(self):
        """Test that LLM blocks with empty user_prompt are rejected."""
        error = validate_block_config("llm", {"user_prompt": ""}, "test-block")
        assert error is not None
        assert "user_prompt" in error.lower()
    
    def test_validate_llm_block_with_user_prompt(self):
        """Test that LLM blocks with user_prompt are accepted."""
        error = validate_block_config("llm", {"user_prompt": "Summarize this"}, "test-block")
        assert error is None
    
    def test_validate_if_else_block_without_condition(self):
        """Test that if_else blocks without condition are rejected."""
        error = validate_block_config("if_else", {}, "test-block")
        assert error is not None
        assert "condition" in error.lower()
        assert "required" in error.lower()
    
    def test_validate_if_else_block_with_condition(self):
        """Test that if_else blocks with condition are accepted."""
        error = validate_block_config("if_else", {"condition": "x > 0"}, "test-block")
        assert error is None
    
    def test_validate_for_loop_block_without_vars(self):
        """Test that for_loop blocks without array_var/item_var are rejected."""
        error = validate_block_config("for_loop", {}, "test-block")
        assert error is not None
        assert ("array_var" in error.lower() or "item_var" in error.lower())
    
    def test_validate_for_loop_block_with_vars(self):
        """Test that for_loop blocks with array_var/item_var are accepted."""
        error = validate_block_config("for_loop", {
            "array_variable": "items",
            "item_var": "item"
        }, "test-block")
        assert error is None
    
    def test_validate_variable_block_without_input(self):
        """Variable blocks must include an input value."""
        error = validate_block_config("variable", {"input_type": "string"}, "test-block")
        assert error is not None
        assert "input" in error.lower()
    
    def test_validate_variable_block_with_string_input(self):
        """Variable blocks accept string literals."""
        error = validate_block_config("variable", {"input_type": "string", "input": "value"}, "test-block")
        assert error is None
    
    def test_validate_variable_block_with_invalid_array_payload(self):
        """Variable blocks reject non-array payloads when array type selected."""
        error = validate_block_config("variable", {"input_type": "array", "input": "not-a-list"}, "test-block")
        assert error is not None
        assert "array" in error.lower()
    
    def test_validate_invalid_block_type(self):
        """Test that invalid block types are rejected."""
        error = validate_block_config("invalid_type", {}, "test-block")
        assert error is not None
        assert "invalid" in error.lower()


class TestWorkflowGraphValidation:
    """Test workflow graph validation."""
    
    def test_validate_graph_with_tool_block_missing_tool_name(self):
        """Test that graphs with tool blocks missing tool_name are rejected."""
        graph_data = {
            "nodes": [
                {
                    "id": "block-1",
                    "type": "tool",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Test Tool",
                        "config": {}  # Missing tool_name
                    }
                }
            ],
            "edges": []
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_workflow_graph(graph_data)
        
        assert "tool_name" in str(exc_info.value).lower()
    
    def test_validate_graph_with_valid_tool_block(self):
        """Test that graphs with valid tool blocks are accepted."""
        graph_data = {
            "nodes": [
                {
                    "id": "block-1",
                    "type": "tool",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Test Tool",
                        "config": {"tool_name": "gmail_read_emails"}
                    }
                }
            ],
            "edges": []
        }
        
        # Should not raise
        schema = validate_workflow_graph(graph_data)
        assert schema is not None
        assert len(schema.blocks) == 1
    
    def test_validate_graph_with_llm_block_missing_user_prompt(self):
        """Test that graphs with LLM blocks missing user_prompt are rejected."""
        graph_data = {
            "nodes": [
                {
                    "id": "block-1",
                    "type": "llm",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Test LLM",
                        "config": {}  # Missing user_prompt
                    }
                }
            ],
            "edges": []
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_workflow_graph(graph_data)
        
        assert "user_prompt" in str(exc_info.value).lower()


class TestProposalErrorHandling:
    """Test that proposal errors are properly handled."""
    
    def test_preview_patch_ops_rejects_invalid_config(self):
        """Test that preview_patch_ops raises error for invalid configs."""
        # Create patch ops that add a tool block without tool_name
        patch_ops = [
            {
                "op": "add_node",
                "node_id": "block-1",
                "node": {
                    "id": "block-1",
                    "type": "tool",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Test Tool",
                        "config": {}  # Missing tool_name
                    }
                }
            }
        ]
        
        with pytest.raises((ValueError, HTTPException)) as exc_info:
            preview_patch_ops({}, patch_ops)
        
        # Should raise validation error
        assert "tool_name" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()

    def test_preview_patch_ops_rejects_tool_params_key(self):
        """Ensure legacy tool_params payloads are rejected."""
        patch_ops = [
            {
                "op": "add_node",
                "node_id": "block-1",
                "node": {
                    "id": "block-1",
                    "type": "tool",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Legacy Tool",
                        "config": {
                            "tool_name": "gmail_read_emails",
                            "tool_params": {"max_results": 3},
                        },
                    },
                },
            }
        ]

        with pytest.raises(ValueError) as exc_info:
            preview_patch_ops({}, patch_ops)

        assert "tool_params" in str(exc_info.value)

    def test_preview_patch_ops_accepts_params_payload(self):
        """Ensure params payloads flow through unchanged."""
        patch_ops = [
            {
                "op": "add_node",
                "node_id": "block-1",
                "node": {
                    "id": "block-1",
                    "type": "tool",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Max 3",
                        "config": {
                            "tool_name": "gmail_read_emails",
                            "params": {"max_results": 3},
                        },
                    },
                },
            }
        ]

        preview = preview_patch_ops({}, patch_ops)
        assert preview["nodes"][0]["data"]["config"]["params"]["max_results"] == 3


class TestToolNodeExecution:
    """Tests for runtime execution semantics."""

    @pytest.mark.asyncio
    async def test_tool_node_uses_configured_params(self, monkeypatch):
        """Tool node should pass through params to the executor."""
        block = BlockDefinition(
            id="block-1",
            type=BlockType.TOOL,
            config={"tool_name": "gmail_read_emails", "params": {"max_results": 3}},
            position={"x": 0, "y": 0},
        )
        state: WorkflowState = {
            "input_data": {},
            "block_outputs": {},
            "block_aliases": {},
            "execution_id": None,
            "user_id": "user-123",
            "loop_state": {},
        }

        class DummyTool:
            def get_parameters_schema(self):
                return {"properties": {"max_results": {"type": "integer"}}}

        monkeypatch.setattr("shared.tools.base.get_tool", lambda _: DummyTool())

        async def fake_resolve_inputs(*_args, **_kwargs):
            return {}

        monkeypatch.setattr(workflow_nodes, "resolve_inputs", fake_resolve_inputs)
        monkeypatch.setattr(workflow_nodes, "build_variable_map", lambda _state: {})

        async def fake_user_get(*_args, **_kwargs):
            class DummyUser:
                id = "user-123"
            return DummyUser()

        monkeypatch.setattr(workflow_nodes.User, "get", fake_user_get)

        captured = {}

        async def fake_execute_tool_with_oauth(**kwargs):
            captured.update(kwargs)
            return [{"id": "message"}]

        monkeypatch.setattr(workflow_nodes, "execute_tool_with_oauth", fake_execute_tool_with_oauth)

        result_state = await workflow_nodes.tool_node(state, block, {}, user_id="user-123")

        assert captured["arguments"]["max_results"] == 3
        assert "block-1" in result_state["block_outputs"]

