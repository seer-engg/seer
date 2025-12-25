"""
Tests for workflow validation guardrails.

These tests ensure that invalid workflow proposals are caught early
and don't reach the accept/reject stage.
"""
import pytest
from .chat_agent import _validate_block_config
from .schema import validate_workflow_graph, BlockType
from fastapi import HTTPException


class TestBlockConfigValidation:
    """Test block configuration validation in agent tools."""
    
    def test_validate_tool_block_without_tool_name(self):
        """Test that tool blocks without tool_name are rejected."""
        error = _validate_block_config("tool", {}, "test-block")
        assert error is not None
        assert "tool_name" in error.lower()
        assert "required" in error.lower()
    
    def test_validate_tool_block_with_tool_name(self):
        """Test that tool blocks with tool_name are accepted."""
        error = _validate_block_config("tool", {"tool_name": "gmail_read_emails"}, "test-block")
        assert error is None
    
    def test_validate_llm_block_without_user_prompt(self):
        """Test that LLM blocks without user_prompt are rejected."""
        error = _validate_block_config("llm", {}, "test-block")
        assert error is not None
        assert "user_prompt" in error.lower()
        assert "required" in error.lower()
    
    def test_validate_llm_block_with_empty_user_prompt(self):
        """Test that LLM blocks with empty user_prompt are rejected."""
        error = _validate_block_config("llm", {"user_prompt": ""}, "test-block")
        assert error is not None
        assert "user_prompt" in error.lower()
    
    def test_validate_llm_block_with_user_prompt(self):
        """Test that LLM blocks with user_prompt are accepted."""
        error = _validate_block_config("llm", {"user_prompt": "Summarize this"}, "test-block")
        assert error is None
    
    def test_validate_if_else_block_without_condition(self):
        """Test that if_else blocks without condition are rejected."""
        error = _validate_block_config("if_else", {}, "test-block")
        assert error is not None
        assert "condition" in error.lower()
        assert "required" in error.lower()
    
    def test_validate_if_else_block_with_condition(self):
        """Test that if_else blocks with condition are accepted."""
        error = _validate_block_config("if_else", {"condition": "x > 0"}, "test-block")
        assert error is None
    
    def test_validate_for_loop_block_without_vars(self):
        """Test that for_loop blocks without array_var/item_var are rejected."""
        error = _validate_block_config("for_loop", {}, "test-block")
        assert error is not None
        assert ("array_var" in error.lower() or "item_var" in error.lower())
    
    def test_validate_for_loop_block_with_vars(self):
        """Test that for_loop blocks with array_var/item_var are accepted."""
        error = _validate_block_config("for_loop", {
            "array_var": "items",
            "item_var": "item"
        }, "test-block")
        assert error is None
    
    def test_validate_invalid_block_type(self):
        """Test that invalid block types are rejected."""
        error = _validate_block_config("invalid_type", {}, "test-block")
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
        from .services import preview_patch_ops
        
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

