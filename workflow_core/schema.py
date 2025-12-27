"""
Workflow schema definitions for controlled schema architecture (v1.0).
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class BlockType(str, Enum):
    """Supported block types in controlled schema v1.0."""
    
    TOOL = "tool"
    LLM = "llm"
    IF_ELSE = "if_else"
    FOR_LOOP = "for_loop"
    INPUT = "input"
    VARIABLE = "variable"


class BlockDefinition(BaseModel):
    """Definition of a workflow block."""
    
    id: str = Field(..., description="Unique block ID (ReactFlow node ID)")
    type: BlockType = Field(..., description="Block type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Block-specific configuration")
    oauth_scope: Optional[str] = Field(None, description="OAuth scope from frontend (for tool blocks)")
    position: Dict[str, float] = Field(..., description="Block position {x, y}")
    
    @field_validator('config')
    @classmethod
    def validate_config(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """Validate block-specific configuration."""
        block_type = info.data.get('type')
        
        if block_type == BlockType.TOOL:
            if 'tool_name' not in v:
                raise ValueError("tool_name is required in config for tool blocks")
            if 'tool_params' in v:
                raise ValueError("tool_params is deprecated; use 'params' instead.")
            params = v.get("params")
            if params is not None and not isinstance(params, dict):
                raise ValueError("'params' must be an object for tool blocks")
        elif block_type == BlockType.LLM:
            # system_prompt is optional, default to empty string
            if 'system_prompt' not in v:
                v['system_prompt'] = ""
            # user_prompt is required and must be non-empty
            if 'user_prompt' not in v or not v.get('user_prompt', '').strip():
                raise ValueError("user_prompt is required in config for LLM blocks and must be non-empty")
        elif block_type == BlockType.IF_ELSE:
            if 'condition' not in v:
                raise ValueError("condition is required in config for if_else blocks")
        elif block_type == BlockType.FOR_LOOP:
            array_mode = v.get("array_mode", "variable")
            item_var = v.get("item_var")
            if not item_var or not str(item_var).strip():
                raise ValueError("item_var is required in config for for_loop blocks")
            if array_mode not in ("variable", "literal"):
                raise ValueError("array_mode must be either 'variable' or 'literal' for for_loop blocks")
            if array_mode == "variable":
                array_var = v.get("array_variable") or v.get("array_var")
                if not array_var or not str(array_var).strip():
                    raise ValueError("array_variable is required when array_mode='variable' for for_loop blocks")
                v["array_variable"] = array_var
            else:
                array_literal = v.get("array_literal")
                if array_literal is None:
                    raise ValueError("array_literal is required when array_mode='literal' for for_loop blocks")
                if not isinstance(array_literal, list):
                    raise ValueError("array_literal must be a list when array_mode='literal' for for_loop blocks")
        elif block_type == BlockType.VARIABLE:
            if "input" not in v:
                raise ValueError("input is required in config for variable blocks")
            input_type = str(v.get("input_type") or "string").lower()
            if input_type not in ("string", "number", "array"):
                raise ValueError("input_type must be 'string', 'number', or 'array' for variable blocks")
            value = v.get("input")
            if input_type == "array":
                if not isinstance(value, list):
                    raise ValueError("Variable block input must be an array when input_type='array'")
            elif input_type == "number":
                if not isinstance(value, (int, float)):
                    raise ValueError("Variable block input must be a number when input_type='number'")
            else:
                if not isinstance(value, str):
                    raise ValueError("Variable block input must be a string when input_type='string'")
        
        return v


class EdgeDefinition(BaseModel):
    """Definition of a connection between blocks."""
    
    id: str = Field(..., description="Unique edge ID")
    source: str = Field(..., description="Source block ID")
    target: str = Field(..., description="Target block ID")
    branch: Optional[Literal["true", "false", "loop", "exit"]] = Field(
        default=None,
        description="Conditional branch hint (if/else or for loop branches)",
    )


class WorkflowSchema(BaseModel):
    """Complete workflow schema definition."""
    
    version: str = Field(default="1.0", description="Schema version")
    blocks: List[BlockDefinition] = Field(..., description="List of blocks in workflow")
    edges: List[EdgeDefinition] = Field(default_factory=list, description="List of edges connecting blocks")
    
    @field_validator('blocks')
    @classmethod
    def validate_blocks(cls, v: List[BlockDefinition]) -> List[BlockDefinition]:
        """Validate that blocks have unique IDs."""
        block_ids = [block.id for block in v]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("Block IDs must be unique")
        return v
    
    @field_validator('edges')
    @classmethod
    def validate_edges(cls, v: List[EdgeDefinition], info) -> List[EdgeDefinition]:
        """Validate that edges reference existing blocks."""
        blocks = info.data.get('blocks', [])
        block_ids = {block.id for block in blocks}
        
        for edge in v:
            if edge.source not in block_ids:
                raise ValueError(f"Edge source '{edge.source}' does not exist in blocks")
            if edge.target not in block_ids:
                raise ValueError(f"Edge target '{edge.target}' does not exist in blocks")
        
        return v


def validate_workflow_graph(graph_data: Dict[str, Any]) -> WorkflowSchema:
    """
    Validate and convert ReactFlow graph data to WorkflowSchema.
    
    Args:
        graph_data: ReactFlow format with 'nodes' and 'edges' arrays
        
    Returns:
        Validated WorkflowSchema
        
    Raises:
        ValueError: If graph data is invalid
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # Convert nodes to BlockDefinition
    blocks = []
    for node in nodes:
        block_type_str = node.get('type', '').replace('_', '_').lower()
        try:
            block_type = BlockType(block_type_str)
        except ValueError:
            raise ValueError(f"Invalid block type: {block_type_str}")
        
        data = node.get('data', {})
        position = node.get('position', {})
        
        block = BlockDefinition(
            id=node['id'],
            type=block_type,
            config=data.get('config', {}),
            oauth_scope=data.get('oauth_scope'),
            position={'x': position.get('x', 0), 'y': position.get('y', 0)},
        )
        blocks.append(block)
    
    def _determine_branch(edge_data: Dict[str, Any]) -> Optional[str]:
        valid_branches = {"true", "false", "loop", "exit"}
        branch = edge_data.get('data', {}).get('branch')
        if branch in valid_branches:
            return branch
        for handle_key in ("sourceHandle", "targetHandle"):
            legacy = edge_data.get(handle_key)
            if legacy in valid_branches:
                return legacy
        return None
    
    # Convert edges to EdgeDefinition
    edge_definitions = []
    for edge in edges:
        edge_def = EdgeDefinition(
            id=edge['id'],
            source=edge['source'],
            target=edge['target'],
            branch=_determine_branch(edge),
        )
        edge_definitions.append(edge_def)
    
    return WorkflowSchema(version="1.0", blocks=blocks, edges=edge_definitions)


__all__ = [
    "BlockType",
    "BlockDefinition",
    "EdgeDefinition",
    "WorkflowSchema",
    "validate_workflow_graph",
]

