"""
Workflow state schema for LangGraph execution.

Defines the state structure that flows through the workflow graph.
"""
from typing_extensions import TypedDict, Annotated
from typing import Any, Dict, List, Optional
import operator


class WorkflowState(TypedDict):
    """LangGraph state for workflow execution.
    
    This state is passed between nodes and updated by each node execution.
    The Annotated types with operator.or_ allow state updates to merge
    dictionaries rather than replace them.
    """
    # Input data from user (via execution panel form)
    input_data: Dict[str, Any]
    
    # Global block outputs: block_id -> {handle_id: value}
    # Example: {"block_a": {"output": "result", "email": "user@example.com"}}
    # This allows any block to reference any previous block's output
    block_outputs: Annotated[Dict[str, Dict[str, Any]], operator.or_]
    
    # Mapping of block_id -> list of alias strings that can be used in templates
    block_aliases: Annotated[Dict[str, List[str]], operator.or_]
    
    # Execution metadata
    execution_id: Optional[int]
    user_id: Optional[str]
    
    # Loop state (for for_loop blocks)
    # Structure: {
    #   "array_var": "items",
    #   "item_var": "item",
    #   "current_index": 0,
    #   "items": [...],
    #   "results": [...]
    # }
    loop_state: Optional[Dict[str, Any]]


__all__ = ["WorkflowState"]

