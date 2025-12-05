"""
Runtime tool store - stores tool schemas and planned executions.
Global store (no worker IDs - workers are ephemeral).
"""
from typing import Dict, Optional
from models import ToolDefinition, ToolExecutionPlanBase

class RuntimeToolStore:
    """Global in-memory store for tool schemas and planned executions."""
    
    def __init__(self):
        # Key: tool_name, Value: ToolDefinition
        self._tool_schemas: Dict[str, ToolDefinition] = {}
        
        # Key: tool_name, Value: ToolExecutionPlanBase (latest planned execution)
        self._planned_executions: Dict[str, ToolExecutionPlanBase] = {}
    
    def store_tool_schema(self, tool_def: ToolDefinition):
        """Store a tool schema."""
        self._tool_schemas[tool_def.name] = tool_def
    
    def get_tool_schema(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool schema."""
        return self._tool_schemas.get(tool_name)
    
    def store_planned_execution(self, plan: ToolExecutionPlanBase):
        """Store planned execution for a tool."""
        if plan.tool_name:
            self._planned_executions[plan.tool_name] = plan
    
    def get_planned_execution(self, tool_name: str) -> Optional[ToolExecutionPlanBase]:
        """Get planned execution for a tool."""
        return self._planned_executions.get(tool_name)
    
    def clear_planned_execution(self, tool_name: str):
        """Clear planned execution after use."""
        if tool_name in self._planned_executions:
            del self._planned_executions[tool_name]

# Global instance
_runtime_tool_store = RuntimeToolStore()

