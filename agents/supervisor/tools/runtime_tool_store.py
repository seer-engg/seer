"""
Runtime tool store - stores tool schemas and planned executions.
Tool schemas are global cache. Planned executions are thread-scoped (NOT in graph state).
"""
from typing import Dict, Optional, Any
from agents.supervisor.models import ToolDefinition

class RuntimeToolStore:
    """Global in-memory store for tool schemas and thread-scoped planned executions."""
    
    def __init__(self):
        # Key: tool_name, Value: ToolDefinition (global cache)
        self._tool_schemas: Dict[str, ToolDefinition] = {}
        
        # Key: thread_id -> tool_name -> plan (thread-scoped planned executions)
        # Format: Dict[str, Dict[str, Dict[str, Any]]]
        self._planned_executions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    def store_tool_schema(self, tool_def: ToolDefinition):
        """Store a tool schema (global cache)."""
        self._tool_schemas[tool_def.name] = tool_def
    
    def get_tool_schema(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool schema (global cache)."""
        return self._tool_schemas.get(tool_name)
    
    def store_planned_execution(self, plan: Dict[str, Any], thread_id: Optional[str] = None):
        """Store planned execution for a tool (thread-scoped).
        
        Args:
            plan: Dict with keys: tool_name, reasoning, params
            thread_id: Optional thread ID. If None, uses "default"
        """
        tool_name = plan.get("tool_name")
        if not tool_name:
            return
        
        thread_id = thread_id or "default"
        
        # Initialize thread dict if needed
        if thread_id not in self._planned_executions:
            self._planned_executions[thread_id] = {}
        
        self._planned_executions[thread_id][tool_name] = plan
    
    def get_planned_execution(self, tool_name: str, thread_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get planned execution for a tool (thread-scoped).
        
        Args:
            tool_name: Name of the tool
            thread_id: Optional thread ID. If None, uses "default"
        
        Returns:
            Dict with keys: tool_name, reasoning, params, or None if not found
        """
        thread_id = thread_id or "default"
        thread_executions = self._planned_executions.get(thread_id, {})
        return thread_executions.get(tool_name)
    
    def clear_planned_execution(self, tool_name: str, thread_id: Optional[str] = None):
        """Clear planned execution after use (thread-scoped).
        
        Args:
            tool_name: Name of the tool
            thread_id: Optional thread ID. If None, uses "default"
        """
        thread_id = thread_id or "default"
        if thread_id in self._planned_executions:
            if tool_name in self._planned_executions[thread_id]:
                del self._planned_executions[thread_id][tool_name]
    
    def clear_thread_executions(self, thread_id: Optional[str] = None):
        """Clear all planned executions for a thread (cleanup).
        
        Args:
            thread_id: Optional thread ID. If None, uses "default"
        """
        thread_id = thread_id or "default"
        if thread_id in self._planned_executions:
            del self._planned_executions[thread_id]

# Global instance
_runtime_tool_store = RuntimeToolStore()

