# Context variable to track current thread_id in tool execution
from typing import Optional, Dict, Any
from contextvars import ContextVar

_current_thread_id: ContextVar[Optional[str]] = ContextVar('_current_thread_id', default=None)

# Global workflow state context (thread-safe via thread_id key)
_workflow_state_context: Dict[str, Dict[str, Any]] = {}


def set_workflow_state_for_thread(thread_id: str, workflow_state: Dict[str, Any]) -> None:
    """Set workflow state for a specific thread."""
    _workflow_state_context[thread_id] = workflow_state

def get_workflow_state_for_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow state for a specific thread."""
    return _workflow_state_context.get(thread_id)