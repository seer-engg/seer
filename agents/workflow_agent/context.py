# Context variable to track current thread_id in tool execution
from typing import Optional, Dict, Any, List
from contextvars import ContextVar

_current_thread_id: ContextVar[Optional[str]] = ContextVar('_current_thread_id', default=None)

# Global workflow state context (thread-safe via thread_id key)
_workflow_state_context: Dict[str, Dict[str, Any]] = {}
_patch_ops_context: Dict[str, List[Dict[str, Any]]] = {}


def set_workflow_state_for_thread(thread_id: str, workflow_state: Dict[str, Any]) -> None:
    """Set workflow state for a specific thread."""
    _workflow_state_context[thread_id] = workflow_state

def get_workflow_state_for_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow state for a specific thread."""
    return _workflow_state_context.get(thread_id)


def append_patch_op_for_thread(thread_id: str, patch_op: Dict[str, Any]) -> None:
    """Store a workflow patch operation emitted by a tool for the given thread."""
    if not thread_id or not patch_op:
        return
    _patch_ops_context.setdefault(thread_id, []).append(patch_op)


def get_patch_ops_for_thread(thread_id: Optional[str], clear: bool = True) -> List[Dict[str, Any]]:
    """Return recorded patch ops for a thread, optionally clearing them."""
    if not thread_id:
        return []
    if clear:
        return _patch_ops_context.pop(thread_id, [])
    return _patch_ops_context.get(thread_id, [])


def clear_patch_ops_for_thread(thread_id: Optional[str]) -> None:
    """Remove any recorded patch ops for a specific thread."""
    if not thread_id:
        return
    _patch_ops_context.pop(thread_id, None)