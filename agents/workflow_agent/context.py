# Context variable to track current thread_id in tool execution
from typing import Optional, Dict, Any
from contextvars import ContextVar

_current_thread_id: ContextVar[Optional[str]] = ContextVar('_current_thread_id', default=None)

# Global workflow state context (thread-safe via thread_id key)
_workflow_state_context: Dict[str, Dict[str, Any]] = {}
_proposed_specs_context: Dict[str, Dict[str, Any]] = {}


def set_workflow_state_for_thread(thread_id: str, workflow_state: Dict[str, Any]) -> None:
    """Set workflow state for a specific thread."""
    _workflow_state_context[thread_id] = workflow_state

def get_workflow_state_for_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow state for a specific thread."""
    return _workflow_state_context.get(thread_id)


def set_proposed_spec_for_thread(thread_id: str, proposal: Dict[str, Any]) -> None:
    """Persist the latest workflow proposal payload (spec + metadata) for a thread."""
    if not thread_id:
        return
    _proposed_specs_context[thread_id] = proposal


def get_proposed_spec_for_thread(thread_id: Optional[str], clear: bool = True) -> Optional[Dict[str, Any]]:
    """Return the most recent workflow proposal payload for a thread."""
    if not thread_id:
        return None
    if clear:
        return _proposed_specs_context.pop(thread_id, None)
    return _proposed_specs_context.get(thread_id)


def clear_proposed_spec_for_thread(thread_id: Optional[str]) -> None:
    """Remove stored workflow spec for a thread."""
    if not thread_id:
        return
    _proposed_specs_context.pop(thread_id, None)