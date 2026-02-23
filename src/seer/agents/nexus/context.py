# Context variable to track current thread_id in tool execution
from typing import Optional, TYPE_CHECKING
from contextvars import ContextVar

_current_thread_id: ContextVar[Optional[str]] = ContextVar('_current_thread_id', default=None)

if TYPE_CHECKING:
    from seer.api.user.models import User
    from seer.database.workflow_models import WorkflowChatSession, WorkflowDiscoveryChatSession


async def get_user_for_thread(thread_id: str) -> Optional["User"]:
    """
    Retrieve User object for a thread from the database.

    Looks up the thread in WorkflowChatSession or WorkflowDiscoveryChatSession
    and returns the associated user.
    """
    # Import here to avoid circular dependency at module load time
    from seer.database.workflow_models import WorkflowChatSession, WorkflowDiscoveryChatSession  # pylint: disable=import-outside-toplevel # Reason: Avoid circular dependency

    # Try chat session first
    session = await WorkflowChatSession.get_or_none(thread_id=thread_id).prefetch_related('user')
    if session:
        return session.user

    # Try discovery session
    discovery = await WorkflowDiscoveryChatSession.get_or_none(thread_id=thread_id).prefetch_related('user')
    if discovery:
        return discovery.user

    return None
