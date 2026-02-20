"""
Memory tools for the Nexus agent.

These tools allow the agent to search and retrieve user memories
from previous sessions, enabling cross-session context awareness.
"""

from typing import List, Dict, Any
from langchain_core.tools import tool

from seer.config import config
from seer.agents.nexus.context import _current_thread_id, get_user_for_thread
from seer.logger import get_logger
from seer.services.memory import UserMemoryService

logger = get_logger(__name__)


def _format_memories_for_tool_response(memories: List[Dict[str, Any]]) -> str:
    """
    Format memories for tool response to the agent.

    Creates a structured, readable format that helps the agent
    understand and use the memory context.
    """
    if not memories:
        return "No relevant memories found for this user."

    lines = [f"Found {len(memories)} relevant memories:\n"]

    for i, memory in enumerate(memories, 1):
        text = memory.get("memory", memory.get("text", str(memory)))
        score = memory.get("score", 0.0)
        metadata = memory.get("metadata", {})

        # Format memory entry
        lines.append(f"{i}. {text}")

        # Add metadata context if available
        meta_parts = []
        if metadata.get("session_title"):
            meta_parts.append(f"Session: {metadata['session_title']}")
        if metadata.get("workflow_id"):
            meta_parts.append(f"Workflow: wf_{metadata['workflow_id']}")
        if score:
            meta_parts.append(f"Relevance: {score:.2f}")

        if meta_parts:
            lines.append(f"   [{', '.join(meta_parts)}]")

        lines.append("")  # Blank line between entries

    return "\n".join(lines)


def _format_session_search_results(memories: List[Dict[str, Any]]) -> str:
    """
    Format session-scoped memory search results.

    Groups memories by session for better context when searching
    past conversations.
    """
    if not memories:
        return "No matching past sessions found."

    # Group by session
    sessions: Dict[int, List[Dict[str, Any]]] = {}
    for memory in memories:
        session_id = memory.get("metadata", {}).get("session_id")
        if session_id:
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(memory)

    if not sessions:
        return "Found memories but none are associated with specific sessions."

    lines = [f"Found matches in {len(sessions)} past session(s):\n"]

    for session_id, session_memories in sessions.items():
        # Get session title from first memory's metadata
        title = session_memories[0].get("metadata", {}).get("session_title", "Untitled")
        lines.append(f"## Session: {title} (ID: {session_id})")

        for memory in session_memories[:3]:  # Limit to 3 per session
            text = memory.get("memory", memory.get("text", str(memory)))
            lines.append(f"  - {text}")

        lines.append("")

    return "\n".join(lines)


@tool
async def recall_memories(query: str, limit: int = 5) -> str:
    """
    Search your memories about this user for relevant context.

    Use this tool when you need to remember:
    - Past workflows the user has built
    - User preferences and patterns
    - Previous decisions or discussions
    - Technical context (e.g., "what database does the user use?")

    Args:
        query: What to search for in memories (semantic search)
        limit: Maximum number of memories to return (default: 5)

    Returns:
        Formatted list of relevant memories with context
    """
    if not config.memory_enabled:
        return "Memory feature is not enabled."

    thread_id = _current_thread_id.get()
    if not thread_id:
        logger.warning("recall_memories called without thread_id context")
        return "Unable to identify user context."

    user = await get_user_for_thread(thread_id)
    if not user:
        logger.warning("recall_memories: No user found for thread %s", thread_id)
        return "Unable to identify user."

    memory_service = UserMemoryService()
    memories = await memory_service.search(
        user_id=user.user_id,
        query=query,
        limit=limit,
    )

    logger.debug(
        "recall_memories for user %s, query='%s': found %d memories",
        user.user_id,
        query[:50],
        len(memories),
    )

    return _format_memories_for_tool_response(memories)


@tool
async def search_past_sessions(query: str, limit: int = 3) -> str:
    """
    Search past conversation sessions with this user.

    Use this tool to find specific past conversations about topics.
    Useful when the user references something from a previous session
    (e.g., "remember when we built that Slack workflow?").

    Args:
        query: What to search for in past sessions
        limit: Maximum number of sessions to return (default: 3)

    Returns:
        Summary of matching sessions with key memories from each
    """
    if not config.memory_enabled:
        return "Memory feature is not enabled."

    thread_id = _current_thread_id.get()
    if not thread_id:
        logger.warning("search_past_sessions called without thread_id context")
        return "Unable to identify user context."

    user = await get_user_for_thread(thread_id)
    if not user:
        logger.warning("search_past_sessions: No user found for thread %s", thread_id)
        return "Unable to identify user."

    memory_service = UserMemoryService()

    # Search with session filter to only get memories linked to sessions
    memories = await memory_service.search(
        user_id=user.user_id,
        query=query,
        limit=limit * 3,  # Get more to group by session
        filters={"has_session_id": True},
    )

    logger.debug(
        "search_past_sessions for user %s, query='%s': found %d memories",
        user.user_id,
        query[:50],
        len(memories),
    )

    return _format_session_search_results(memories)


@tool
async def get_user_profile() -> str:
    """
    Get a summary of what is known about this user from memory.

    Use this tool when starting a new conversation to understand
    the user's context, preferences, and history.

    Returns:
        Summary of user facts and preferences from memory
    """
    if not config.memory_enabled:
        return "Memory feature is not enabled."

    thread_id = _current_thread_id.get()
    if not thread_id:
        logger.warning("get_user_profile called without thread_id context")
        return "Unable to identify user context."

    user = await get_user_for_thread(thread_id)
    if not user:
        logger.warning("get_user_profile: No user found for thread %s", thread_id)
        return "Unable to identify user."

    memory_service = UserMemoryService()
    all_memories = await memory_service.get_all(user_id=user.user_id)

    if not all_memories:
        return "No stored memories for this user yet. This appears to be a new user or their first session."

    # Format as a profile summary
    lines = [f"User Profile (based on {len(all_memories)} stored memories):\n"]

    # Show up to 15 most relevant facts
    for memory in all_memories[:15]:
        text = memory.get("memory", memory.get("text", str(memory)))
        lines.append(f"- {text}")

    if len(all_memories) > 15:
        lines.append(f"\n... and {len(all_memories) - 15} more memories")

    return "\n".join(lines)


# Export tools for registration
memory_tools = [
    recall_memories,
    search_past_sessions,
    get_user_profile,
]

__all__ = [
    "recall_memories",
    "search_past_sessions",
    "get_user_profile",
    "memory_tools",
]
