"""
Background tasks for memory extraction from chat sessions.

Extracts facts and context from completed chat sessions using Mem0,
enabling cross-session user memory for the Nexus agent.
"""

from datetime import datetime, timezone
from typing import List, Optional

from seer.config import config
from seer.database import User
from seer.database.workflow_models import (
    WorkflowChatMessage,
    WorkflowChatSession,
)
from seer.logger import get_logger
from seer.services.memory import UserMemoryService
from seer.worker.broker_instance import broker

logger = get_logger(__name__)


def _format_messages_for_extraction(messages: List[WorkflowChatMessage]) -> str:
    """
    Format chat messages into a conversation string for memory extraction.

    Mem0 works best with natural conversation format. We include both
    user and assistant messages to capture full context.

    Args:
        messages: List of WorkflowChatMessage objects

    Returns:
        Formatted conversation string
    """
    lines = []

    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        content = msg.content

        # Truncate very long messages to avoid exceeding token limits
        if len(content) > 2000:
            content = content[:1997] + "..."

        lines.append(f"{role_label}: {content}")

    return "\n\n".join(lines)


@broker.task
async def extract_session_memories(session_id: int) -> dict:
    """
    Extract and store memories from a completed chat session.

    Called automatically after chat session completes or manually for
    backfilling. Mem0 automatically extracts facts, preferences, and
    patterns from the conversation.

    Args:
        session_id: Database ID of the WorkflowChatSession

    Returns:
        Dict with extraction results
    """
    if not config.memory_enabled or not config.memory_extraction_enabled:
        logger.debug("Memory extraction disabled, skipping session %d", session_id)
        return {"skipped": True, "reason": "memory_extraction_disabled"}

    logger.info("Starting memory extraction for session %d", session_id)

    try:
        # Fetch session with related user and messages
        session = await WorkflowChatSession.get(id=session_id).prefetch_related(
            "user", "workflow"
        )

        # Fetch messages separately (for proper ordering)
        messages = await WorkflowChatMessage.filter(session_id=session_id).order_by("created_at").all()

        if not messages:
            logger.debug("No messages in session %d, skipping extraction", session_id)
            return {"skipped": True, "reason": "no_messages"}

        # Get user ID for memory scope
        user: User = session.user
        user_id = user.user_id  # Clerk user_id (string)

        # Format conversation for extraction
        conversation_text = _format_messages_for_extraction(messages)

        # Build metadata for filtering later
        metadata = {
            "session_id": session_id,
            "thread_id": session.thread_id,
            "workflow_id": session.workflow_id if session.workflow else None,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "message_count": len(messages),
            "source": "chat_session",
        }

        # Add session title if available
        if session.title:
            metadata["session_title"] = session.title

        # Extract memories using Mem0
        memory_service = UserMemoryService()
        result = await memory_service.add_memory(
            user_id=user_id,
            content=conversation_text,
            metadata=metadata,
        )

        if result is None:
            logger.warning("Memory extraction returned None for session %d", session_id)
            return {"success": False, "reason": "extraction_failed"}

        # Count extracted memories
        extracted_count = len(result.get("results", [])) if isinstance(result, dict) else 0

        logger.info(
            "Extracted %d memories from session %d for user %s",
            extracted_count,
            session_id,
            user_id,
        )

        return {
            "success": True,
            "session_id": session_id,
            "user_id": user_id,
            "memories_extracted": extracted_count,
        }

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Background task must not crash worker
        logger.error(
            "Memory extraction failed for session %d: %s",
            session_id,
            e,
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


@broker.task
async def backfill_user_memories(
    user_id_internal: int,
    limit: Optional[int] = 50,
) -> dict:
    """
    Backfill memories for a user from their recent chat sessions.

    Useful for enabling memory for existing users or after clearing
    memories for a fresh start.

    Args:
        user_id_internal: Database ID of the User
        limit: Maximum sessions to process

    Returns:
        Dict with backfill results
    """
    if not config.memory_enabled:
        return {"skipped": True, "reason": "memory_disabled"}

    logger.info("Starting memory backfill for user %d (limit=%d)", user_id_internal, limit)

    try:
        # Get user's recent completed sessions
        sessions = await WorkflowChatSession.filter(
            user_id=user_id_internal,
            current_execution_status="completed",
        ).order_by("-updated_at").limit(limit).all()

        results = []
        for session in sessions:
            result = await extract_session_memories(session.id)
            results.append(result)

        successful = sum(1 for r in results if r.get("success"))
        total_memories = sum(r.get("memories_extracted", 0) for r in results if r.get("success"))

        logger.info(
            "Memory backfill completed for user %d: %d/%d sessions processed, %d memories extracted",
            user_id_internal,
            successful,
            len(sessions),
            total_memories,
        )

        return {
            "success": True,
            "sessions_processed": len(sessions),
            "sessions_successful": successful,
            "total_memories_extracted": total_memories,
        }

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Background task
        logger.error("Memory backfill failed for user %d: %s", user_id_internal, e)
        return {"success": False, "error": str(e)}


__all__ = ["extract_session_memories", "backfill_user_memories"]
