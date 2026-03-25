"""WhatsApp webhook routing and session management."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from seer.database import User
from seer.database.whatsapp_models import WhatsAppChatSession, WhatsAppMessageLog, WhatsAppUserLink
from seer.logger import get_logger
from seer.services.whatsapp.client import send_text_message

logger = get_logger(__name__)


def extract_messages(body: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract message entries from Meta webhook payload."""
    messages = []
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                if msg.get("type") == "text":
                    messages.append({
                        "message_id": msg["id"],
                        "phone": msg["from"],
                        "text": msg.get("text", {}).get("body", ""),
                    })
    return messages


async def is_duplicate(message_id: str) -> bool:
    """Check if we already processed this message."""
    exists = await WhatsAppMessageLog.filter(message_id=message_id).exists()
    if exists:
        return True
    await WhatsAppMessageLog.create(message_id=message_id)
    return False


async def get_user_for_phone(phone: str) -> Optional[User]:
    """Look up verified user link for a phone number."""
    link = await WhatsAppUserLink.filter(phone_number=phone, verified=True).first()
    if not link:
        return None
    await link.fetch_related("user")
    return link.user


async def get_or_create_chat_session(phone: str, user: User):
    """Get active WhatsApp chat session or create a new one."""
    from seer.api.chat.services import create_session  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    wa_session = await WhatsAppChatSession.filter(phone_number=phone, active=True).first()
    if wa_session:
        await wa_session.fetch_related("chat_session")
        return wa_session.chat_session

    chat_session = await create_session(user, title="WhatsApp Chat")
    await WhatsAppChatSession.create(
        phone_number=phone,
        chat_session=chat_session,
    )
    return chat_session


async def start_new_chat_session(phone: str, user: User):
    """Deactivate current session and start fresh."""
    await WhatsAppChatSession.filter(phone_number=phone, active=True).update(active=False)
    return await get_or_create_chat_session(phone, user)


async def route_message(phone: str, text: str, user: User) -> None:
    """Route incoming message to workflow trigger or chat."""
    stripped = text.strip()

    # /new → start fresh chat session
    if stripped.lower() == "/new":
        await start_new_chat_session(phone, user)
        await send_text_message(phone, "Started a new chat session.")
        return

    # /run <workflow-name> <input> → trigger workflow
    if stripped.lower().startswith("/run "):
        await _handle_workflow_command(phone, stripped, user)
        return

    # Default → general chat
    await _handle_chat_message(phone, stripped, user)


async def _handle_workflow_command(phone: str, text: str, user: User) -> None:
    """Parse /run command and dispatch workflow."""
    parts = text.split(maxsplit=2)
    if len(parts) < 2:
        await send_text_message(phone, "Usage: /run <workflow-name> [input]")
        return

    workflow_name = parts[1]
    workflow_input = parts[2] if len(parts) > 2 else ""

    from seer.database.workflow_models import (  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
        Workflow,
        WorkflowRun,
        WorkflowRunSource,
        WorkflowRunStatus,
        WorkflowVersion,
        WorkflowVersionStatus,
    )
    from seer.worker.tasks.workflows import workflow_execution_task  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    workflow = await Workflow.filter(
        user=user,
        title__icontains=workflow_name,
    ).first()

    if not workflow:
        await send_text_message(phone, f"Workflow '{workflow_name}' not found.")
        return

    published = await WorkflowVersion.filter(
        workflow=workflow, status=WorkflowVersionStatus.RELEASED
    ).first()
    if not published:
        await send_text_message(phone, f"Workflow '{workflow.title}' has no published version.")
        return

    try:
        inputs = {"text": workflow_input, "__whatsapp_reply_phone": phone}
        run = await WorkflowRun.create(
            user=user,
            workflow=workflow,
            workflow_version=published,
            spec=published.spec,
            inputs=inputs,
            config={},
            source=WorkflowRunSource.TRIGGER,
            status=WorkflowRunStatus.QUEUED,
        )
        await WorkflowRun.filter(id=run.id).update(thread_id=run.run_id)
        await workflow_execution_task.kiq(run_id=run.id, user_id=user.id)
        await send_text_message(phone, f"Workflow '{workflow.title}' started. Run ID: {run.run_id}")
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Must reply to user even on failure
        logger.error("Failed to trigger workflow via WhatsApp", exc_info=True)
        await send_text_message(phone, f"Failed to start workflow: {e}")


async def _handle_chat_message(phone: str, text: str, user: User) -> None:
    """Send message to general chat and queue response."""
    from seer.api.chat.services import save_message  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.worker.tasks.general_chat import general_chat_task  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    session = await get_or_create_chat_session(phone, user)
    await save_message(session_id=session.id, role="user", content=text)

    await general_chat_task.kiq(
        session_id=session.id,
        message=text,
        user_id=user.id,
    )


async def handle_unlinked_phone(phone: str) -> None:
    """Send a helpful message to an unlinked phone number."""
    try:
        await send_text_message(
            phone,
            "Your phone number isn't linked to a Seer account. "
            "Link it in Seer Settings > WhatsApp to get started.",
        )
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: Best-effort reply to unlinked user
        logger.warning("Could not send unlinked-phone message", extra={"phone": phone})
