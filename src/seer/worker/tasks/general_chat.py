"""Background task for general chat execution."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from seer.config import config
from seer.database.chat_models import GeneralChatMessage, GeneralChatSession
from seer.database.workflow_models import ChatExecutionStatus
from seer.llm import get_llm
from seer.worker.broker_instance import broker

logger = logging.getLogger(__name__)


def _build_langchain_messages(messages: List[GeneralChatMessage]):
    """Convert DB messages to LangChain message format."""
    from langchain_core.messages import AIMessage, HumanMessage  # pylint: disable=import-outside-toplevel  # Reason: Late import

    lc_messages = []
    for msg in messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))
    return lc_messages


@broker.task
async def general_chat_task(  # pylint: disable=too-many-positional-arguments  # Reason: Task interface requires all params
    session_id: int,
    message: str,
    user_id: int,
    model: Optional[str] = None,
    generate_image: bool = False,
    image_model: Optional[str] = None,
    image_size: str = "1024x1024",
) -> None:
    """Execute general chat completion in background."""
    logger.info("Starting general chat task", extra={"session_id": session_id, "user_id": user_id})

    session = await GeneralChatSession.get(id=session_id)
    session.current_execution_status = ChatExecutionStatus.RUNNING
    session.current_execution_started_at = datetime.now(timezone.utc)
    await session.save(update_fields=["current_execution_status", "current_execution_started_at"])

    try:
        # Build conversation history
        history = await GeneralChatMessage.filter(session=session).order_by("created_at").all()
        lc_messages = _build_langchain_messages(history)

        # Simple chat completion (no agent/tools)
        llm = get_llm(model=model or config.default_llm_model)
        result = await llm.ainvoke(lc_messages)
        response_text = result.content if hasattr(result, "content") else str(result)

        # Image generation if requested
        image_urls = None
        if generate_image:
            try:
                from seer.services.image_gen import generate_image as gen_img  # pylint: disable=import-outside-toplevel  # Reason: Optional feature

                img_url, _ = await gen_img(prompt=message, model=image_model or "sourceful/riverflow-v2-fast", size=image_size)
                if img_url:
                    image_urls = [img_url]
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Image gen failure shouldn't fail the chat
                logger.warning("Image generation failed: %s", e)

        # Save assistant message
        from seer.api.chat.services import save_message  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

        await save_message(
            session_id=session_id, role="assistant", content=response_text,
            model=model or config.default_llm_model, image_urls=image_urls,
        )

        session.current_execution_status = ChatExecutionStatus.COMPLETED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_error = None
        await session.save(update_fields=["current_execution_status", "current_execution_finished_at", "current_execution_error"])

        logger.info("General chat task completed", extra={"session_id": session_id})

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Background task must catch all exceptions
        logger.error("General chat task failed", exc_info=True, extra={"session_id": session_id, "error": str(e)})
        session.current_execution_status = ChatExecutionStatus.FAILED
        session.current_execution_finished_at = datetime.now(timezone.utc)
        session.current_execution_error = {"type": "execution_error", "detail": str(e), "status": 500}
        await session.save(update_fields=["current_execution_status", "current_execution_finished_at", "current_execution_error"])
