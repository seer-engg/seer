"""Service layer for general chat sessions."""
from __future__ import annotations

import uuid
from typing import List, Optional

from seer.database import User
from seer.database.chat_models import GeneralChatMessage, GeneralChatSession


async def create_session(user: User, title: Optional[str] = None) -> GeneralChatSession:
    thread_id = str(uuid.uuid4())
    return await GeneralChatSession.create(user=user, thread_id=thread_id, title=title)


async def list_sessions(user: User, limit: int = 50, offset: int = 0) -> List[GeneralChatSession]:
    return await GeneralChatSession.filter(user=user).offset(offset).limit(limit).all()


async def get_session(session_id: int, user: User) -> Optional[GeneralChatSession]:
    return await GeneralChatSession.get_or_none(id=session_id, user=user)


async def get_session_with_messages(session_id: int, user: User):
    session = await GeneralChatSession.get_or_none(id=session_id, user=user)
    if not session:
        return None, []
    messages = await GeneralChatMessage.filter(session=session).order_by("created_at").all()
    return session, messages


async def delete_session(session_id: int, user: User) -> bool:
    session = await GeneralChatSession.get_or_none(id=session_id, user=user)
    if not session:
        return False
    await GeneralChatMessage.filter(session=session).delete()
    await session.delete()
    return True


async def update_session_title(session_id: int, user: User, title: str) -> Optional[GeneralChatSession]:
    session = await GeneralChatSession.get_or_none(id=session_id, user=user)
    if not session:
        return None
    session.title = title
    await session.save(update_fields=["title"])
    return session


async def save_message(  # pylint: disable=too-many-positional-arguments  # Reason: Message fields are all distinct required params
    session_id: int,
    role: str,
    content: str,
    model: Optional[str] = None,
    image_urls: Optional[list] = None,
    thinking: Optional[list] = None,
) -> GeneralChatMessage:
    session = await GeneralChatSession.get(id=session_id)
    return await GeneralChatMessage.create(
        session=session, role=role, content=content, model=model, image_urls=image_urls, thinking=thinking,
    )


async def auto_generate_title(session_id: int) -> None:
    """Generate title from first user message if no title set."""
    session = await GeneralChatSession.get(id=session_id)
    if session.title:
        return
    first_msg = await GeneralChatMessage.filter(session=session, role="user").order_by("created_at").first()
    if first_msg:
        title = first_msg.content[:100].strip()
        if len(first_msg.content) > 100:
            title += "..."
        session.title = title
        await session.save(update_fields=["title"])
