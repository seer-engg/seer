"""Pydantic schemas for general chat API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatSessionCreate(BaseModel):
    title: Optional[str] = None


class ChatSessionUpdate(BaseModel):
    title: str


class ChatSendRequest(BaseModel):
    message: str
    model: Optional[str] = None
    generate_image: bool = False
    image_model: Optional[str] = None
    image_size: Optional[str] = "1024x1024"


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    model: Optional[str] = None
    image_urls: Optional[List[str]] = None
    thinking: Optional[List[str]] = None
    created_at: datetime


class ChatSessionResponse(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    current_execution_status: Optional[str] = None


class ChatSessionDetailResponse(ChatSessionResponse):
    messages: List[ChatMessageResponse] = []


class ChatStatusResponse(BaseModel):
    status: Optional[str] = None
    response: Optional[str] = None
    image_urls: Optional[List[str]] = None
    thinking: Optional[List[str]] = None
    error: Optional[Dict[str, Any]] = None


class ChatSendResponse(BaseModel):
    session_id: int
    execution_status: str
