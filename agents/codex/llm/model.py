from __future__ import annotations

import os
from typing import Iterable, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def get_chat_model(model: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    return ChatOpenAI(model=model_name, temperature=temperature)

