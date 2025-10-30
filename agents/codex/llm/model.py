from __future__ import annotations

import os
from typing import Iterable, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def get_chat_model(model: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    llm = ChatOpenAI(
        model="gpt-5-codex",
        use_responses_api=True,             # <— key change
        output_version="responses/v1",      # nicer content blocks from Responses
        reasoning={"effort": "medium"},     # optional; supported by Responses models
    )
    # llm = ChatOpenAI(model='gpt-4o')
    return llm

