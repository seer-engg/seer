from __future__ import annotations

import os
from typing import Iterable, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def get_chat_model(model: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    return ChatOpenAI(model=model_name, temperature=temperature)


def generate_plan_steps(model: ChatOpenAI, request: str, repo_context: str | None = None) -> List[str]:
    system_parts: List[str] = [
        "You are an expert software planner. Draft specific steps to fulfill the request.",
        "Focus on clarity, minimal steps, and feasibility.",
    ]
    if repo_context:
        system_parts.append("Repository summary is provided. Consider existing structure.")

    messages: List = [
        SystemMessage("\n".join(system_parts)),
        HumanMessage(
            (
                "Request:\n" + request
                + ("\n\nRepo summary:\n" + repo_context if repo_context else "")
                + "\n\nReturn 3-7 bullet steps."
            )
        ),
    ]
    response = model.invoke(messages)
    text = (response.content or "").strip()
    lines = [l.strip(" -\t") for l in text.splitlines() if l.strip()]
    # Keep only bullet-y looking lines or non-empty lines as steps
    steps = [l for l in lines if l]
    # Fallback if the model answered in a paragraph
    if len(steps) <= 1:
        steps = [text] if text else ["Review codebase and prepare a plan."]
    return steps[:7]
