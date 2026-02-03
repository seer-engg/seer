"""
Clarification question tools for agent interaction.

Provides batch question support to reduce API round-trips when the agent
needs to gather multiple pieces of information from the user.
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import List

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel

from seer.logger import get_logger

logger = get_logger(__name__)


class QuestionType(str, Enum):
    """Type of clarification question."""
    SINGLE_CHOICE = "single_choice"
    MULTI_CHOICE = "multi_choice"


class QuestionOption(BaseModel):
    """Option for a clarification question."""
    value: str
    label: str
    is_wildcard: bool = False


@tool
def ask_clarification_questions(
    questions: List[dict],
) -> str:
    """
    Ask the user multiple clarification questions at once.

    Use this when you need to gather multiple pieces of information from the user.
    This is more efficient than asking questions one at a time, as it reduces
    the number of API round-trips.

    Args:
        questions: List of question objects, each containing:
            - question: The question text to display
            - question_type: "single_choice" or "multi_choice"
            - options: List of options with value, label, and optional is_wildcard
            - reasoning: Explain why you're asking this question
            - min_selections: Minimum number of selections (for multi-choice, default 1)
            - max_selections: Maximum number of selections (for multi-choice, optional)

    Returns:
        JSON string containing list of answers, one per question in the same order.
        Each answer has: {"question_id": "...", "selected_values": [...], "custom_input": "..."}

    Example:
        ask_clarification_questions([
            {
                "question": "Which email provider should we use?",
                "question_type": "single_choice",
                "options": [
                    {"value": "gmail", "label": "Gmail"},
                    {"value": "outlook", "label": "Outlook"},
                    {"value": "other", "label": "Other", "is_wildcard": True}
                ],
                "reasoning": "Need to know which email service to configure"
            },
            {
                "question": "Which notification channels should we enable?",
                "question_type": "multi_choice",
                "options": [
                    {"value": "slack", "label": "Slack"},
                    {"value": "email", "label": "Email"},
                    {"value": "sms", "label": "SMS"}
                ],
                "reasoning": "Need to know where to send notifications",
                "min_selections": 1
            }
        ])
    """
    if not questions:
        raise ValueError("At least one question is required")

    if len(questions) > 10:
        raise ValueError("Maximum 10 questions allowed per batch")

    # Build questions with unique IDs
    questions_payload = []
    for q in questions:
        question_id = f"q_{uuid.uuid4().hex[:8]}"
        options = q.get("options", [])

        questions_payload.append({
            "question_id": question_id,
            "question": q.get("question", ""),
            "question_type": q.get("question_type", "single_choice"),
            "options": [
                opt.model_dump() if hasattr(opt, 'model_dump') else opt
                for opt in options
            ],
            "min_selections": q.get("min_selections", 1),
            "max_selections": q.get("max_selections"),
            "reasoning": q.get("reasoning", ""),
        })

    # Build interrupt payload for batch questions
    interrupt_payload = {
        "type": "clarification_questions",  # Note: plural
        "questions": questions_payload,
    }

    logger.info(
        "Agent asking %d clarification questions: question_ids=%s",
        len(questions_payload),
        [q["question_id"] for q in questions_payload]
    )

    # Trigger LangGraph interrupt - execution pauses here until resumed
    answers = interrupt(interrupt_payload)

    logger.info("Clarification questions answered: %d answers received", len(answers) if answers else 0)

    # When resumed, answers contains list of user responses
    # Format: [{"question_id": "...", "selected_values": [...], "custom_input": "..."}, ...]
    return json.dumps(answers)
