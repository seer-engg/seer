"""
Clarification question tools for agent interaction.
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import List, Optional

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

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


class AskClarificationInput(BaseModel):
    """Input schema for asking clarification questions."""
    question: str = Field(..., description="The question to ask")
    question_type: QuestionType = Field(..., description="Type of question")
    options: List[QuestionOption] = Field(..., description="Available options")
    reasoning: str = Field(..., description="Why you're asking this question")
    min_selections: int = Field(default=1)
    max_selections: Optional[int] = Field(default=None)


@tool
def ask_clarification_question(
    question: str,
    question_type: QuestionType,
    options: List[QuestionOption],
    reasoning: str,
    *,
    min_selections: int = 1,
    max_selections: Optional[int] = None,
) -> str:
    """
    Ask the user a structured clarification question.

    Use this when you need the user to choose from specific options to proceed.
    The question will interrupt the workflow and wait for user response.

    Args:
        question: The question text to display
        question_type: "single_choice" or "multi_choice"
        options: List of options with value, label, and optional is_wildcard
        reasoning: Explain why you're asking this question
        min_selections: Minimum number of selections (for multi-choice)
        max_selections: Maximum number of selections (for multi-choice)

    Returns:
        The user's answer as a JSON string with selected values and optional custom input

    Examples:
        Single-choice with wildcard:
        ask_clarification_question(
            question="Which email provider should we use?",
            question_type="single_choice",
            options=[
                {"value": "gmail", "label": "Gmail"},
                {"value": "outlook", "label": "Outlook"},
                {"value": "other", "label": "Other (specify)", "is_wildcard": True}
            ],
            reasoning="Need to know which email integration to configure"
        )

        Multi-choice:
        ask_clarification_question(
            question="Which integrations should we enable?",
            question_type="multi_choice",
            options=[
                {"value": "gmail", "label": "Gmail"},
                {"value": "slack", "label": "Slack"},
                {"value": "github", "label": "GitHub"}
            ],
            reasoning="User wants multiple integrations but didn't specify which ones",
            min_selections=1,
            max_selections=3
        )
    """
    # Generate unique question ID
    question_id = f"q_{uuid.uuid4().hex[:8]}"

    # Build interrupt payload
    interrupt_payload = {
        "type": "clarification_question",
        "question_id": question_id,
        "question": question,
        "question_type": question_type,
        "options": [opt.model_dump() if hasattr(opt, 'model_dump') else opt for opt in options],
        "min_selections": min_selections,
        "max_selections": max_selections,
        "reasoning": reasoning,
    }

    logger.info("Agent asking clarification question: question_id=%s, type=%s", question_id, question_type)

    # Trigger LangGraph interrupt - execution pauses here until resumed
    answer = interrupt(interrupt_payload)

    logger.info("Clarification question answered: question_id=%s, answer=%s", question_id, answer)

    # When resumed, answer contains user's response
    # Format: {"selected_values": ["value1", "value2"], "custom_input": "..."}
    return json.dumps(answer)
