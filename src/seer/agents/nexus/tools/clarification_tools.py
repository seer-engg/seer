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
    RESOURCE_PICKER = "resource_picker"


class QuestionOption(BaseModel):
    """Option for a clarification question."""
    value: str
    label: str
    is_wildcard: bool = False


def _build_resource_picker_payload(q: dict, question_payload: dict) -> None:
    """Add resource picker specific fields to question payload."""
    question_payload["provider"] = q.get("provider", "")
    question_payload["resource_type"] = q.get("resource_type", "")
    question_payload["display_field"] = q.get("display_field", "name")
    question_payload["value_field"] = q.get("value_field", "id")
    question_payload["search_enabled"] = q.get("search_enabled", True)
    question_payload["hierarchy"] = q.get("hierarchy", False)
    question_payload["depends_on_field"] = q.get("depends_on_field")
    question_payload["options"] = []

    if q.get("depends_on"):
        question_payload["_depends_on_ref"] = q.get("depends_on")


def _build_choice_question_payload(q: dict, question_payload: dict) -> None:
    """Add choice question specific fields to question payload."""
    options = q.get("options", [])
    question_payload["options"] = [
        opt.model_dump() if hasattr(opt, 'model_dump') else opt
        for opt in options
    ]
    question_payload["min_selections"] = q.get("min_selections", 1)
    question_payload["max_selections"] = q.get("max_selections")


def _resolve_depends_on_reference(
    depends_on_ref: str,
    question_id_map: dict,
    questions_payload: List[dict]
) -> str | None:
    """Resolve a depends_on reference to an actual question ID."""
    # Try to find by ID prefix or index
    for key, actual_id in question_id_map.items():
        if depends_on_ref == key or depends_on_ref in actual_id:
            return actual_id

    # Also check if it matches any question_id directly
    for qp in questions_payload:
        if qp["question_id"] == depends_on_ref or depends_on_ref in qp.get("question", ""):
            return qp["question_id"]

    return None


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
            - question_type: "single_choice", "multi_choice", or "resource_picker"
            - options: List of options with value, label, and optional is_wildcard (for choice types)
            - reasoning: Explain why you're asking this question
            - min_selections: Minimum number of selections (for multi-choice, default 1)
            - max_selections: Maximum number of selections (for multi-choice, optional)

            For resource_picker type, include:
            - provider: OAuth provider (google, github, discord, supabase_mgmt)
            - resource_type: Type of resource (google_spreadsheet, guild, channel, etc.)
            - display_field: Field to display (default: "name")
            - value_field: Field to use as value (default: "id")
            - search_enabled: Whether search is supported (default: True)
            - hierarchy: Whether folder navigation is supported (default: False)
            - depends_on: Question ID this depends on (for cascading pickers)
            - depends_on_field: Field name from the dependent resource

    Returns:
        JSON string containing list of answers, one per question in the same order.
        Each answer has: {"question_id": "...", "selected_values": [...], "custom_input": "..."}

    Example for choice questions:
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
            }
        ])

    Example for resource picker:
        ask_clarification_questions([
            {
                "question": "Which Google spreadsheet should we use?",
                "question_type": "resource_picker",
                "provider": "google",
                "resource_type": "google_spreadsheet",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "reasoning": "The google_sheets_read tool requires a spreadsheet_id"
            }
        ])

    Example for dependent resource pickers (Discord channel depends on guild):
        ask_clarification_questions([
            {
                "question": "Which Discord server?",
                "question_type": "resource_picker",
                "provider": "discord",
                "resource_type": "guild",
                "value_field": "resource_id",
                "reasoning": "Need to select server first"
            },
            {
                "question": "Which channel in that server?",
                "question_type": "resource_picker",
                "provider": "discord",
                "resource_type": "channel",
                "depends_on": "q_guild",  # References the guild question
                "depends_on_field": "guild_id",
                "reasoning": "Select channel from the chosen server"
            }
        ])
    """
    if not questions:
        raise ValueError("At least one question is required")

    if len(questions) > 10:
        raise ValueError("Maximum 10 questions allowed per batch")

    # Build questions with unique IDs
    questions_payload = []
    question_id_map = {}  # Map user-provided depends_on values to actual IDs

    for idx, q in enumerate(questions):
        question_id = f"q_{uuid.uuid4().hex[:8]}"
        question_type = q.get("question_type", "single_choice")

        # Track question IDs for depends_on resolution
        question_id_map[f"q_{idx}"] = question_id

        # Build base question payload
        question_payload = {
            "question_id": question_id,
            "question": q.get("question", ""),
            "question_type": question_type,
            "reasoning": q.get("reasoning", ""),
        }

        if question_type == QuestionType.RESOURCE_PICKER.value:
            _build_resource_picker_payload(q, question_payload)
        else:
            _build_choice_question_payload(q, question_payload)

        questions_payload.append(question_payload)

    # Resolve depends_on references
    for question_payload in questions_payload:
        if "_depends_on_ref" in question_payload:
            depends_on_ref = question_payload.pop("_depends_on_ref")
            question_payload["depends_on"] = _resolve_depends_on_reference(
                depends_on_ref, question_id_map, questions_payload
            )

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
