"""
Unit tests for clarification question tools.

Tests the ask_clarification_questions tool for batch questions.
"""
# pylint: disable=import-outside-toplevel  # Reason: Test-specific imports are acceptable
import json
import pytest
from unittest.mock import patch, MagicMock
from seer.agents.nexus.tools.clarification_tools import (
    ask_clarification_questions,
    QuestionType,
    QuestionOption,
)


class TestQuestionOption:
    """Test QuestionOption model validation."""

    def test_valid_option(self):
        """Test valid QuestionOption creation."""
        option = QuestionOption(
            value="gmail",
            label="Gmail",
            is_wildcard=False
        )

        assert option.value == "gmail"
        assert option.label == "Gmail"
        assert option.is_wildcard is False

    def test_wildcard_option(self):
        """Test wildcard option creation."""
        option = QuestionOption(
            value="other",
            label="Other (specify)",
            is_wildcard=True
        )

        assert option.value == "other"
        assert option.label == "Other (specify)"
        assert option.is_wildcard is True

    def test_default_is_wildcard(self):
        """Test that is_wildcard defaults to False."""
        option = QuestionOption(
            value="test",
            label="Test"
        )

        assert option.is_wildcard is False


class TestAskClarificationQuestions:
    """Test ask_clarification_questions batch tool."""

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_single_question_in_batch(self, mock_uuid, mock_interrupt):
        """Test asking a single question using batch tool."""
        mock_uuid.return_value = MagicMock(hex="abcd1234")

        mock_interrupt.return_value = [
            {"question_id": "q_abcd1234", "selected_values": ["gmail"], "custom_input": None},
        ]

        questions = [
            {
                "question": "Which email provider?",
                "question_type": "single_choice",
                "options": [
                    {"value": "gmail", "label": "Gmail"},
                    {"value": "outlook", "label": "Outlook"},
                ],
                "reasoning": "Need to configure email"
            }
        ]

        result = ask_clarification_questions.invoke({"questions": questions})

        # Verify interrupt was called with correct payload
        mock_interrupt.assert_called_once()
        call_args = mock_interrupt.call_args[0][0]

        assert call_args["type"] == "clarification_questions"
        assert len(call_args["questions"]) == 1

        q1 = call_args["questions"][0]
        assert q1["question_id"] == "q_abcd1234"
        assert q1["question"] == "Which email provider?"
        assert q1["question_type"] == "single_choice"

        result_data = json.loads(result)
        assert len(result_data) == 1
        assert result_data[0]["selected_values"] == ["gmail"]

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_multiple_questions(self, mock_uuid, mock_interrupt):
        """Test asking multiple questions at once."""
        mock_uuid.side_effect = [
            MagicMock(hex="aabb1122"),
            MagicMock(hex="ccdd3344"),
        ]

        mock_interrupt.return_value = [
            {"question_id": "q_aabb1122", "selected_values": ["gmail"], "custom_input": None},
            {"question_id": "q_ccdd3344", "selected_values": ["slack", "email"], "custom_input": None},
        ]

        questions = [
            {
                "question": "Which email provider?",
                "question_type": "single_choice",
                "options": [
                    {"value": "gmail", "label": "Gmail"},
                    {"value": "outlook", "label": "Outlook"},
                ],
                "reasoning": "Need to configure email"
            },
            {
                "question": "Which notification channels?",
                "question_type": "multi_choice",
                "options": [
                    {"value": "slack", "label": "Slack"},
                    {"value": "email", "label": "Email"},
                    {"value": "sms", "label": "SMS"},
                ],
                "reasoning": "Need to know where to send notifications",
                "min_selections": 1
            }
        ]

        result = ask_clarification_questions.invoke({"questions": questions})

        mock_interrupt.assert_called_once()
        call_args = mock_interrupt.call_args[0][0]

        assert call_args["type"] == "clarification_questions"
        assert len(call_args["questions"]) == 2

        q1 = call_args["questions"][0]
        assert q1["question_id"] == "q_aabb1122"
        assert q1["question"] == "Which email provider?"
        assert q1["question_type"] == "single_choice"
        assert len(q1["options"]) == 2

        q2 = call_args["questions"][1]
        assert q2["question_id"] == "q_ccdd3344"
        assert q2["question"] == "Which notification channels?"
        assert q2["question_type"] == "multi_choice"
        assert q2["min_selections"] == 1

        result_data = json.loads(result)
        assert len(result_data) == 2
        assert result_data[0]["selected_values"] == ["gmail"]
        assert result_data[1]["selected_values"] == ["slack", "email"]

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_wildcard_option(self, mock_uuid, mock_interrupt):
        """Test batch questions with wildcard option."""
        mock_uuid.side_effect = [MagicMock(hex="wild0001")]

        mock_interrupt.return_value = [
            {"question_id": "q_wild0001", "selected_values": ["other"], "custom_input": "ProtonMail"},
        ]

        questions = [
            {
                "question": "Which email provider?",
                "question_type": "single_choice",
                "options": [
                    {"value": "gmail", "label": "Gmail"},
                    {"value": "other", "label": "Other", "is_wildcard": True},
                ],
                "reasoning": "Need email config"
            }
        ]

        result = ask_clarification_questions.invoke({"questions": questions})

        call_args = mock_interrupt.call_args[0][0]
        wildcard_opts = [
            opt for opt in call_args["questions"][0]["options"]
            if opt.get("is_wildcard")
        ]
        assert len(wildcard_opts) == 1
        assert wildcard_opts[0]["value"] == "other"

        result_data = json.loads(result)
        assert result_data[0]["custom_input"] == "ProtonMail"

    def test_empty_questions_raises_error(self):
        """Test that empty questions list raises an error."""
        with pytest.raises(ValueError, match="At least one question is required"):
            ask_clarification_questions.invoke({"questions": []})

    def test_max_questions_limit(self):
        """Test that more than 10 questions raises an error."""
        questions = [
            {
                "question": f"Question {i}?",
                "question_type": "single_choice",
                "options": [{"value": "a", "label": "A"}],
                "reasoning": f"Reason {i}"
            }
            for i in range(11)
        ]

        with pytest.raises(ValueError, match="Maximum 10 questions allowed"):
            ask_clarification_questions.invoke({"questions": questions})

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_unique_question_ids_in_batch(self, mock_uuid, mock_interrupt):
        """Test that each question in batch gets unique ID."""
        mock_uuid.side_effect = [
            MagicMock(hex="uniq0001"),
            MagicMock(hex="uniq0002"),
            MagicMock(hex="uniq0003"),
        ]

        mock_interrupt.return_value = [
            {"question_id": "q_uniq0001", "selected_values": ["a"], "custom_input": None},
            {"question_id": "q_uniq0002", "selected_values": ["b"], "custom_input": None},
            {"question_id": "q_uniq0003", "selected_values": ["c"], "custom_input": None},
        ]

        questions = [
            {
                "question": f"Question {i}?",
                "question_type": "single_choice",
                "options": [{"value": chr(97 + i), "label": chr(65 + i)}],
                "reasoning": f"Reason {i}"
            }
            for i in range(3)
        ]

        ask_clarification_questions.invoke({"questions": questions})

        call_args = mock_interrupt.call_args[0][0]
        question_ids = [q["question_id"] for q in call_args["questions"]]

        assert len(set(question_ids)) == 3
        assert question_ids == ["q_uniq0001", "q_uniq0002", "q_uniq0003"]

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_default_values(self, mock_uuid, mock_interrupt):
        """Test default values for optional fields."""
        mock_uuid.return_value = MagicMock(hex="def00001")

        mock_interrupt.return_value = [
            {"question_id": "q_def00001", "selected_values": ["a"], "custom_input": None},
        ]

        questions = [
            {
                "question": "Test?",
                "question_type": "single_choice",
                "options": [{"value": "a", "label": "A"}],
                "reasoning": "Testing"
                # Note: min_selections and max_selections not provided
            }
        ]

        ask_clarification_questions.invoke({"questions": questions})

        call_args = mock_interrupt.call_args[0][0]
        q = call_args["questions"][0]

        # Check defaults
        assert q["min_selections"] == 1
        assert q["max_selections"] is None
