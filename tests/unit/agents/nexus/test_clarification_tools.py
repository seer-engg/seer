"""
Unit tests for clarification question tools.

Tests the ask_clarification_question tool and its schema validation.
"""
# pylint: disable=import-outside-toplevel  # Reason: Test-specific imports are acceptable
import json
import pytest
from unittest.mock import patch, MagicMock
from seer.agents.nexus.tools.clarification_tools import (
    ask_clarification_question,
    QuestionType,
    QuestionOption,
    AskClarificationInput,
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


class TestAskClarificationInput:
    """Test AskClarificationInput schema validation."""

    def test_valid_single_choice(self):
        """Test valid single-choice question input."""
        input_data = AskClarificationInput(
            question="Which email provider?",
            question_type=QuestionType.SINGLE_CHOICE,
            options=[
                QuestionOption(value="gmail", label="Gmail"),
                QuestionOption(value="outlook", label="Outlook"),
            ],
            reasoning="Need to configure email integration"
        )

        assert input_data.question == "Which email provider?"
        assert input_data.question_type == QuestionType.SINGLE_CHOICE
        assert len(input_data.options) == 2
        assert input_data.min_selections == 1
        assert input_data.max_selections is None

    def test_valid_multi_choice(self):
        """Test valid multi-choice question input."""
        input_data = AskClarificationInput(
            question="Which integrations?",
            question_type=QuestionType.MULTI_CHOICE,
            options=[
                QuestionOption(value="gmail", label="Gmail"),
                QuestionOption(value="slack", label="Slack"),
                QuestionOption(value="github", label="GitHub"),
            ],
            reasoning="User wants multiple integrations",
            min_selections=1,
            max_selections=3
        )

        assert input_data.question_type == QuestionType.MULTI_CHOICE
        assert input_data.min_selections == 1
        assert input_data.max_selections == 3


class TestAskClarificationQuestion:
    """Test ask_clarification_question tool."""

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_single_choice_question(self, mock_uuid, mock_interrupt):
        """Test asking a single-choice question."""
        # Mock UUID generation
        mock_uuid.return_value = MagicMock(hex="abcd1234")

        # Mock interrupt to return user's answer
        mock_interrupt.return_value = {
            "selected_values": ["gmail"],
            "custom_input": None
        }

        options = [
            QuestionOption(value="gmail", label="Gmail"),
            QuestionOption(value="outlook", label="Outlook"),
        ]

        result = ask_clarification_question.invoke({
            "question": "Which email provider?",
            "question_type": QuestionType.SINGLE_CHOICE,
            "options": options,
            "reasoning": "Need to configure email"
        })

        # Verify interrupt was called with correct payload
        mock_interrupt.assert_called_once()
        call_args = mock_interrupt.call_args[0][0]

        assert call_args["type"] == "clarification_question"
        assert call_args["question_id"] == "q_abcd1234"
        assert call_args["question"] == "Which email provider?"
        assert call_args["question_type"] == QuestionType.SINGLE_CHOICE
        assert len(call_args["options"]) == 2
        assert call_args["min_selections"] == 1
        assert call_args["max_selections"] is None
        assert call_args["reasoning"] == "Need to configure email"

        # Verify result
        result_data = json.loads(result)
        assert result_data["selected_values"] == ["gmail"]
        assert result_data["custom_input"] is None

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_wildcard_with_custom_input(self, mock_uuid, mock_interrupt):
        """Test question with wildcard option and custom input."""
        mock_uuid.return_value = MagicMock(hex="xyz789")

        # Mock user selecting wildcard with custom input
        mock_interrupt.return_value = {
            "selected_values": ["other"],
            "custom_input": "ProtonMail"
        }

        options = [
            QuestionOption(value="gmail", label="Gmail"),
            QuestionOption(value="outlook", label="Outlook"),
            QuestionOption(value="other", label="Other", is_wildcard=True),
        ]

        result = ask_clarification_question.invoke({
            "question": "Which email provider?",
            "question_type": QuestionType.SINGLE_CHOICE,
            "options": options,
            "reasoning": "Need to configure email"
        })

        # Verify wildcard option included
        call_args = mock_interrupt.call_args[0][0]
        wildcard_opt = [opt for opt in call_args["options"] if opt.get("is_wildcard")]
        assert len(wildcard_opt) == 1
        assert wildcard_opt[0]["value"] == "other"

        # Verify result includes custom input
        result_data = json.loads(result)
        assert result_data["selected_values"] == ["other"]
        assert result_data["custom_input"] == "ProtonMail"

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_multi_choice_question(self, mock_uuid, mock_interrupt):
        """Test multi-choice question."""
        mock_uuid.return_value = MagicMock(hex="multi123")

        # Mock user selecting multiple options
        mock_interrupt.return_value = {
            "selected_values": ["gmail", "slack"],
            "custom_input": None
        }

        options = [
            QuestionOption(value="gmail", label="Gmail"),
            QuestionOption(value="slack", label="Slack"),
            QuestionOption(value="github", label="GitHub"),
        ]

        result = ask_clarification_question.invoke({
            "question": "Which integrations?",
            "question_type": QuestionType.MULTI_CHOICE,
            "options": options,
            "reasoning": "User wants multiple integrations",
            "min_selections": 1,
            "max_selections": 3
        })

        # Verify multi-choice configuration
        call_args = mock_interrupt.call_args[0][0]
        assert call_args["question_type"] == QuestionType.MULTI_CHOICE
        assert call_args["min_selections"] == 1
        assert call_args["max_selections"] == 3

        # Verify multiple selections in result
        result_data = json.loads(result)
        assert len(result_data["selected_values"]) == 2
        assert "gmail" in result_data["selected_values"]
        assert "slack" in result_data["selected_values"]

    @patch('seer.agents.nexus.tools.clarification_tools.interrupt')
    @patch('seer.agents.nexus.tools.clarification_tools.uuid.uuid4')
    def test_question_id_generation(self, mock_uuid, mock_interrupt):
        """Test that question IDs are unique."""
        mock_interrupt.return_value = {"selected_values": ["test"], "custom_input": None}

        # Test multiple calls generate different IDs
        ids = []
        for i in range(3):
            mock_uuid.return_value = MagicMock(hex=f"test{i}")
            options = [QuestionOption(value="test", label="Test")]

            ask_clarification_question.invoke({
                "question": "Test?",
                "question_type": QuestionType.SINGLE_CHOICE,
                "options": options,
                "reasoning": "Testing"
            })

            call_args = mock_interrupt.call_args[0][0]
            ids.append(call_args["question_id"])

        # All IDs should be different
        assert len(set(ids)) == 3
        assert all(qid.startswith("q_") for qid in ids)
