"""
Unit tests for resource picker schema models in chat_schema.py.

Tests the ClarificationQuestion model's resource picker fields.
"""
import pytest
from seer.api.agents.workflow.chat_schema import (
    ClarificationQuestion,
    ClarificationQuestionOption,
    ClarificationQuestions,
    QuestionType,
)


@pytest.mark.unit
class TestClarificationQuestionResourcePicker:
    """Test ClarificationQuestion model with resource picker type."""

    def test_resource_picker_question_type_exists(self):
        """Test that RESOURCE_PICKER is a valid QuestionType."""
        assert QuestionType.RESOURCE_PICKER.value == "resource_picker"

    def test_resource_picker_question_basic(self):
        """Test creating a basic resource picker question."""
        question = ClarificationQuestion(
            question_id="q_test123",
            question="Which Google spreadsheet should we use?",
            question_type=QuestionType.RESOURCE_PICKER,
            provider="google",
            resource_type="google_spreadsheet",
            reasoning="The google_sheets_read tool requires a spreadsheet_id"
        )

        assert question.question_id == "q_test123"
        assert question.question_type == QuestionType.RESOURCE_PICKER
        assert question.provider == "google"
        assert question.resource_type == "google_spreadsheet"
        # Check defaults
        assert question.display_field == "name"
        assert question.value_field == "id"
        assert question.search_enabled is True
        assert question.hierarchy is False
        assert question.depends_on is None
        assert question.depends_on_field is None

    def test_resource_picker_question_all_fields(self):
        """Test resource picker question with all fields specified."""
        question = ClarificationQuestion(
            question_id="q_channel01",
            question="Which channel in that server?",
            question_type=QuestionType.RESOURCE_PICKER,
            provider="discord",
            resource_type="channel",
            display_field="channel_name",
            value_field="channel_id",
            search_enabled=False,
            hierarchy=True,
            depends_on="q_guild01",
            depends_on_field="guild_id",
            reasoning="Select channel from the chosen server"
        )

        assert question.provider == "discord"
        assert question.resource_type == "channel"
        assert question.display_field == "channel_name"
        assert question.value_field == "channel_id"
        assert question.search_enabled is False
        assert question.hierarchy is True
        assert question.depends_on == "q_guild01"
        assert question.depends_on_field == "guild_id"

    def test_resource_picker_question_serialization(self):
        """Test that resource picker question serializes correctly."""
        question = ClarificationQuestion(
            question_id="q_test456",
            question="Which spreadsheet?",
            question_type=QuestionType.RESOURCE_PICKER,
            provider="google",
            resource_type="google_spreadsheet",
            reasoning="Need spreadsheet"
        )

        data = question.model_dump()

        assert data["question_id"] == "q_test456"
        assert data["question_type"] == "resource_picker"
        assert data["provider"] == "google"
        assert data["resource_type"] == "google_spreadsheet"
        assert data["display_field"] == "name"
        assert data["value_field"] == "id"
        assert data["search_enabled"] is True
        assert data["hierarchy"] is False

    def test_single_choice_question_still_works(self):
        """Test that single_choice questions still work correctly."""
        question = ClarificationQuestion(
            question_id="q_choice01",
            question="Which email provider?",
            question_type=QuestionType.SINGLE_CHOICE,
            options=[
                ClarificationQuestionOption(value="gmail", label="Gmail"),
                ClarificationQuestionOption(value="outlook", label="Outlook"),
            ],
            reasoning="Need email config"
        )

        assert question.question_type == QuestionType.SINGLE_CHOICE
        assert len(question.options) == 2
        assert question.min_selections == 1
        # Resource picker fields should be None/defaults for non-picker questions
        assert question.provider is None
        assert question.resource_type is None

    def test_clarification_questions_batch_with_resource_picker(self):
        """Test ClarificationQuestions with mixed question types."""
        questions = ClarificationQuestions(
            questions=[
                ClarificationQuestion(
                    question_id="q_choice01",
                    question="Which email provider?",
                    question_type=QuestionType.SINGLE_CHOICE,
                    options=[
                        ClarificationQuestionOption(value="gmail", label="Gmail"),
                    ],
                    reasoning="Need email"
                ),
                ClarificationQuestion(
                    question_id="q_picker01",
                    question="Which spreadsheet?",
                    question_type=QuestionType.RESOURCE_PICKER,
                    provider="google",
                    resource_type="google_spreadsheet",
                    reasoning="Need spreadsheet"
                ),
            ]
        )

        assert len(questions.questions) == 2
        assert questions.questions[0].question_type == QuestionType.SINGLE_CHOICE
        assert questions.questions[1].question_type == QuestionType.RESOURCE_PICKER

    def test_dependent_resource_pickers_in_batch(self):
        """Test dependent resource pickers serialize correctly."""
        questions = ClarificationQuestions(
            questions=[
                ClarificationQuestion(
                    question_id="q_guild01",
                    question="Which Discord server?",
                    question_type=QuestionType.RESOURCE_PICKER,
                    provider="discord",
                    resource_type="guild",
                    value_field="resource_id",
                    reasoning="Need to select server first"
                ),
                ClarificationQuestion(
                    question_id="q_channel01",
                    question="Which channel?",
                    question_type=QuestionType.RESOURCE_PICKER,
                    provider="discord",
                    resource_type="channel",
                    depends_on="q_guild01",
                    depends_on_field="guild_id",
                    reasoning="Select channel"
                ),
            ]
        )

        data = questions.model_dump()

        assert len(data["questions"]) == 2
        assert data["questions"][0]["depends_on"] is None
        assert data["questions"][1]["depends_on"] == "q_guild01"
        assert data["questions"][1]["depends_on_field"] == "guild_id"
