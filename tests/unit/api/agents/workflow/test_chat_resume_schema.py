"""Unit tests for chat resume request validation."""

import pytest
from pydantic import ValidationError

from seer.api.agents.workflow.chat_schema import ChatResumeRequest


@pytest.mark.unit
class TestChatResumeRequest:
    """Validate mutually exclusive resume payload shapes."""

    def test_accepts_free_text_resume_message(self):
        request = ChatResumeRequest(
            thread_id="thread-123",
            message=" Use Gmail for invoices only. ",
        )

        assert request.message == "Use Gmail for invoices only."
        assert request.answers is None
        assert request.command is None

    def test_rejects_multiple_resume_payload_types(self):
        with pytest.raises(ValidationError, match="Exactly one of answers, message, or command must be provided"):
            ChatResumeRequest(
                thread_id="thread-123",
                message="Use Gmail",
                command={"resume": "ignored"},
            )
