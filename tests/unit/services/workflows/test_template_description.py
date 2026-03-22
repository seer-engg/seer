"""Tests for generate_template_description — specifically JSON extraction from LLM responses."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.workflows.services.share_template import generate_template_description


def _mock_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    return resp


@pytest.fixture()
def _mock_db(monkeypatch):
    """Patch DB queries so generate_template_description doesn't hit the database."""
    mock_workflow = MagicMock()
    mock_workflow.name = "Test Workflow"

    mock_version = MagicMock()
    mock_version.spec = {"graph": {"nodes": [{"id": "n1", "type": "tool", "data": {"label": "Send Email"}}]}}

    with (
        patch("seer.api.workflows.services.share_template.Workflow") as wf_cls,
        patch("seer.api.workflows.services.share_template.WorkflowVersion") as wv_cls,
        patch("seer.api.workflows.services.share_template.parse_workflow_public_id", return_value=1),
    ):
        wf_cls.filter.return_value.first = AsyncMock(return_value=mock_workflow)
        wv_cls.filter.return_value.first = AsyncMock(side_effect=[mock_version, None])
        yield


VALID_JSON = '{"name": "Email Bot", "description": "Sends emails", "category": "communication", "tags": ["email"]}'
MARKDOWN_WRAPPED = f"```json\n{VALID_JSON}\n```"
MARKDOWN_GENERIC = f"```\n{VALID_JSON}\n```"
MARKDOWN_WITH_PREAMBLE = f"Here is the JSON:\n```json\n{VALID_JSON}\n```\nHope this helps!"


@pytest.mark.usefixtures("_mock_db")
class TestGenerateTemplateDescription:
    @patch("seer.llm.get_llm")
    async def test_raw_json_response(self, mock_get_llm):
        llm = AsyncMock()
        llm.ainvoke.return_value = _mock_llm_response(VALID_JSON)
        mock_get_llm.return_value = llm

        result = await generate_template_description(MagicMock(), "wf_1")
        assert result.name == "Email Bot"
        assert result.description == "Sends emails"

    @patch("seer.llm.get_llm")
    async def test_markdown_json_fence(self, mock_get_llm):
        """Reproduces PYTHON-FASTAPI-5V: LLM wraps JSON in ```json ... ``` fences."""
        llm = AsyncMock()
        llm.ainvoke.return_value = _mock_llm_response(MARKDOWN_WRAPPED)
        mock_get_llm.return_value = llm

        result = await generate_template_description(MagicMock(), "wf_1")
        assert result.name == "Email Bot"
        assert result.category == "communication"

    @patch("seer.llm.get_llm")
    async def test_markdown_generic_fence(self, mock_get_llm):
        llm = AsyncMock()
        llm.ainvoke.return_value = _mock_llm_response(MARKDOWN_GENERIC)
        mock_get_llm.return_value = llm

        result = await generate_template_description(MagicMock(), "wf_1")
        assert result.name == "Email Bot"

    @patch("seer.llm.get_llm")
    async def test_markdown_with_preamble(self, mock_get_llm):
        llm = AsyncMock()
        llm.ainvoke.return_value = _mock_llm_response(MARKDOWN_WITH_PREAMBLE)
        mock_get_llm.return_value = llm

        result = await generate_template_description(MagicMock(), "wf_1")
        assert result.name == "Email Bot"
        assert result.tags == ["email"]
