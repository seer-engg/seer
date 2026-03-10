"""
Unit tests for AgentNode artifact generation.

Covers:
- HTML → PDF / DOCX converter sanity checks (bytes output)
- create_artifact tool factory (mock file system, verify store_file_with_record called)
- _collect_artifacts_from_messages (ToolMessage parsing)
- execute_async with artifacts disabled (no __artifacts key)
- execute_async with artifacts enabled (mock agent returning ToolMessages)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from seer.core.files.models import WORKFLOW_FILE_REF_TYPE, WorkflowFileRef
from seer.core.nodes.agent_node import _collect_artifacts_from_messages, _extract_agent_config
from seer.core.nodes.artifacts.tool import ARTIFACT_TOOL_NAME


# =============================================================================
# Fixtures
# =============================================================================


def _make_file_ref_dict(filename: str = "report.pdf", fmt: str = "application/pdf") -> Dict[str, Any]:
    """Build a minimal WorkflowFileRef dict for test assertions."""
    return {
        "_type": WORKFLOW_FILE_REF_TYPE,
        "file_id": "test-file-id-1",
        "storage_path": f"s3://bucket/run_1/{filename}",
        "filename": filename,
        "mime_type": fmt,
        "size_bytes": 1024,
        "workflow_run_id": "run_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "md5_hash": None,
    }


def _make_artifact_tool_message(file_ref_dict: Dict[str, Any]) -> ToolMessage:
    """Create a ToolMessage as if returned by the create_artifact tool."""
    return ToolMessage(
        content=json.dumps(file_ref_dict, default=str),
        tool_call_id="call_abc",
        name=ARTIFACT_TOOL_NAME,
    )


# =============================================================================
# Converter sanity tests (import-guarded)
# =============================================================================


@pytest.mark.unit
class TestConverters:
    """Tests for content → bytes converters."""

    def test_markdown_to_html_produces_styled_document(self):
        """markdown_to_html should return a full HTML document with converted content."""
        from seer.core.nodes.artifacts.converters import markdown_to_html

        result = markdown_to_html("# Hello\n\nA paragraph with **bold** text.\n\n| A | B |\n|---|---|\n| 1 | 2 |")
        assert "<!DOCTYPE html>" in result
        assert "<h1" in result
        assert "<strong>bold</strong>" in result
        assert "<table>" in result

    def test_markdown_to_html_fenced_code(self):
        """markdown_to_html should handle fenced code blocks."""
        from seer.core.nodes.artifacts.converters import markdown_to_html

        result = markdown_to_html("```python\nprint('hi')\n```")
        assert "<code" in result

    def test_html_to_pdf_returns_bytes(self):
        """html_to_pdf should return non-empty bytes starting with %PDF."""
        pytest.importorskip("weasyprint", reason="weasyprint not installed")
        from seer.core.nodes.artifacts.converters import html_to_pdf

        result = html_to_pdf("<html><body><h1>Hello</h1></body></html>")
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:4] == b"%PDF"

    def test_html_to_docx_returns_bytes(self):
        """html_to_docx should return non-empty bytes (valid zip/DOCX magic bytes)."""
        pytest.importorskip("htmldocx", reason="htmldocx not installed")
        from seer.core.nodes.artifacts.converters import html_to_docx

        result = html_to_docx("<html><body><p>Hello world</p></body></html>")
        assert isinstance(result, bytes)
        assert len(result) > 0
        # DOCX files are ZIP archives; magic bytes are PK (0x50 0x4B)
        assert result[:2] == b"PK"

    def test_format_mime_constants(self):
        """FORMAT_MIME should contain pdf and docx entries."""
        from seer.core.nodes.artifacts.converters import FORMAT_MIME

        assert "pdf" in FORMAT_MIME
        assert "docx" in FORMAT_MIME
        assert FORMAT_MIME["pdf"] == "application/pdf"
        assert "word" in FORMAT_MIME["docx"] or "openxmlformats" in FORMAT_MIME["docx"]


# =============================================================================
# create_artifact tool factory tests
# =============================================================================


@pytest.mark.unit
class TestCreateArtifactTool:
    """Tests for make_create_artifact_tool factory."""

    def _make_ctx(self, has_file_system: bool = True) -> MagicMock:
        """Build a minimal NodeExecutionContext mock."""
        ctx = MagicMock()
        ctx.runtime_context.has_file_system = has_file_system
        ctx.runtime_context.workflow_run_id = "run_42"
        ctx.runtime_context.user = MagicMock()
        ctx.runtime_context.user.user_id = "user_1"
        return ctx

    @pytest.mark.asyncio
    async def test_store_file_with_record_called_for_pdf(self):
        """Tool should call store_file_with_record with correct args for PDF."""
        from seer.core.nodes.artifacts.tool import make_create_artifact_tool

        ctx = self._make_ctx()
        file_ref = WorkflowFileRef(
            file_id="fid",
            storage_path="s3://b/r/report.pdf",
            filename="report.pdf",
            mime_type="application/pdf",
            size_bytes=500,
            workflow_run_id="run_42",
            created_at=datetime.now(timezone.utc),
        )
        ctx.runtime_context.file_system.store_file_with_record = AsyncMock(return_value=file_ref)

        with patch("seer.core.nodes.artifacts.tool.html_to_pdf", return_value=b"%PDF-fake"):
            tool = make_create_artifact_tool(ctx, node_id="agent1")
            result = await tool.coroutine(
                html_content="<p>test</p>",
                filename="report.pdf",
                format="pdf",
            )

        ctx.runtime_context.file_system.store_file_with_record.assert_called_once()
        call_kwargs = ctx.runtime_context.file_system.store_file_with_record.call_args.kwargs
        assert call_kwargs["filename"] == "report.pdf"
        assert call_kwargs["mime_type"] == "application/pdf"
        assert call_kwargs["source_tool"] == ARTIFACT_TOOL_NAME
        assert call_kwargs["source_node_id"] == "agent1"

        # Result should be JSON-decodable WorkflowFileRef dict
        parsed = json.loads(result)
        assert parsed["_type"] == WORKFLOW_FILE_REF_TYPE

    @pytest.mark.asyncio
    async def test_store_file_with_record_called_for_docx(self):
        """Tool should call store_file_with_record with DOCX mime type."""
        from seer.core.nodes.artifacts.tool import make_create_artifact_tool

        ctx = self._make_ctx()
        file_ref = WorkflowFileRef(
            file_id="fid2",
            storage_path="s3://b/r/summary.docx",
            filename="summary.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            size_bytes=800,
            workflow_run_id="run_42",
            created_at=datetime.now(timezone.utc),
        )
        ctx.runtime_context.file_system.store_file_with_record = AsyncMock(return_value=file_ref)

        with patch("seer.core.nodes.artifacts.tool.html_to_docx", return_value=b"PKfake"):
            tool = make_create_artifact_tool(ctx, node_id="agent1")
            result = await tool.coroutine(
                html_content="<p>test</p>",
                filename="summary.docx",
                format="docx",
            )

        call_kwargs = ctx.runtime_context.file_system.store_file_with_record.call_args.kwargs
        assert "openxmlformats" in call_kwargs["mime_type"]

    @pytest.mark.asyncio
    async def test_raises_when_no_file_system(self):
        """Tool should raise ExecutionError when file system is not configured."""
        from seer.core.errors import ExecutionError
        from seer.core.nodes.artifacts.tool import make_create_artifact_tool

        ctx = self._make_ctx(has_file_system=False)

        with patch("seer.core.nodes.artifacts.tool.html_to_pdf", return_value=b"%PDF-fake"):
            tool = make_create_artifact_tool(ctx, node_id="agent1")
            with pytest.raises(ExecutionError, match="Artifact storage not configured"):
                await tool.coroutine(
                    html_content="<p>test</p>",
                    filename="report.pdf",
                    format="pdf",
                )

    @pytest.mark.asyncio
    async def test_markdown_content_type_calls_markdown_to_html(self):
        """Tool should convert markdown to HTML before PDF conversion when content_type='markdown'."""
        from seer.core.nodes.artifacts.tool import make_create_artifact_tool

        ctx = self._make_ctx()
        file_ref = WorkflowFileRef(
            file_id="fid3",
            storage_path="s3://b/r/report.pdf",
            filename="report.pdf",
            mime_type="application/pdf",
            size_bytes=500,
            workflow_run_id="run_42",
            created_at=datetime.now(timezone.utc),
        )
        ctx.runtime_context.file_system.store_file_with_record = AsyncMock(return_value=file_ref)

        with (
            patch("seer.core.nodes.artifacts.tool.markdown_to_html", return_value="<h1>Hello</h1>") as mock_md,
            patch("seer.core.nodes.artifacts.tool.html_to_pdf", return_value=b"%PDF-fake"),
        ):
            tool = make_create_artifact_tool(ctx, node_id="agent1")
            await tool.coroutine(
                html_content="# Hello",
                filename="report.pdf",
                format="pdf",
                content_type="markdown",
            )

        mock_md.assert_called_once_with("# Hello")

    @pytest.mark.asyncio
    async def test_html_content_type_skips_markdown_conversion(self):
        """Tool should NOT call markdown_to_html when content_type='html'."""
        from seer.core.nodes.artifacts.tool import make_create_artifact_tool

        ctx = self._make_ctx()
        file_ref = WorkflowFileRef(
            file_id="fid4",
            storage_path="s3://b/r/report.pdf",
            filename="report.pdf",
            mime_type="application/pdf",
            size_bytes=500,
            workflow_run_id="run_42",
            created_at=datetime.now(timezone.utc),
        )
        ctx.runtime_context.file_system.store_file_with_record = AsyncMock(return_value=file_ref)

        with (
            patch("seer.core.nodes.artifacts.tool.markdown_to_html") as mock_md,
            patch("seer.core.nodes.artifacts.tool.html_to_pdf", return_value=b"%PDF-fake"),
        ):
            tool = make_create_artifact_tool(ctx, node_id="agent1")
            await tool.coroutine(
                html_content="<p>test</p>",
                filename="report.pdf",
                format="pdf",
            )

        mock_md.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_on_unsupported_format(self):
        """Tool should raise ExecutionError for unknown format."""
        from seer.core.errors import ExecutionError
        from seer.core.nodes.artifacts.tool import make_create_artifact_tool

        ctx = self._make_ctx()
        tool = make_create_artifact_tool(ctx, node_id="agent1")
        with pytest.raises(ExecutionError, match="unsupported format"):
            await tool.coroutine(
                html_content="<p>test</p>",
                filename="report.txt",
                format="txt",
            )


# =============================================================================
# _collect_artifacts_from_messages tests
# =============================================================================


@pytest.mark.unit
class TestCollectArtifactsFromMessages:
    """Tests for _collect_artifacts_from_messages helper."""

    def test_collects_valid_artifact_messages(self):
        """Should return file ref dicts from create_artifact ToolMessages."""
        ref_dict = _make_file_ref_dict("report.pdf")
        messages: List[Any] = [
            AIMessage(content="I will create the report"),
            _make_artifact_tool_message(ref_dict),
            AIMessage(content="Done."),
        ]
        artifacts = _collect_artifacts_from_messages(messages)
        assert len(artifacts) == 1
        assert artifacts[0]["_type"] == WORKFLOW_FILE_REF_TYPE
        assert artifacts[0]["filename"] == "report.pdf"

    def test_ignores_non_artifact_tool_messages(self):
        """Should ignore ToolMessages from other tools."""
        other_msg = ToolMessage(content='{"result": "ok"}', tool_call_id="c1", name="some_other_tool")
        messages: List[Any] = [other_msg]
        artifacts = _collect_artifacts_from_messages(messages)
        assert artifacts == []

    def test_ignores_non_file_ref_content(self):
        """Should skip ToolMessages where content is not a valid file ref."""
        bad_msg = ToolMessage(
            content='{"foo": "bar"}',  # not a file ref
            tool_call_id="c2",
            name=ARTIFACT_TOOL_NAME,
        )
        messages: List[Any] = [bad_msg]
        artifacts = _collect_artifacts_from_messages(messages)
        assert artifacts == []

    def test_handles_invalid_json(self):
        """Should log warning and skip ToolMessages with invalid JSON."""
        bad_msg = ToolMessage(
            content="not json at all",
            tool_call_id="c3",
            name=ARTIFACT_TOOL_NAME,
        )
        messages: List[Any] = [bad_msg]
        artifacts = _collect_artifacts_from_messages(messages)
        assert artifacts == []

    def test_collects_multiple_artifacts(self):
        """Should collect multiple file refs across multiple ToolMessages."""
        ref1 = _make_file_ref_dict("report.pdf")
        ref2 = _make_file_ref_dict("summary.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        ref2["file_id"] = "test-file-id-2"
        messages: List[Any] = [
            _make_artifact_tool_message(ref1),
            AIMessage(content="Generated first file"),
            _make_artifact_tool_message(ref2),
        ]
        artifacts = _collect_artifacts_from_messages(messages)
        assert len(artifacts) == 2

    def test_empty_messages_returns_empty(self):
        """Empty message list should produce no artifacts."""
        assert _collect_artifacts_from_messages([]) == []


# =============================================================================
# _extract_agent_config tests (enable_artifacts flag)
# =============================================================================


@pytest.mark.unit
class TestExtractAgentConfigArtifacts:
    """Tests for enable_artifacts in _extract_agent_config."""

    def _make_node(self, extra_inputs: Dict[str, Any] = None) -> MagicMock:
        node = MagicMock()
        node.id = "agent_1"
        node.inputs = {"model": "claude-sonnet-4-6", "prompt": "Do something", **(extra_inputs or {})}
        return node

    def test_enable_artifacts_defaults_to_true(self):
        """enable_artifacts should default to True when not set."""
        node = self._make_node()
        config = _extract_agent_config(node)
        assert config["enable_artifacts"] is True

    def test_enable_artifacts_true_when_set(self):
        """enable_artifacts should be True when set in inputs."""
        node = self._make_node({"enable_artifacts": True})
        config = _extract_agent_config(node)
        assert config["enable_artifacts"] is True

    def test_enable_artifacts_coerced_from_truthy_value(self):
        """enable_artifacts should be coerced via bool()."""
        node = self._make_node({"enable_artifacts": 1})
        config = _extract_agent_config(node)
        assert config["enable_artifacts"] is True


# =============================================================================
# execute_async integration-style tests
# =============================================================================


@pytest.mark.unit
class TestExecuteAsyncArtifacts:
    """Tests for artifact output in AgentNodeType.execute_async.

    Since render_template and evaluate_value are imported lazily inside
    execute_async, they must be patched at their source module paths
    (seer.core.expr.evaluator.*) rather than at agent_node level.
    """

    def _make_services(self) -> MagicMock:
        services = MagicMock()
        services.model_registry.get.return_value.get_chat_model.return_value = MagicMock()
        services.type_env.get.return_value = None
        return services

    def _make_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.state = {}
        ctx.config = {}
        ctx.locals_ctx = {}
        ctx.trigger = None
        ctx.loop_body_map = None
        ctx.nested_loop_parents = None
        ctx.runtime_context.has_file_system = True
        ctx.runtime_context.workflow_run_id = "run_1"
        ctx.runtime_context.user = MagicMock()
        ctx.runtime_context.user.user_id = "user_1"
        return ctx

    def _make_node(self, enable_artifacts: bool = False) -> MagicMock:
        node = MagicMock()
        node.id = "agent_1"
        node.inputs = {
            "model": "claude-sonnet-4-6",
            "prompt": "Write a report",
            "tools": [],
            "enable_artifacts": enable_artifacts,
        }
        # MagicMock != OutputMode.json is truthy by default, so _handle_json_output returns text
        return node

    def _base_patches(self, mock_agent: AsyncMock) -> List[Any]:
        """Return patch context managers needed for all execute_async tests."""
        return [
            patch("seer.core.nodes.agent_node._bind_tools_for_agent", new=AsyncMock(return_value=[])),
            patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent),
            patch("seer.core.nodes.agent_node._resolve_llm_file_inputs", new=AsyncMock(return_value=({}, []))),
            patch("seer.core.nodes.agent_node.AgentNodeType._check_credit_limit", new=AsyncMock()),
            patch("seer.core.nodes.agent_node.AgentNodeType._track_usage_async"),
            # render_template and evaluate_value are locally imported in execute_async,
            # so they must be patched at the source module.
            patch("seer.core.expr.evaluator.render_template", return_value="Write a report"),
            patch("seer.core.expr.evaluator.evaluate_value", side_effect=lambda ctx, v: v),
        ]

    @pytest.mark.asyncio
    async def test_no_artifacts_key_when_disabled(self):
        """__artifacts key should be absent when enable_artifacts is False."""
        from seer.core.nodes.agent_node import AgentNodeType

        node = self._make_node(enable_artifacts=False)
        ctx = self._make_ctx()
        services = self._make_services()

        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Here is the report.")]})

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self._base_patches(mock_agent):
                stack.enter_context(p)
            node_type = AgentNodeType()
            output = await node_type.execute_async(node, ctx, services)

        assert "agent_1__artifacts" not in output
        assert "agent_1" in output

    @pytest.mark.asyncio
    async def test_artifacts_key_present_when_enabled(self):
        """__artifacts key should be present (even if empty) when enable_artifacts is True."""
        from seer.core.nodes.agent_node import AgentNodeType

        node = self._make_node(enable_artifacts=True)
        ctx = self._make_ctx()
        services = self._make_services()

        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Report done.")]})

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self._base_patches(mock_agent):
                stack.enter_context(p)
            stack.enter_context(
                # make_create_artifact_tool is locally imported in execute_async,
                # so patch at the source module.
                patch("seer.core.nodes.artifacts.tool.make_create_artifact_tool", return_value=MagicMock())
            )
            node_type = AgentNodeType()
            output = await node_type.execute_async(node, ctx, services)

        assert "agent_1__artifacts" in output
        assert isinstance(output["agent_1__artifacts"], list)

    @pytest.mark.asyncio
    async def test_artifacts_collected_from_tool_messages(self):
        """Artifacts in ToolMessages should appear in __artifacts output."""
        from seer.core.nodes.agent_node import AgentNodeType

        node = self._make_node(enable_artifacts=True)
        ctx = self._make_ctx()
        services = self._make_services()

        ref_dict = _make_file_ref_dict("report.pdf")
        artifact_msg = _make_artifact_tool_message(ref_dict)
        agent_result = {"messages": [AIMessage(content="Here is your report."), artifact_msg]}

        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value=agent_result)

        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self._base_patches(mock_agent):
                stack.enter_context(p)
            stack.enter_context(
                patch("seer.core.nodes.artifacts.tool.make_create_artifact_tool", return_value=MagicMock())
            )
            node_type = AgentNodeType()
            output = await node_type.execute_async(node, ctx, services)

        artifacts = output.get("agent_1__artifacts", [])
        assert len(artifacts) == 1
        assert artifacts[0]["filename"] == "report.pdf"
        assert artifacts[0]["_type"] == WORKFLOW_FILE_REF_TYPE


# =============================================================================
# _build_agent_trace artifacts field tests
# =============================================================================


@pytest.mark.unit
class TestBuildAgentTraceArtifacts:
    """Tests that _build_agent_trace stores full artifact list, not a count."""

    def test_artifacts_stored_as_list_not_count(self):
        """Trace should contain the full artifact list, not len(artifacts)."""
        from seer.core.nodes.agent_node import _build_agent_trace

        ref_dict = _make_file_ref_dict("report.pdf")
        trace = _build_agent_trace(
            node_id="agent_1",
            inputs={"prompt": "Write a report"},
            status="succeeded",
            success_data={
                "prompt": "Write a report",
                "tool_names": [],
                "steps": [],
                "result_value": "Done",
                "artifacts": [ref_dict],
            },
        )

        assert isinstance(trace["artifacts"], list), "artifacts should be a list, not a count"
        assert len(trace["artifacts"]) == 1
        assert trace["artifacts"][0]["filename"] == "report.pdf"
        assert trace["artifacts"][0]["_type"] == WORKFLOW_FILE_REF_TYPE

    def test_artifacts_defaults_to_empty_list_when_absent(self):
        """Trace artifacts field should default to [] when not provided."""
        from seer.core.nodes.agent_node import _build_agent_trace

        trace = _build_agent_trace(
            node_id="agent_1",
            inputs={},
            status="succeeded",
            success_data={
                "prompt": "Hi",
                "tool_names": [],
                "steps": [],
                "result_value": "Hi back",
            },
        )

        assert trace["artifacts"] == []
