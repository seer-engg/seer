"""
Simplified workflow validation tests for v2 schema.

Tests tool discovery workflows without complex trigger configurations.
"""
import json
from pathlib import Path
import pytest
from seer.core.runtime.global_compiler import WorkflowCompilerSingleton
from seer.database import User
from seer.logger import get_logger

logger = get_logger(__name__)


def load_template(template_name: str) -> dict:
    """Load a template from the templates directory."""
    templates_dir = Path(__file__).parent.parent / "templates"
    template_path = templates_dir / f"{template_name}.json"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestSimplifiedWorkflowValidation:
    """Validate simplified v2 workflows compile successfully."""

    @pytest.mark.asyncio
    async def test_simple_tool_workflow_compiles(self):
        """Test single tool workflow compiles."""
        template = load_template("simple_tool_workflow")
        spec = template["spec"]

        logger.info("Validating template: %s", template["name"])

        compiler = WorkflowCompilerSingleton.instance()
        mock_user = User(id=1, email="test@example.com")

        try:
            result = await compiler.compile(mock_user, spec)
            logger.info("✅ Template compiled successfully")
            logger.info("Nodes: %s", [node["id"] for node in spec["nodes"]])
            assert result is not None, "Compilation should succeed"
        except Exception as e:  # pylint: disable=broad-except  # Test needs to catch all compilation failures for reporting
            logger.error("❌ Template compilation failed: %s", e)
            pytest.fail(f"Template failed to compile: {e}")

    @pytest.mark.asyncio
    async def test_multi_tool_workflow_compiles(self):
        """Test multi-tool chained workflow compiles."""
        template = load_template("multi_tool_workflow")
        spec = template["spec"]

        logger.info("Validating template: %s", template["name"])

        compiler = WorkflowCompilerSingleton.instance()
        mock_user = User(id=1, email="test@example.com")

        try:
            result = await compiler.compile(mock_user, spec)
            logger.info("✅ Template compiled successfully")
            logger.info("Nodes: %s", [node["id"] for node in spec["nodes"]])
            logger.info("Edges: %s", len(spec.get("edges", [])))
            assert result is not None, "Compilation should succeed"
        except Exception as e:  # pylint: disable=broad-except  # Test needs to catch all compilation failures for reporting
            logger.error("❌ Template compilation failed: %s", e)
            pytest.fail(f"Template failed to compile: {e}")

    @pytest.mark.asyncio
    async def test_llm_tool_workflow_compiles(self):
        """Test LLM + tool mixed workflow compiles."""
        template = load_template("llm_tool_workflow")
        spec = template["spec"]

        logger.info("Validating template: %s", template["name"])

        compiler = WorkflowCompilerSingleton.instance()
        mock_user = User(id=1, email="test@example.com")

        try:
            result = await compiler.compile(mock_user, spec)
            logger.info("✅ Template compiled successfully")
            logger.info("Nodes: %s", [node["id"] for node in spec["nodes"]])
            logger.info("Node types: %s", [node["type"] for node in spec["nodes"]])
            assert result is not None, "Compilation should succeed"
        except Exception as e:  # pylint: disable=broad-except  # Test needs to catch all compilation failures for reporting
            logger.error("❌ Template compilation failed: %s", e)
            pytest.fail(f"Template failed to compile: {e}")

    @pytest.mark.asyncio
    async def test_all_simplified_templates_have_queries(self):
        """Verify all templates have test queries for RLM verification."""
        templates_to_test = [
            "simple_tool_workflow",
            "multi_tool_workflow",
            "llm_tool_workflow"
        ]

        for template_name in templates_to_test:
            template = load_template(template_name)

            # Check test query fields
            assert "queries" in template, f"{template_name}: missing 'queries' field"
            assert "expected_tools" in template, f"{template_name}: missing 'expected_tools' field"

            assert len(template["queries"]) >= 3, f"{template_name}: should have at least 3 test queries"
            assert len(template["expected_tools"]) >= 1, f"{template_name}: should have at least 1 expected tool"

            logger.info("✅ Template %s has required RLM test fields", template["name"])
            logger.info("  Queries: %d", len(template["queries"]))
            logger.info("  Expected tools: %s", template["expected_tools"])
