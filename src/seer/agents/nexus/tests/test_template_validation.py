"""
Template validation tests.

Ensures all workflow templates have valid structure and metadata.
"""
import json
from pathlib import Path

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


class TestTemplateValidation:
    """Validate that all templates compile successfully."""


    def test_customer_support_email_structure(self):
        """Validate customer support email template structure."""
        template = load_template("customer_support_email")
        spec = template["spec"]

        # Verify key structural elements
        assert spec["version"] == "2"
        assert len(spec["triggers"]) == 1
        assert spec["triggers"][0]["key"] == "poll.gmail.email_received"
        assert len(spec["nodes"]) > 10  # Complex workflow
        assert len(spec["edges"]) > 10  # With conditional branching

        # Verify conditional nodes exist
        node_types = {node["type"] for node in spec["nodes"]}
        assert "if" in node_types, "Should have conditional branching"
        assert "agent" in node_types, "Should have agent classification"
        assert "tool" in node_types, "Should have tool nodes"

        logger.info("✅ Customer support template structure valid")

    def test_lead_generation_form_structure(self):
        """Validate lead generation form template structure."""
        template = load_template("lead_generation_form")
        spec = template["spec"]

        # Verify key structural elements
        assert spec["version"] == "2"
        assert len(spec["triggers"]) == 1
        assert spec["triggers"][0]["key"] == "form.hosted"
        assert "schemas" in spec["triggers"][0], "Form trigger should have field schemas"

        # Verify conditional routing
        node_types = {node["type"] for node in spec["nodes"]}
        assert "if" in node_types, "Should have conditional quality routing"
        assert "agent" in node_types, "Should have AI scoring"

        logger.info("✅ Lead generation template structure valid")

    def test_email_support_chatbot_structure(self):
        """Validate email support chatbot template structure."""
        template = load_template("email_support_chatbot")
        spec = template["spec"]

        # Verify key structural elements
        assert spec["version"] == "2"
        assert len(spec["triggers"]) == 1
        assert spec["triggers"][0]["key"] == "poll.gmail.email_received"

        # Verify KB search pattern
        node_types = {node["type"] for node in spec["nodes"]}
        assert "if" in node_types, "Should have conditional KB routing"
        assert "agent" in node_types, "Should have AI response generation"

        # Check for KB-specific nodes
        node_ids = {node["id"] for node in spec["nodes"]}
        assert "search_kb" in node_ids, "Should search knowledge base"
        assert "check_kb_results" in node_ids, "Should check KB results"

        logger.info("✅ Email chatbot template structure valid")

    def test_all_templates_have_required_fields(self):
        """Verify all templates have required metadata fields."""
        templates_dir = Path(__file__).parent.parent / "templates"
        template_files = list(templates_dir.glob("*.json"))

        assert len(template_files) >= 6, "Should have at least 6 templates"

        for template_file in template_files:
            with open(template_file, "r", encoding="utf-8") as f:
                template = json.load(f)

            # Check metadata fields
            assert "name" in template, f"{template_file.name}: missing 'name'"
            assert "description" in template, f"{template_file.name}: missing 'description'"
            assert "tags" in template, f"{template_file.name}: missing 'tags'"
            assert "spec" in template, f"{template_file.name}: missing 'spec'"

            # Check spec fields
            spec = template["spec"]
            assert "version" in spec, f"{template_file.name}: missing 'version' in spec"
            assert "nodes" in spec, f"{template_file.name}: missing 'nodes' in spec"

            # Version 1 specs require 'output', version 2 uses explicit edges
            if spec.get("version") == "1":
                assert "output" in spec, f"{template_file.name}: version 1 spec missing 'output'"
            elif spec.get("version") == "2":
                assert "edges" in spec, f"{template_file.name}: version 2 spec missing 'edges'"

            logger.info("✅ Template %s has all required fields", template["name"])
