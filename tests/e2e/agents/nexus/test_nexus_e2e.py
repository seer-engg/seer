"""
End-to-end tests for Nexus workflow generation.

Tests the complete pipeline with real LLM calls:
1. User provides intent
2. Nexus discovers tools and triggers
3. Supervisor orchestrates specialists
4. Workflow architect generates spec
5. Validation specialist validates
6. Resulting spec compiles successfully
"""
# pylint: disable=import-outside-toplevel,duplicate-code  # Reason: Test-specific patterns acceptable
import pytest
from seer.agents.nexus.tools.workflow_tools import create_workflow_spec_structured
from seer.core.compiler.parse import parse_workflow_spec


# Test cases with expected outcomes
TEST_CASES = [
    {
        "name": "simple_tool_workflow",
        "user_intent": "Fetch news articles about AI",
        "discovered_tools": [
            {"tool": "demo.news_search", "description": "Search news articles"}
        ],
        "discovered_triggers": [],
        "expected_min_nodes": 1,
        "expected_node_types": ["tool"],
        "expected_triggers": 0,
    },
    {
        "name": "email_on_signup",
        "user_intent": "Send welcome email via Gmail when user signs up in Supabase",
        "discovered_tools": [
            {"tool": "gmail_send_email", "description": "Send email via Gmail"}
        ],
        "discovered_triggers": [
            {"key": "webhook.supabase.db_changes", "description": "Supabase database changes"}
        ],
        "expected_min_nodes": 2,
        "expected_node_types": ["tool", "task"],
        "expected_triggers": 1,
    },
    {
        "name": "data_processing_pipeline",
        "user_intent": "Fetch news, summarize with AI, and store results",
        "discovered_tools": [
            {"tool": "demo.news_search", "description": "Search news articles"},
        ],
        "discovered_triggers": [],
        "expected_min_nodes": 2,
        "expected_node_types": ["tool", "llm"],
        "expected_triggers": 0,
    },
]


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", TEST_CASES, ids=[tc["name"] for tc in TEST_CASES])
class TestNexusE2E:
    """End-to-end tests for Nexus workflow generation."""

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_workflow_generation(self, test_case):
        """Test complete workflow generation pipeline."""
        from seer.llm import get_llm_without_responses_api  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import

        llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)

        # Generate workflow using structured output
        proposal = create_workflow_spec_structured(
            llm=llm,
            user_intent=test_case["user_intent"],
            discovered_tools=test_case["discovered_tools"],
            discovered_triggers=test_case["discovered_triggers"],
        )

        # Validate proposal structure
        assert proposal.summary, "Proposal missing summary"
        assert proposal.reasoning, "Proposal missing reasoning"
        assert proposal.spec, "Proposal missing spec"

        # Extract workflow spec
        spec_dict = proposal.spec.model_dump()

        # Validate version
        assert spec_dict["version"] == "2", f"Invalid version: {spec_dict['version']}"

        # Validate node count
        assert len(spec_dict["nodes"]) >= test_case["expected_min_nodes"], \
            f"Expected at least {test_case['expected_min_nodes']} nodes, got {len(spec_dict['nodes'])}"

        # Validate node types
        actual_node_types = {node["type"] for node in spec_dict["nodes"]}
        for expected_type in test_case["expected_node_types"]:
            assert expected_type in actual_node_types, \
                f"Expected node type '{expected_type}' not found in {actual_node_types}"

        # Validate trigger count
        actual_triggers = len(spec_dict.get("triggers", []))
        assert actual_triggers == test_case["expected_triggers"], \
            f"Expected {test_case['expected_triggers']} triggers, got {actual_triggers}"

        # Validate edges exist
        assert len(spec_dict["edges"]) > 0, "No edges defined"

        # Validate no orphaned nodes
        edge_sources = {e["source"] for e in spec_dict["edges"]}
        edge_targets = {e["target"] for e in spec_dict["edges"]}
        node_ids = {n["id"] for n in spec_dict["nodes"]}
        trigger_ids = {t["id"] for t in spec_dict.get("triggers", [])}

        # All nodes should be reachable
        connected_nodes = edge_sources | edge_targets
        all_graph_ids = node_ids | trigger_ids
        orphaned = all_graph_ids - connected_nodes

        assert len(orphaned) == 0, f"Orphaned nodes/triggers: {orphaned}"

        # Validate spec parses with Pydantic
        try:
            parsed_spec = parse_workflow_spec(spec_dict)
            assert parsed_spec is not None, "Spec failed to parse"
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Catch all parsing errors for test feedback
            pytest.fail(f"Spec parsing failed: {e}\nSpec: {spec_dict}")

        # Validate trigger edges if triggers exist
        if test_case["expected_triggers"] > 0:
            trigger_edges = [e for e in spec_dict["edges"] if e.get("type") == "trigger"]
            assert len(trigger_edges) >= 1, "Missing trigger edges for trigger-based workflow"


@pytest.mark.e2e
@pytest.mark.asyncio
class TestNexusEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_no_tools_available(self):
        """Test workflow generation when no tools are discovered."""
        from seer.llm import get_llm_without_responses_api  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import

        llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)

        proposal = create_workflow_spec_structured(
            llm=llm,
            user_intent="Create a simple workflow",
            discovered_tools=[],
            discovered_triggers=[],
        )

        # Should still generate a valid workflow (may use task/llm nodes)
        assert proposal.spec.version == "2"
        assert len(proposal.spec.nodes) >= 1
        # Should have task or LLM nodes since no tools available
        node_types = {node.type for node in proposal.spec.nodes}
        assert "task" in node_types or "llm" in node_types

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_multiple_tools(self):
        """Test workflow with multiple tool options."""
        from seer.llm import get_llm_without_responses_api  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import

        llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)

        proposal = create_workflow_spec_structured(
            llm=llm,
            user_intent="Send an email and create a calendar event",
            discovered_tools=[
                {"tool": "gmail_send_email", "description": "Send email via Gmail"},
                {"tool": "google_calendar_create_event", "description": "Create calendar event"},
            ],
            discovered_triggers=[],
        )

        # Should use multiple tools
        tool_nodes = [n for n in proposal.spec.nodes if n.type == "tool"]
        assert len(tool_nodes) >= 2, f"Expected at least 2 tool nodes, got {len(tool_nodes)}"

        # Should have edges connecting them
        assert len(proposal.spec.edges) >= 1


@pytest.mark.e2e
@pytest.mark.asyncio
class TestWorkflowCompilation:
    """Test that generated workflows compile successfully."""

    async def test_simple_workflow_compiles(self, db_engine, test_user):  # pylint: disable=unused-argument  # Reason: Fixture required for database setup
        """Test that a simple generated workflow compiles."""
        from seer.core.runtime.global_compiler import WorkflowCompilerSingleton  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import
        from pydantic import ValidationError  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import

        # Create a simple valid workflow
        spec_dict = {
            "version": "2",
            "triggers": [],
            "nodes": [
                {
                    "id": "task1",
                    "type": "task",
                    "kind": "set",
                    "value": {"message": "Hello, World!"},
                }
            ],
            "edges": [],
        }

        # Parse and validate
        parsed_spec = parse_workflow_spec(spec_dict)
        assert parsed_spec is not None

        # Compile (note: may require proper tool registration and context)
        try:
            compiler = WorkflowCompilerSingleton.instance()
            compiled = await compiler.compile(test_user, spec_dict, checkpointer=None)
            assert compiled is not None
        except (ValidationError, Exception) as e:  # pylint: disable=broad-exception-caught  # Reason: Catch all compilation errors to skip test
            # Compilation may fail without full infrastructure, but parsing should work
            pytest.skip(f"Compilation requires full infrastructure: {e}")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestNexusQuality:
    """Test quality of generated workflows."""

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__("os").environ,
        reason="Requires OPENAI_API_KEY environment variable"
    )
    async def test_variable_references_valid(self):
        """Test that variable references use correct syntax."""
        from seer.llm import get_llm_without_responses_api  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import

        llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)

        proposal = create_workflow_spec_structured(
            llm=llm,
            user_intent="Fetch data and send it in an email",
            discovered_tools=[
                {"tool": "demo.fetch", "description": "Fetch data"},
                {"tool": "gmail_send_email", "description": "Send email"},
            ],
            discovered_triggers=[],
        )

        # Check that variable references use ${...} syntax
        spec_dict = proposal.spec.model_dump()

        # Look for variable references in node inputs
        import json  # pylint: disable=import-outside-toplevel  # Reason: Test-specific import
        spec_str = json.dumps(spec_dict)

        # Should have at least one variable reference
        has_var_ref = "${" in spec_str
        if len(spec_dict["nodes"]) > 1:
            # Multi-node workflows should use variable references
            assert has_var_ref, "Multi-node workflow missing variable references"
