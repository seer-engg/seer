#!/usr/bin/env python3
# pylint: disable=too-complex,too-many-statements  # Reason: Demo script with detailed output for user feedback
"""
Manual test script for Nexus workflow generation.

Usage:
    uv run python scripts/test_nexus_live.py "Your workflow prompt"

Examples:
    uv run python scripts/test_nexus_live.py "Send email when user signs up"
    uv run python scripts/test_nexus_live.py "Fetch news and summarize with AI"
"""
import sys
import asyncio
import json
from seer.agents.nexus.tools.workflow_tools import create_workflow_spec_structured
from seer.llm import get_llm_without_responses_api
from seer.core.compiler.parse import parse_workflow_spec


# Example tool/trigger catalogs for testing
SAMPLE_TOOLS = [
    {"tool": "gmail_send_email", "integration": "gmail", "description": "Send email via Gmail"},
    {"tool": "demo.news_search", "integration": "demo", "description": "Search news articles"},
    {"tool": "supabase_table_insert", "integration": "supabase", "description": "Insert row into Supabase table"},
    {"tool": "slack_send_message", "integration": "slack", "description": "Send message to Slack channel"},
]

SAMPLE_TRIGGERS = [
    {"key": "webhook.supabase.db_changes", "provider": "supabase", "description": "Supabase database change webhook"},
    {"key": "poll.gmail.email_received", "provider": "gmail", "description": "New email received poll"},
    {"key": "form.hosted", "provider": "seer", "description": "Hosted form submission"},
]


def print_section(title: str):
    """Print a section header."""
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)
    print()


def select_relevant_tools(user_intent: str) -> list:
    """Simple keyword matching to select relevant tools."""
    intent_lower = user_intent.lower()
    relevant = []

    for tool in SAMPLE_TOOLS:
        tool_keywords = tool["tool"].lower().split("_") + tool["integration"].lower().split()
        if any(keyword in intent_lower for keyword in tool_keywords):
            relevant.append(tool)

    return relevant if relevant else []


def select_relevant_triggers(user_intent: str) -> list:
    """Simple keyword matching to select relevant triggers."""
    intent_lower = user_intent.lower()
    relevant = []

    # Trigger keywords
    trigger_keywords = ["when", "on", "trigger", "webhook", "poll", "form", "signup", "change"]
    if not any(kw in intent_lower for kw in trigger_keywords):
        return []

    for trigger in SAMPLE_TRIGGERS:
        trigger_words = trigger["key"].lower().split(".") + trigger["description"].lower().split()
        if any(word in intent_lower for word in trigger_words):
            relevant.append(trigger)

    return relevant


async def test_workflow_generation(user_prompt: str):
    """Test Nexus workflow generation with a user prompt."""
    print_section("Nexus Live Test")
    print(f"User Intent: {user_prompt}")

    # Simulate tool/trigger discovery
    discovered_tools = select_relevant_tools(user_prompt)
    discovered_triggers = select_relevant_triggers(user_prompt)

    print()
    print(f"Discovered Tools: {len(discovered_tools)}")
    for tool in discovered_tools:
        print(f"  - {tool['tool']} ({tool['integration']}): {tool['description']}")

    print()
    print(f"Discovered Triggers: {len(discovered_triggers)}")
    for trigger in discovered_triggers:
        print(f"  - {trigger['key']} ({trigger['provider']}): {trigger['description']}")

    # Generate workflow
    print_section("Generating Workflow")
    print("Calling LLM with structured output...")

    try:
        llm = get_llm_without_responses_api(model="gpt-4o-mini", temperature=0)

        proposal = create_workflow_spec_structured(
            llm=llm,
            user_intent=user_prompt,
            discovered_tools=discovered_tools,
            discovered_triggers=discovered_triggers,
        )

        print("✓ Workflow generated successfully")

        # Display proposal
        print_section("Workflow Proposal")
        print(f"Summary: {proposal.summary}")
        print()
        print(f"Reasoning: {proposal.reasoning}")

        # Display workflow spec
        print_section("Workflow Specification")
        spec_dict = proposal.spec.model_dump()

        print(f"Version: {spec_dict['version']}")
        print(f"Nodes: {len(spec_dict['nodes'])}")
        print(f"Edges: {len(spec_dict['edges'])}")
        print(f"Triggers: {len(spec_dict.get('triggers', []))}")
        print()

        # Show nodes
        print("Nodes:")
        for node in spec_dict["nodes"]:
            node_type = node["type"]
            node_id = node["id"]
            print(f"  [{node_id}] {node_type}", end="")
            if node_type == "tool":
                print(f" - {node.get('tool', 'N/A')}")
            elif node_type == "task":
                print(f" - {node.get('kind', 'N/A')}")
            elif node_type == "llm":
                model = node.get("inputs", {}).get("model", "N/A")
                print(f" - {model}")
            else:
                print()

        # Show edges
        print()
        print("Edges:")
        for edge in spec_dict["edges"]:
            edge_type = edge.get("type", "default")
            print(f"  {edge['source']} -> {edge['target']} ({edge_type})")

        # Show triggers
        if spec_dict.get("triggers"):
            print()
            print("Triggers:")
            for trigger in spec_dict["triggers"]:
                print(f"  [{trigger['id']}] {trigger['key']} ({trigger['mode']})")

        # Validate with Pydantic
        print_section("Validation")
        print("Validating with Pydantic...")

        try:
            parsed_spec = parse_workflow_spec(spec_dict)
            print("✓ Pydantic validation passed")
            print(f"  Parsed {len(parsed_spec.nodes)} nodes")
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Demo script catches all for user feedback
            print(f"✗ Pydantic validation failed: {e}")
            return False

        # Show full spec as JSON
        print_section("Full Workflow JSON")
        print(json.dumps(spec_dict, indent=2))

        print_section("Test Complete")
        print("✓ All validations passed")
        print(f"✓ Generated workflow has {len(spec_dict['nodes'])} nodes and {len(spec_dict['edges'])} edges")

        return True

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Demo script catches all for user feedback
        print(f"✗ Workflow generation failed: {e}")
        import traceback  # pylint: disable=import-outside-toplevel  # Reason: Only import for error handling
        traceback.print_exc()
        return False


async def interactive_mode():
    """Interactive mode for testing multiple prompts."""
    print_section("Nexus Interactive Test Mode")
    print("Enter workflow prompts to test (type 'quit' to exit)")
    print()

    while True:
        try:
            prompt = input("Workflow prompt: ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            if not prompt:
                continue

            success = await test_workflow_generation(prompt)
            if not success:
                print("\n⚠️  Test failed. Try another prompt or fix the issue.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Interactive mode catches all for user feedback
            print(f"\n✗ Error: {e}")
            import traceback  # pylint: disable=import-outside-toplevel  # Reason: Only import for error handling
            traceback.print_exc()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # No arguments - run interactive mode
        asyncio.run(interactive_mode())
    else:
        # Single prompt from command line
        prompt = " ".join(sys.argv[1:])
        success = asyncio.run(test_workflow_generation(prompt))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
