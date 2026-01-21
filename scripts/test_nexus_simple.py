#!/usr/bin/env python3
"""
Simplified test script for Nexus API.

Creates a test workflow and starts an interactive chat session.
Type messages to design workflows with Nexus, accept/reject proposals.
Type 'quit' to exit.

Usage:
    uv run python scripts/test_nexus_simple.py
"""
import asyncio
import json
import sys
from typing import Any

import httpx

from nexus_common import (
    ChatSession,
    Colors,
    NexusAPIError,
    create_test_token,
    create_test_workflow,
    get_user_input,
    handle_proposal_acceptance,
    process_user_message,
)


def print_header() -> None:
    """Print chat header."""
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}Nexus Interactive Chat{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")
    print("Type your messages to chat with Nexus. Type 'quit' to exit.\n")
    print(f"{Colors.CYAN}Example prompts:{Colors.ENDC}")
    print("  • Build a workflow that sends daily email reports")
    print("  • Create a workflow to sync data from Airtable to Notion")
    print("  • Design a customer onboarding workflow\n")


def display_proposal(proposal: dict[str, Any]) -> None:
    """Display workflow proposal summary."""
    print(f"{Colors.BOLD}{Colors.YELLOW}📋 Workflow Proposal #{proposal['id']}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Summary: {proposal.get('summary', 'No summary')}{Colors.ENDC}")

    spec = proposal.get('spec', {})
    nodes = spec.get('nodes', [])
    node_types = ', '.join(n.get('type', 'unknown') for n in nodes[:3])
    if len(nodes) > 3:
        node_types += '...'
    print(f"{Colors.YELLOW}Nodes: {len(nodes)} ({node_types}){Colors.ENDC}\n")


async def handle_proposal(session: ChatSession, proposal: dict[str, Any]) -> None:
    """Handle workflow proposal acceptance/rejection."""
    display_proposal(proposal)

    while True:
        action = input(f"{Colors.YELLOW}Accept proposal? (yes/no/view): {Colors.ENDC}").strip().lower()

        if action == "view":
            print(f"\n{Colors.CYAN}Full Proposal Spec:{Colors.ENDC}")
            print(json.dumps(proposal.get('spec', {}), indent=2))
            print()
        elif action in ("yes", "y"):
            print(f"{Colors.CYAN}Accepting proposal...{Colors.ENDC}")
            result = await session.accept_proposal(proposal["id"])
            await handle_proposal_acceptance(result)
            print()  # Blank line
            break
        elif action in ("no", "n"):
            print(f"{Colors.CYAN}Rejecting proposal...{Colors.ENDC}")
            await session.reject_proposal(proposal["id"])
            print(f"{Colors.YELLOW}✓ Proposal rejected{Colors.ENDC}\n")
            break
        else:
            print(f"{Colors.RED}Please enter 'yes', 'no', or 'view'{Colors.ENDC}")


async def interactive_chat_loop(session: ChatSession) -> None:
    """Run interactive chat loop with Nexus."""
    print_header()

    while True:
        try:
            user_input = get_user_input()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "/quit"):
                print(f"{Colors.YELLOW}Goodbye!{Colors.ENDC}")
                break

            await process_user_message(session, user_input, handle_proposal)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Use 'quit' to exit.{Colors.ENDC}")
        except NexusAPIError as e:
            print(f"{Colors.RED}API Error: {e}{Colors.ENDC}\n")


async def main() -> None:
    """Main entry point."""
    base_url = "http://localhost:8000"
    token = create_test_token()

    async with httpx.AsyncClient() as client:
        try:
            # Check API health
            response = await client.get(f"{base_url}/health")
            if response.status_code != 200:
                print(f"{Colors.RED}Error: API not available at {base_url}{Colors.ENDC}")
                print(f"{Colors.YELLOW}Make sure the API is running: docker compose up{Colors.ENDC}")
                sys.exit(1)

            print(f"{Colors.GREEN}✓ Connected to API at {base_url}{Colors.ENDC}")

            # Create workflow
            workflow_id = await create_test_workflow(client, base_url, token)
            print()  # Blank line after workflow creation

            # Start interactive chat
            session = ChatSession(client, base_url, token, workflow_id)
            await interactive_chat_loop(session)

        except NexusAPIError as e:
            print(f"{Colors.RED}Fatal error: {e}{Colors.ENDC}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
