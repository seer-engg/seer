#!/usr/bin/env python3
"""
Interactive test script for Nexus API.

Tests the Nexus chat agent locally by:
- Creating a test workflow
- Sending chat messages to design workflow
- Viewing agent responses with thinking
- Accepting/rejecting proposals
- Supporting both single-agent and supervisor modes

Usage:
    uv run python scripts/test_nexus_local.py [options]
"""
import argparse
import asyncio
import json
import os
import sys

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


def print_header(text: str) -> None:
    """Print colored header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")


def print_chat_instructions() -> None:
    """Print chat instructions and example prompts."""
    print("Type your messages to chat with Nexus about workflow design.")
    print("Commands: /quit (exit), /clear (new session), /help")
    print(f"\n{Colors.CYAN}Example prompts:{Colors.ENDC}")
    print("  • Build a workflow that sends daily email reports")
    print("  • Create a workflow to sync data from Airtable to Notion")
    print("  • Design a customer onboarding workflow with Slack notifications")
    print()


def print_help() -> None:
    """Print help message."""
    print(f"\n{Colors.CYAN}Commands:{Colors.ENDC}")
    print("  /quit  - Exit the chat")
    print("  /clear - Start a new session")
    print("  /help  - Show this help message")
    print()


def display_proposal(proposal: dict) -> None:
    """Display workflow proposal summary."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}📋 Workflow Proposal #{proposal['id']}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Summary: {proposal.get('summary', 'No summary')}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Status: {proposal.get('status', 'pending')}{Colors.ENDC}")

    spec = proposal.get('spec', {})
    nodes = spec.get('nodes', [])
    node_summary = ', '.join(n.get('type', 'unknown') for n in nodes[:3])
    if len(nodes) > 3:
        node_summary += '...'
    print(f"{Colors.YELLOW}Nodes: {len(nodes)} ({node_summary}){Colors.ENDC}")


async def handle_proposal(session: ChatSession, proposal: dict) -> None:
    """Handle workflow proposal acceptance/rejection."""
    display_proposal(proposal)

    while True:
        action = input(f"\n{Colors.YELLOW}Accept proposal? (yes/no/view): {Colors.ENDC}").strip().lower()

        if action == "view":
            print(f"\n{Colors.CYAN}Full Proposal Spec:{Colors.ENDC}")
            print(json.dumps(proposal.get('spec', {}), indent=2))
        elif action in ("yes", "y"):
            print(f"{Colors.CYAN}Accepting proposal...{Colors.ENDC}")
            result = await session.accept_proposal(proposal["id"])
            await handle_proposal_acceptance(result)
            break
        elif action in ("no", "n"):
            print(f"{Colors.CYAN}Rejecting proposal...{Colors.ENDC}")
            await session.reject_proposal(proposal["id"])
            print(f"{Colors.YELLOW}✓ Proposal rejected{Colors.ENDC}")
            break
        else:
            print(f"{Colors.RED}Invalid choice. Please enter 'yes', 'no', or 'view'{Colors.ENDC}")


def handle_command(command: str, session: ChatSession) -> bool:
    """Handle chat commands. Returns True if should continue, False if should exit."""
    if command == "/quit":
        print(f"{Colors.YELLOW}Goodbye!{Colors.ENDC}")
        return False
    if command == "/clear":
        session.clear_session()
        print(f"{Colors.YELLOW}✓ Started new session{Colors.ENDC}")
        return True
    if command == "/help":
        print_help()
        return True
    return True


async def interactive_chat_loop(session: ChatSession) -> None:
    """Run interactive chat loop with Nexus."""
    print_header("Nexus Interactive Chat")
    print_chat_instructions()

    while True:
        try:
            user_input = get_user_input()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                should_continue = handle_command(user_input, session)
                if not should_continue:
                    break
                continue

            # Process message
            await process_user_message(session, user_input, handle_proposal)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted. Use /quit to exit.{Colors.ENDC}")
        except NexusAPIError as e:
            print(f"{Colors.BOLD}{Colors.RED}Error:{Colors.ENDC} {e}")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive Nexus API test script")
    parser.add_argument(
        "--workflow-id",
        type=str,
        help="Existing workflow ID (if not provided, creates new test workflow)"
    )
    parser.add_argument(
        "--supervisor-mode",
        action="store_true",
        help="Enable supervisor mode with multi-agent architecture"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for API (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    # Set supervisor mode if requested
    if args.supervisor_mode:
        os.environ["NEXUS_SUPERVISOR_MODE"] = "true"
        print(f"{Colors.CYAN}🎯 Supervisor mode enabled{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}🎯 Single-agent mode{Colors.ENDC}")

    # Create JWT token
    token = create_test_token()

    # Create HTTP client
    async with httpx.AsyncClient() as client:
        try:
            # Check API health
            response = await client.get(f"{args.base_url}/health")
            if response.status_code != 200:
                print(f"{Colors.BOLD}{Colors.RED}Error:{Colors.ENDC} API not available at {args.base_url}")
                sys.exit(1)
            print(f"{Colors.GREEN}✓ Connected to API at {args.base_url}{Colors.ENDC}")

            # Get or create workflow
            if args.workflow_id:
                workflow_id = args.workflow_id
                print(f"{Colors.GREEN}✓ Using existing workflow: {workflow_id}{Colors.ENDC}")
            else:
                workflow_id = await create_test_workflow(client, args.base_url, token)

            # Start interactive chat
            session = ChatSession(client, args.base_url, token, workflow_id)
            await interactive_chat_loop(session)

        except NexusAPIError as e:
            print(f"{Colors.BOLD}{Colors.RED}Error:{Colors.ENDC} Fatal error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
