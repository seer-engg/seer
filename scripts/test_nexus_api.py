#!/usr/bin/env python3
# pylint: disable=broad-exception-caught  # Reason: Test script needs to catch all exceptions to report test failures
"""
Automated API tests for Nexus chat agent.
Tests workflow creation, multi-turn conversations, proposal handling, and error cases.
Run with: uv run python scripts/test_nexus_api.py
"""
import asyncio
import sys
import traceback

import httpx

from nexus_common import (
    ChatSession,
    Colors,
    NexusAPIError,
    create_test_token,
    create_test_workflow,
)


class TestResult:
    """Tracks test pass/fail status."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[tuple[str, str]] = []

    def pass_test(self, name: str) -> None:
        """Mark test as passed."""
        print(f"{Colors.GREEN}✓ PASS{Colors.ENDC}: {name}")
        self.passed += 1

    def fail_test(self, name: str, reason: str) -> None:
        """Mark test as failed."""
        print(f"{Colors.RED}✗ FAIL{Colors.ENDC}: {name}")
        print(f"  {Colors.RED}Reason: {reason}{Colors.ENDC}")
        self.failed += 1
        self.errors.append((name, reason))

    def summary(self) -> None:
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}Test Summary{Colors.ENDC}")
        print(f"Total: {total}, Passed: {Colors.GREEN}{self.passed}{Colors.ENDC}, Failed: {Colors.RED}{self.failed}{Colors.ENDC}")

        if self.errors:
            print(f"\n{Colors.RED}Failed Tests:{Colors.ENDC}")
            for name, reason in self.errors:
                print(f"  • {name}: {reason}")

        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


async def test_chat_creates_session(session: ChatSession, results: TestResult) -> None:
    """Test: Chat interaction creates session and returns valid response."""
    test_name = "Chat creates session"

    try:
        session.clear_session()
        response = await session.send_message("Hello, can you help me create a workflow?")

        if not response.get("response"):
            results.fail_test(test_name, "No response text returned")
            return

        if not response.get("thread_id"):
            results.fail_test(test_name, "No thread_id returned")
            return

        if not response.get("session_id"):
            results.fail_test(test_name, "No session_id returned")
            return

        results.pass_test(test_name)

    except Exception as e:
        results.fail_test(test_name, str(e))


async def test_multi_turn_conversation(session: ChatSession, results: TestResult) -> None:
    """Test: Multi-turn conversation maintains context."""
    test_name = "Multi-turn conversation"

    try:
        session.clear_session()

        # First message
        response1 = await session.send_message("I want to create a workflow for email notifications")
        thread_id = response1.get("thread_id")

        # Second message - should use same thread
        response2 = await session.send_message("Can you show me what tools are available?")

        if response2.get("thread_id") != thread_id:
            results.fail_test(test_name, "Thread ID changed between messages")
            return

        if not response2.get("response"):
            results.fail_test(test_name, "No response in second message")
            return

        results.pass_test(test_name)

    except Exception as e:
        results.fail_test(test_name, str(e))


async def test_workflow_proposal_creation(session: ChatSession, results: TestResult) -> None:
    """Test: Specific workflow request generates proposal."""
    test_name = "Workflow proposal creation"

    try:
        session.clear_session()

        response = await session.send_message(
            "Create a workflow that sends me a Slack message when I receive an email from john@example.com"
        )

        # Agent should eventually generate a proposal
        # It might not be in first response if using supervisor architecture
        has_proposal = response.get("proposal") is not None
        has_thinking = response.get("thinking") is not None

        if not has_proposal and not has_thinking:
            results.fail_test(test_name, "No proposal or thinking steps returned for workflow request")
            return

        if has_proposal:
            proposal = response["proposal"]
            if not proposal.get("id"):
                results.fail_test(test_name, "Proposal missing ID")
                return

            # Spec might be nested or at top level
            spec = proposal.get("spec") or proposal.get("workflow_spec")
            if not spec:
                results.fail_test(test_name, "Proposal missing workflow spec")
                return

        results.pass_test(test_name)

    except Exception as e:
        results.fail_test(test_name, str(e))


async def test_proposal_acceptance(session: ChatSession, results: TestResult) -> None:
    """Test: Accepting proposal applies changes."""
    test_name = "Proposal acceptance"

    try:
        session.clear_session()

        # Request workflow
        response = await session.send_message(
            "Create a simple workflow with a schedule trigger that runs every hour"
        )

        proposal = response.get("proposal")
        if not proposal:
            # Try a follow-up to get proposal
            response = await session.send_message("Yes, create that workflow")
            proposal = response.get("proposal")

        if not proposal:
            results.fail_test(test_name, "Could not generate proposal for test")
            return

        proposal_id = proposal.get("id")
        if not proposal_id:
            results.fail_test(test_name, "Proposal missing ID")
            return

        # Accept proposal
        accept_response = await session.accept_proposal(proposal_id)

        if not accept_response.get("success"):
            results.fail_test(test_name, f"Accept response indicates failure: {accept_response}")
            return

        results.pass_test(test_name)

    except Exception as e:
        results.fail_test(test_name, str(e))


async def test_error_invalid_workflow_id(client: httpx.AsyncClient, base_url: str, token: str, results: TestResult) -> None:
    """Test: Invalid workflow ID returns proper error."""
    test_name = "Invalid workflow ID error"

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post(
            f"{base_url}/api/nexus/invalid-workflow-999/chat",
            json={"message": "test", "workflow_state": {}},
            headers=headers,
            timeout=30.0
        )

        # Should return 404 or similar error
        if response.status_code == 200:
            results.fail_test(test_name, "Invalid workflow ID did not return error")
            return

        if response.status_code not in [400, 404]:
            results.fail_test(test_name, f"Unexpected status code: {response.status_code}")
            return

        results.pass_test(test_name)

    except Exception as e:
        results.fail_test(test_name, str(e))


async def test_error_malformed_request(client: httpx.AsyncClient, base_url: str, token: str, workflow_id: str, results: TestResult) -> None:
    """Test: Malformed request returns proper error."""
    test_name = "Malformed request error"

    try:
        headers = {"Authorization": f"Bearer {token}"}

        # Missing required 'message' field
        response = await client.post(
            f"{base_url}/api/nexus/{workflow_id}/chat",
            json={"workflow_state": {}},
            headers=headers,
            timeout=30.0
        )

        if response.status_code == 200:
            results.fail_test(test_name, "Malformed request did not return error")
            return

        if response.status_code != 422:  # FastAPI validation error
            results.fail_test(test_name, f"Expected 422, got {response.status_code}")
            return

        results.pass_test(test_name)

    except Exception as e:
        results.fail_test(test_name, str(e))


async def test_list_sessions(client: httpx.AsyncClient, base_url: str, token: str, workflow_id: str, results: TestResult) -> None:
    """Test: List sessions endpoint works."""
    test_name = "List chat sessions"

    try:
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get(
            f"{base_url}/api/nexus/{workflow_id}/chat/sessions",
            headers=headers,
            timeout=30.0
        )

        if response.status_code != 200:
            results.fail_test(test_name, f"Status code: {response.status_code}")
            return

        sessions = response.json()
        if not isinstance(sessions, list):
            results.fail_test(test_name, "Response is not a list")
            return

        results.pass_test(test_name)

    except Exception as e:
        results.fail_test(test_name, str(e))


async def test_get_session_with_messages(session: ChatSession, client: httpx.AsyncClient, base_url: str, token: str, results: TestResult) -> None:
    """Test: Get session with messages endpoint."""
    test_name = "Get session with messages"

    try:
        session.clear_session()

        # Create a session by sending message
        await session.send_message("Test message for session retrieval")

        if not session.session_id:
            results.fail_test(test_name, "No session_id after chat")
            return

        # Get session details
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(
            f"{base_url}/api/nexus/{session.workflow_id}/chat/sessions/{session.session_id}",
            headers=headers,
            timeout=30.0
        )

        if response.status_code != 200:
            results.fail_test(test_name, f"Status code: {response.status_code}")
            return

        session_data = response.json()
        if "messages" not in session_data:
            results.fail_test(test_name, "Response missing 'messages' field")
            return

        results.pass_test(test_name)

    except Exception as e:
        results.fail_test(test_name, str(e))


async def run_all_tests(base_url: str = "http://localhost:8000") -> TestResult:
    """Run all API tests."""
    results = TestResult()

    print(f"\n{Colors.BOLD}{Colors.HEADER}Nexus API Test Suite{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    print(f"Testing against: {base_url}")
    print(f"{Colors.YELLOW}Note: Server must be running at {base_url}{Colors.ENDC}\n")

    # Setup
    token = create_test_token()

    async with httpx.AsyncClient() as client:
        try:
            # Create test workflow
            print(f"{Colors.CYAN}Setting up test workflow...{Colors.ENDC}")
            workflow_id = await create_test_workflow(client, base_url, token)
            session = ChatSession(client, base_url, token, workflow_id)
            print()

            # Run tests
            print(f"{Colors.BOLD}Running tests...{Colors.ENDC}\n")

            await test_chat_creates_session(session, results)
            await test_multi_turn_conversation(session, results)
            await test_workflow_proposal_creation(session, results)
            await test_proposal_acceptance(session, results)
            await test_error_invalid_workflow_id(client, base_url, token, results)
            await test_error_malformed_request(client, base_url, token, workflow_id, results)
            await test_list_sessions(client, base_url, token, workflow_id, results)
            await test_get_session_with_messages(session, client, base_url, token, results)

        except NexusAPIError as e:
            print(f"\n{Colors.RED}Setup Error: {e}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Make sure the server is running: uv run uvicorn seer.api.main:app --port 8000{Colors.ENDC}")
            sys.exit(1)
        except Exception as e:
            print(f"\n{Colors.RED}Unexpected Error: {e}{Colors.ENDC}")
            traceback.print_exc()
            sys.exit(1)

    return results


async def main() -> None:
    """Main entry point."""
    results = await run_all_tests()
    results.summary()

    # Exit with error code if tests failed
    if results.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
