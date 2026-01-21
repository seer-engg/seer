"""
Common utilities for Nexus test scripts.
Shared classes, functions, and constants.
"""
from typing import Any, Optional

import httpx
import jwt


class NexusAPIError(Exception):
    """Error communicating with Nexus API."""


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ChatSession:
    """Manages Nexus chat session state."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, token: str, workflow_id: str):
        self.client = client
        self.base_url = base_url
        self.token = token
        self.workflow_id = workflow_id
        self.thread_id: Optional[str] = None
        self.session_id: Optional[int] = None
        self.headers = {"Authorization": f"Bearer {token}"}

    def clear_session(self) -> None:
        """Clear session state to start fresh."""
        self.thread_id = None
        self.session_id = None

    async def send_message(self, message: str) -> dict[str, Any]:
        """Send a chat message to Nexus."""
        chat_data: dict[str, Any] = {
            "message": message,
            "workflow_state": {"nodes": [], "edges": []},
        }

        if self.thread_id:
            chat_data["thread_id"] = self.thread_id
        if self.session_id:
            chat_data["session_id"] = self.session_id

        response = await self.client.post(
            f"{self.base_url}/api/nexus/{self.workflow_id}/chat",
            json=chat_data,
            headers=self.headers,
            timeout=120.0
        )

        if response.status_code != 200:
            raise NexusAPIError(f"Chat request failed: {response.status_code} - {response.text}")

        result = response.json()
        self.thread_id = result.get("thread_id")
        self.session_id = result.get("session_id")
        return result

    async def accept_proposal(self, proposal_id: int) -> dict[str, Any]:
        """Accept a workflow proposal."""
        response = await self.client.post(
            f"{self.base_url}/api/nexus/{self.workflow_id}/proposals/{proposal_id}/accept",
            headers=self.headers,
            timeout=30.0
        )

        if response.status_code != 200:
            raise NexusAPIError(f"Failed to accept proposal: {response.status_code} - {response.text}")

        return response.json()

    async def reject_proposal(self, proposal_id: int) -> dict[str, Any]:
        """Reject a workflow proposal."""
        response = await self.client.post(
            f"{self.base_url}/api/nexus/{self.workflow_id}/proposals/{proposal_id}/reject",
            headers=self.headers,
            timeout=30.0
        )

        if response.status_code != 200:
            raise NexusAPIError(f"Failed to reject proposal: {response.status_code} - {response.text}")

        return response.json()


def create_test_token(user_id: str = "test-user-nexus", email: str = "test@nexus.local") -> str:
    """Create a test JWT token for local authentication."""
    payload = {
        "sub": user_id,
        "email": email,
        "first_name": "Test",
        "last_name": "User",
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")


async def create_test_workflow(client: httpx.AsyncClient, base_url: str, token: str) -> str:
    """Create a test workflow for Nexus chat."""
    headers = {"Authorization": f"Bearer {token}"}

    workflow_data = {
        "name": "Nexus Test Workflow",
        "description": "Test workflow for Nexus interactive testing",
        "tags": ["test", "nexus"],
        "spec": {"nodes": [], "edges": []}
    }

    response = await client.post(
        f"{base_url}/api/v1/workflows",
        json=workflow_data,
        headers=headers,
        timeout=30.0
    )

    if response.status_code != 201:
        raise NexusAPIError(f"Failed to create workflow: {response.status_code} - {response.text}")

    result = response.json()
    workflow_id = result["workflow_id"]
    print(f"{Colors.GREEN}✓ Created test workflow: {workflow_id}{Colors.ENDC}")
    return workflow_id


def display_thinking(thinking_steps: list[str]) -> None:
    """Display agent thinking steps."""
    if not thinking_steps:
        return
    print(f"\n{Colors.CYAN}[Thinking]{Colors.ENDC}")
    for step in thinking_steps:
        print(f"{Colors.CYAN}  • {step}{Colors.ENDC}")
    print()


def get_user_input() -> str:
    """Get user input from terminal."""
    return input(f"{Colors.BOLD}{Colors.BLUE}You > {Colors.ENDC}").strip()


async def handle_proposal_acceptance(result: dict) -> None:
    """Handle successful proposal acceptance."""
    print(f"{Colors.GREEN}✓ Proposal accepted!{Colors.ENDC}")

    workflow_graph = result.get("workflow_graph")
    if workflow_graph:
        nodes = workflow_graph.get("nodes", [])
        print(f"{Colors.GREEN}Updated workflow has {len(nodes)} nodes{Colors.ENDC}")


async def process_user_message(session: ChatSession, user_input: str, handle_proposal_fn) -> None:
    """Process a single user message and display response."""
    print(f"{Colors.CYAN}Sending message...{Colors.ENDC}")
    response = await session.send_message(user_input)

    # Display thinking steps
    thinking = response.get("thinking")
    if thinking:
        display_thinking(thinking)

    # Display agent response
    agent_response = response.get("response", "")
    print(f"{Colors.BOLD}{Colors.GREEN}Nexus:{Colors.ENDC} {agent_response}")

    # Handle proposal
    proposal = response.get("proposal")
    if proposal:
        await handle_proposal_fn(session, proposal)

    # Show proposal error if any
    proposal_error = response.get("proposal_error")
    if proposal_error:
        print(f"{Colors.BOLD}{Colors.RED}Error:{Colors.ENDC} Proposal validation error: {proposal_error}")

    print()  # Blank line for readability
