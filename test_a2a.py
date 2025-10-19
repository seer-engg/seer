import asyncio
import aiohttp
from shared.a2a_utils import send_a2a_message


async def send_message(session, port, assistant_id, text):
    """Send a message to an agent and return the response text."""
    try:
        response = await send_a2a_message(assistant_id, port, text)
        data = response.json() if isinstance(response, str) else response
        if data.get("success"):
            return data.get("response", "No response")
        else:
            return f"Error: {data.get('error', 'Unknown error')}"
    except Exception as e:
        print(f"Response error from port {port}: {e}")
        return f"Error from port {port}: {str(e)}"


async def simulate_conversation():
    """Simulate a conversation between two agents using configuration-based UUIDs."""
    # Import configuration
    from shared.config import get_assistant_id
    
    # Use the UUIDs from configuration
    agent_a_id = get_assistant_id("orchestrator")
    agent_b_id = get_assistant_id("customer_success")
    message = "Hello! Let's have a conversation."
    
    for i in range(3):
        print(f"--- Round {i + 1} ---")
        # Agent A responds (orchestrator)
        message = await send_message(None, 8000, agent_a_id, message)
        print(f"🔵 Orchestrator: {message}")
        # Agent B responds (customer success)
        message = await send_message(None, 8001, agent_b_id, message)
        print(f"🔴 Customer Success: {message}")
        print()

if __name__ == "__main__":
    asyncio.run(simulate_conversation())