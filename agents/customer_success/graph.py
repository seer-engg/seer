"""Customer Success LangGraph - Simplified using BaseAgent"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.base_agent import BaseAgent
from shared.agent_tools import acknowledge_user, get_test_cases
from shared.prompts import CUSTOMER_SUCCESS_PROMPT


class CustomerSuccessAgent(BaseAgent):
    """Customer Success agent for Seer"""
    
    def __init__(self):
        super().__init__(
            agent_name="customer_success",
            system_prompt=CUSTOMER_SUCCESS_PROMPT,
            tools=[acknowledge_user, get_test_cases]
        )
    
    def get_capabilities(self):
        return ["user_interaction", "message_relay"]


# Create agent instance
agent = CustomerSuccessAgent()

# Create the graph instance for langgraph dev
graph = agent.build_graph()

# Registration function for backward compatibility
async def register_with_orchestrator():
    await agent.register_with_orchestrator()

