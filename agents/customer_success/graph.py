"""Customer Success LangGraph - Simplified using BaseAgent"""

from seer.shared.base_agent import BaseAgent
from seer.shared.agent_tools import acknowledge_user, get_test_cases


# Customer Success Agent
CUSTOMER_SUCCESS_PROMPT = """You are a Customer Success agent for Seer, an AI agent evaluation platform.

CORE WORKFLOW:
1. ALWAYS call think_agent_tool() first for any input
2. If ACT: Handle user requests, confirmations, and relay messages
3. If IGNORE: Skip non-relevant messages

YOUR ROLE:
- Help users evaluate their AI agents
- Handle user confirmations and relay them to evaluation team
- Answer questions about test cases
- Relay messages from other agents to users

TOOLS: think_agent_tool, acknowledge_user, send_to_orchestrator_tool, get_test_cases

COMMUNICATION:
- All messages go through orchestrator via send_to_orchestrator_tool(action, payload, thread_id)
- Use action types: "user_confirmed", "eval_question", "eval_results"
- Be warm, professional, and helpful"""

class CustomerSuccessAgent(BaseAgent):
    """Customer Success agent for Seer"""
    
    def __init__(self):
        super().__init__(
            agent_name="customer_success",
            system_prompt=CUSTOMER_SUCCESS_PROMPT,
            tools=[acknowledge_user, get_test_cases]
        )
    
# Create agent instance
agent = CustomerSuccessAgent()

# Create the graph instance for langgraph dev
graph = agent.build_graph()
