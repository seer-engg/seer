import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from seer.shared.error_handling import create_error_response
from seer.shared.a2a_utils import send_a2a_message
from seer.shared.config import get_config
from seer.shared.llm import get_llm

# Import orchestrator modules
from seer.agents.orchestrator.data_manager import DataManager


class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


      
@tool
async def delegate_to_eval_agent(request_data: str) -> str:
    """
    Delegate an evaluation request to the eval_agent.
    Use this when user wants to evaluate their AI agent.
    
    Args:
        request_data: JSON string with agent_url, agent_id, and expectations
    """
    try:
        data = json.loads(request_data)
        agent_url = data.get("agent_url")
        agent_id = data.get("agent_id")
        expectations = data.get("expectations")
        thread_id = data.get("thread_id", "")
        
        if not all([agent_url, agent_id, expectations]):
            return create_error_response("Missing required fields: agent_url, agent_id, expectations")
        
        # Construct message for eval agent
        message = (
            f"Please evaluate the agent at {agent_url} (ID: {agent_id}).\n"
            f"User expectations: {expectations}"
        )
        
        # Send to eval agent
        response = await send_a2a_message(
            target_agent_id="eval_agent",
            target_port=8002,
            message=message,
            thread_id=thread_id
        )
        
        return response
        
    except Exception as e:
        return create_error_response(f"Failed to delegate to eval agent: {str(e)}", e)

@tool
async def delegate_to_coding_agent(request_data: str) -> str:
    """
    Delegate a code review request to the coding_agent.
    Use this when user wants code analysis or review.
    
    Args:
        request_data: JSON string with repo_url, repo_id, and optional test_results
    """
    try:
        data = json.loads(request_data)
        repo_url = data.get("repo_url")
        repo_id = data.get("repo_id")
        thread_id = data.get("thread_id", "")
        test_results = data.get("test_results")
        
        if not all([repo_url, repo_id]):
            return create_error_response("Missing required fields: repo_url, repo_id")
        
        # Construct message for coding agent
        message_dict = {
            "repo_url": repo_url,
            "repo_id": repo_id
        }
        if test_results:
            message_dict["test_results"] = test_results
        
        message = json.dumps(message_dict)
        
        # Send to coding agent
        response = await send_a2a_message(
            target_agent_id="coding_agent",
            target_port=8003,
            message=message,
            thread_id=thread_id
        )
        
        return response
        
    except Exception as e:
        return create_error_response(f"Failed to delegate to coding agent: {str(e)}", e)


# Orchestrator Agent - Now conversational with routing capabilities
ORCHESTRATOR_PROMPT = """You are the Orchestrator Agent - a conversational AI that helps users evaluate and improve their AI agents.

YOUR ROLE:
1. **Conversational Interface**: Engage directly with users in a warm, professional, and helpful manner
2. **Intent Detection**: Understand what users want to do (evaluate agent, review code, query data)
3. **Agent Coordination**: Delegate to specialized agents (eval_agent, coding_agent) when needed
4. **Data Management**: Handle all database operations for storing and retrieving data
5. **Response Relay**: Acknowledge requests quickly and relay responses from other agents to users

CAPABILITIES:
- Generate and run evaluation tests for AI agents
- Analyze code repositories
- Store and retrieve eval suites, test results, and conversation history
- Coordinate with specialized agents through A2A protocol

WORKFLOW:
1. Parse user input to understand their intent
2. For evaluation requests: acknowledge and delegate to eval_agent
3. For code review requests: acknowledge and delegate to coding_agent
4. For data queries: use your tools to fetch and return data
5. For general questions: respond conversationally

COMMUNICATION STYLE:
- Be warm, professional, and encouraging
- Acknowledge requests quickly ("Got it! I'll evaluate your agent...")
- Keep users informed of progress
- Use emojis sparingly for visual clarity (✅, 📊, 🔍)

FIRST-TURN RULE:
On the first user message about any task, ALWAYS reply with a brief acknowledgment and a clear next step. Do NOT call tools in this turn; wait for explicit user confirmation (e.g., "run tests", "go ahead") or a follow-up message before delegating to other agents.

You are the main interface for users - make their experience smooth and delightful!"""


class OrchestratorAgent:
    """Conversational Orchestrator agent with A2A routing"""
    
    def __init__(self):
        # Initialize modules
        self.data_manager = DataManager()
        
        # Seed registry from deployment-config
        self._seed_registry_from_config()

    def _seed_registry_from_config(self):
        """Seed agent registry from deployment config"""
        try:
            cfg = get_config()
            for name in cfg.list_agents():
                info = cfg.get_agent_config(name)
                port = info.get("port")
                graph_name = info.get("graph_name")
                if port and graph_name:
                    # Store in DB + memory (use graph_name as stable identifier)
                    self.data_manager.register_agent(name, port, graph_name, [])
        except Exception:
            pass
    
    def build_graph(self):
        """Build the orchestrator graph with conversational pattern"""
        llm = get_llm()

        def orchestrator_node(state: OrchestratorState):
            """Main conversational orchestrator node"""
            messages = [SystemMessage(content=ORCHESTRATOR_PROMPT)] + state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: OrchestratorState):
            """Check if we should continue to tools or end"""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        # Create graph
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("orchestrator", orchestrator_node)
        workflow.add_node("tools", ToolNode([delegate_to_eval_agent, delegate_to_coding_agent]))

        workflow.set_entry_point("orchestrator")
        workflow.add_conditional_edges("orchestrator", should_continue)
        workflow.add_edge("tools", "orchestrator")

        return workflow.compile()

# Create agent instance
agent = OrchestratorAgent()

# Create the graph instance for langgraph dev
graph = agent.build_graph()
