import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from seer.shared.error_handling import create_error_response
from seer.shared.a2a_utils import send_a2a_message
from seer.shared.config import get_config
from seer.shared.llm import get_llm
from seer.shared.logger import get_logger

# Import orchestrator modules
from seer.agents.orchestrator.data_manager import DataManager

# Get logger for orchestrator
logger = get_logger('orchestrator')


class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str  # Track current thread ID for data operations


# Global data manager instance for tools
_data_manager = DataManager()


@tool
def save_target_expectations(
    expectations: str, 
    config: RunnableConfig
) -> str:
    """
    Save target agent expectations collected from user.
    Expectations should be a comma-separated or newline-separated list of user expectations.
    
    Args:
        expectations: String containing user's expectations (will be split into list)
    """
    try:
        # Extract thread_id from LangGraph config
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        
        # Parse expectations into list
        if isinstance(expectations, str):
            # Split by commas or newlines
            expectations_list = [e.strip() for e in expectations.replace('\n', ',').split(',') if e.strip()]
        else:
            expectations_list = expectations
        
        logger.info(f"Saving {len(expectations_list)} expectations for thread {thread_id}")
        result = _data_manager.save_target_agent_expectation(thread_id, expectations_list)
        return json.dumps({
            "success": True,
            "message": f"Saved {len(expectations_list)} expectations for thread {thread_id}",
            "data": result
        })
    except Exception as e:
        logger.error(f"Failed to save expectations: {str(e)}")
        return create_error_response(f"Failed to save expectations: {str(e)}", e)


@tool
def save_target_config(
    config_data: str, 
    config: RunnableConfig
) -> str:
    """
    Save target agent configuration collected from user.
    
    Args:
        config_data: JSON string with keys: target_agent_port, target_agent_url, 
                    target_agent_github_url, target_agent_assistant_id
    """
    try:
        # Extract thread_id from LangGraph config
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        
        agent_config = json.loads(config_data)
        result = _data_manager.save_target_agent_config(
            thread_id=thread_id,
            target_agent_port=agent_config.get("target_agent_port"),
            target_agent_url=agent_config.get("target_agent_url"),
            target_agent_github_url=agent_config.get("target_agent_github_url"),
            target_agent_assistant_id=agent_config.get("target_agent_assistant_id")
        )
        return json.dumps({
            "success": True,
            "message": f"Saved agent configuration for thread {thread_id}",
            "data": result
        })
    except Exception as e:
        return create_error_response(f"Failed to save config: {str(e)}", e)


@tool
def check_delegation_readiness(config: RunnableConfig) -> str:
    """
    Check if we have collected both expectations and config for delegation.
    Returns readiness status and what's missing if not ready.
    Call this tool with no arguments - thread_id is automatically extracted.
    """
    try:
        # Extract thread_id from LangGraph config
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        
        result = _data_manager.check_readiness_for_delegation(thread_id)
        return json.dumps({
            "success": True,
            "ready": result["ready"],
            "has_expectations": result["has_expectations"],
            "has_config": result["has_config"],
            "message": "Ready to delegate!" if result["ready"] else "Missing data - need to collect more info",
            "data": result
        })
    except Exception as e:
        return create_error_response(f"Failed to check readiness: {str(e)}", e)


      
@tool
async def delegate_to_eval_agent(config: RunnableConfig) -> str:
    """
    Delegate an evaluation request to the eval_agent.
    Use this ONLY after expectations and config have been saved.
    Automatically retrieves saved expectations and config from database.
    Call this tool with no arguments - thread_id is automatically extracted.
    """
    try:
        # Extract thread_id from LangGraph config
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        
        logger.info(f"Delegating to eval agent for thread {thread_id}")
        
        # Check readiness first
        readiness = _data_manager.check_readiness_for_delegation(thread_id)
        
        if not readiness["ready"]:
            logger.warning(f"Cannot delegate - missing data: expectations={readiness['has_expectations']}, config={readiness['has_config']}")
            return json.dumps({
                "success": False,
                "error": "Cannot delegate yet - missing required data",
                "has_expectations": readiness["has_expectations"],
                "has_config": readiness["has_config"],
                "message": "Please collect expectations and config first using save_target_expectations and save_target_config tools"
            })
        
        # Get saved data
        expectations_data = readiness["expectations"]
        agent_config = readiness["config"]
        
        # Construct message for eval agent
        message = (
            f"Please evaluate the agent with the following details:\n"
            f"- URL: {agent_config['target_agent_url']}\n"
            f"- Port: {agent_config['target_agent_port']}\n"
            f"- Assistant ID: {agent_config['target_agent_assistant_id']}\n"
            f"- GitHub URL: {agent_config['target_agent_github_url']}\n\n"
            f"User expectations:\n" + 
            "\n".join(f"  {i+1}. {exp}" for i, exp in enumerate(expectations_data))
        )
        
        # Send to eval agent
        logger.info(f"Sending evaluation request to eval_agent on port 8002")
        response = await send_a2a_message(
            target_agent_id="eval_agent",
            target_port=8002,
            message=message,
            thread_id=thread_id
        )
        
        logger.info("Successfully delegated to eval agent")
        return response
        
    except Exception as e:
        logger.error(f"Failed to delegate to eval agent: {str(e)}")
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
3. **Information Collection**: Gather target agent expectations and configuration before delegation
4. **Agent Coordination**: Delegate to specialized agents (eval_agent, coding_agent) when needed
5. **Data Management**: Handle all database operations for storing and retrieving data
6. **Response Relay**: Acknowledge requests quickly and relay responses from other agents to users

CAPABILITIES:
- Generate and run evaluation tests for AI agents
- Analyze code repositories
- Store and retrieve eval suites, test results, and conversation history
- Coordinate with specialized agents through A2A protocol

WORKFLOW FOR EVALUATION REQUESTS (CRITICAL):
When a user wants to evaluate their agent, you MUST follow this sequence:

1. **Initial Acknowledgment**: Greet the user and express readiness to help

2. **Collect Expectations** (REQUIRED FIRST):
   - Ask: "What are your expectations for this agent? Please list the key behaviors or capabilities you want to test"
   - Wait for user response with their expectations
   - Use `save_target_expectations(expectations="user's expectations here")` to save them
   - The tool automatically uses the current thread_id
   - Confirm: "Got it! I've saved your expectations."

3. **Collect Configuration** (REQUIRED SECOND):
   - Ask for each piece of information:
     * Target agent URL (where the agent is deployed)
     * Target agent port (e.g., 8001, 8002)
     * Target agent GitHub URL (source code repository)
     * Target agent assistant ID (LangGraph assistant/graph ID)
   - Once collected, format as JSON and use:
     `save_target_config(config_data='{"target_agent_port": 8001, "target_agent_url": "...", "target_agent_github_url": "...", "target_agent_assistant_id": "..."}')`
   - Confirm: "Perfect! I've saved the agent configuration."

4. **Check Readiness** (OPTIONAL):
   - Use `check_delegation_readiness()` to verify all data is collected
   - If not ready, ask for missing information

5. **Delegate to Eval Agent** (FINAL STEP):
   - Once both expectations and config are saved, use `delegate_to_eval_agent()`
   - Inform user: "I'm now delegating this to the evaluation agent to create and run tests!"

IMPORTANT RULES:
- NEVER skip collecting expectations and config
- ALWAYS save expectations before config
- ALWAYS check readiness before delegating
- All tools automatically use the current thread_id from context
- Be patient and guide users through each step
- Validate that all required fields are provided before saving

COMMUNICATION STYLE:
- Be warm, professional, and encouraging
- Guide users step-by-step through the collection process
- Keep users informed of progress
- Use emojis sparingly for visual clarity (✅, 📊, 🔍)

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
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools([
            save_target_expectations,
            save_target_config,
            check_delegation_readiness,
            delegate_to_eval_agent,
            delegate_to_coding_agent
        ])

        def orchestrator_node(state: OrchestratorState):
            """Main conversational orchestrator node"""
            messages = [SystemMessage(content=ORCHESTRATOR_PROMPT)] + state["messages"]
            response = llm_with_tools.invoke(messages)
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
        workflow.add_node("tools", ToolNode([
            save_target_expectations,
            save_target_config,
            check_delegation_readiness,
            delegate_to_eval_agent,
            delegate_to_coding_agent
        ]))

        workflow.set_entry_point("orchestrator")
        workflow.add_conditional_edges("orchestrator", should_continue)
        workflow.add_edge("tools", "orchestrator")

        return workflow.compile()

# Create agent instance
agent = OrchestratorAgent()

# Create the graph instance for langgraph dev
graph = agent.build_graph()
