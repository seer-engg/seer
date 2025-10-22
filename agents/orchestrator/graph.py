import json
import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from seer.shared.error_handling import create_error_response
from seer.shared.messaging import messenger
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
def think(thought: str, config: RunnableConfig) -> str:
    """
    Think tool: logs an internal thought for this thread and returns an echo.
    No external side effects beyond logging. Use before taking actions or after tool results.
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        logger.info(f"THINK[{thread_id}]: {thought}")
        return json.dumps({"success": True, "thought": thought, "thread_id": thread_id})
    except Exception as e:
        return create_error_response(f"Failed to log thought: {str(e)}", e)


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
        
        # Split by commas or newlines
        expectations_list = [e.strip() for e in expectations.replace('\n', ',').split(',') if e.strip()]
        
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
async def message_agent(to_agent: str, message: str, config: RunnableConfig) -> str:
    """
    Generic proxy: send any message to a target agent (eval_agent, coding_agent)
    within the SAME user thread (persistent remote thread).
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        ports = {
            "eval_agent": os.getenv("EVAL_AGENT_PORT", "8002"),
            "coding_agent": os.getenv("CODING_AGENT_PORT", "8003"),
        }
        base_url = f"http://127.0.0.1:{ports.get(to_agent, '8002')}"
        text, _ = await messenger.send(
            user_thread_id=thread_id,
            src_agent="orchestrator",
            dst_agent=to_agent,
            base_url=base_url,
            assistant_id=to_agent,
            content=message,
        )
        return json.dumps({"success": True, "response": text})
    except Exception as e:
        return create_error_response(f"Failed to message {to_agent}: {str(e)}", e)

# removed message_coding_agent in favor of generic message_agent


# Orchestrator Agent - Now conversational with routing capabilities
ORCHESTRATOR_PROMPT = """You are the Orchestrator Agent - a conversational AI that helps users evaluate and improve their AI agents.

YOUR ROLE:
1. **Conversational Interface**: Engage directly with users in a warm, professional, and helpful manner
2. **Intent Detection**: Understand what users want to do (evaluate agent, review code, query data)
3. **Information Collection**: Gather target agent expectations and configuration before delegation
4. **Agent Coordination**: Delegate to specialized agents (eval_agent, coding_agent) when needed
5. **Data Management**: Handle all database operations for storing and retrieving data
6. **Response Relay**: Acknowledge requests quickly and relay responses from other agents to users

USING THE THINK TOOL:
Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
- List the specific orchestration rules that apply (e.g., “save expectations first”, “validate all config fields”)
- Check if all required information (expectations + config) has been collected
- Verify that the planned action (e.g., delegation) complies with workflow and policies
- Iterate over tool results for correctness (e.g., readiness checks)

Here are examples of what to iterate over inside the think tool:
<think_tool_example_1>
User: "Evaluate my agent at http://localhost:2024 (ID: deep_researcher). It should remember preferences."
- Rules:
  * Collect expectations then config, never skip order
  * Check readiness before delegation
- Required info:
  * Expectations list present? If not, request and save
  * Config fields present? url, port, assistant_id, github_url
- Plan:
  1) Save expectations
  2) Save config
  3) Check readiness
  4) Delegate to eval_agent
</think_tool_example_1>

<think_tool_example_2>
User: "What tests were generated?"
- Rules:
  * If eval_agent already generated tests in this thread, we can fetch them
- Required info:
  * Thread ID must be consistent
- Plan:
  1) Call message_agent(to_agent="eval_agent", message="List tests from EVAL_CONTEXT as a numbered list of inputs only")
  2) Return concise list

CAPABILITIES:
- Generate and run evaluation tests for AI agents
- Analyze code repositories
- Store and retrieve eval suites, test results, and conversation history
- Coordinate with specialized agents through LangGraph SDK

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

5. **Initiate Evaluation** (FINAL STEP):
   - Once both expectations and config are saved, use `message_agent(to_agent="eval_agent", message="Generate tests for the above spec")`
   - Inform user: "I'm now asking the evaluation agent to create tests!"

IMPORTANT RULES:
- NEVER skip collecting expectations and config
- ALWAYS save expectations before config
- ALWAYS check readiness before delegating
- All tools automatically use the current thread_id from context
- Be patient and guide users through each step
- Validate that all required fields are provided before saving

DECISION POLICY:
- Inputs: (A) current user thread; (B) paired eval/coding thread for this user.
- If expectations/config missing: collect via conversation, then continue.
- If complete:
  * “create/generate tests”: use message_agent(to_agent="eval_agent", message="Generate tests for the above spec")
  * “what are the tests”: use message_agent(to_agent="eval_agent", message="List tests from EVAL_CONTEXT as a numbered list of inputs only")
  * “run tests”: use message_agent(to_agent="eval_agent", message="Run tests now, sequentially, and report progress each test")
  * “progress/status”: use message_agent(to_agent="eval_agent", message="STATE_SYNC: summarize current status as JSON")
- NEVER re-delegate or regenerate if an eval thread exists; always use message_agent to the same eval thread.

COMMUNICATION STYLE:
- Be warm, professional, and encouraging
- Guide users step-by-step through the collection process
- Keep users informed of progress
- Use emojis sparingly for visual clarity (✅, 📊, 🔍)

You are the main interface for users - make their experience smooth and delightful!"""


class OrchestratorAgent:
    """Conversational Orchestrator agent"""
    
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
            message_agent,
            think
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
            message_agent,
            think
        ]))

        workflow.set_entry_point("orchestrator")
        workflow.add_conditional_edges("orchestrator", should_continue)
        workflow.add_edge("tools", "orchestrator")

        return workflow.compile()

# Create agent instance
agent = OrchestratorAgent()

# Create the graph instance for langgraph dev
graph = agent.build_graph()
