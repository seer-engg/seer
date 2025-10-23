import json
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_core.tools import InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langgraph.types import Command

from shared.error_handling import create_error_response
from shared.messaging import messenger
from shared.llm import get_llm
from shared.logger import get_logger
from shared.state_models import (
    EvalSuite,
    TestResult,
    TargetAgentExpectation,
    TargetAgentConfig,
    RemoteThreadLink,
    AgentActivity
)

# Get logger for orchestrator
logger = get_logger('orchestrator')


class OrchestratorState(TypedDict, total=False):
    """
    Orchestrator state with all data stored in LangGraph's persistence layer.
    No external database needed - everything is in state!
    """
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Remote thread tracking
    eval_agent_thread_id: str  # Remote thread id for eval_agent
    
    # Target agent information
    target_expectations: Optional[TargetAgentExpectation]
    target_config: Optional[TargetAgentConfig]
    
    # Evaluation data
    eval_suites: List[EvalSuite]  # All eval suites for this thread
    test_results: List[TestResult]  # All test results for this thread
    
    # Remote thread links for inter-agent communication
    remote_thread_links: List[RemoteThreadLink]
    
    # Agent activity logs (optional, for debugging)
    agent_activities: List[AgentActivity]


@tool
def get_eval_thread_id(runtime: ToolRuntime) -> str:
    """Get the persistent eval_agent remote thread id from orchestrator state."""
    return runtime.state.get("eval_agent_thread_id", "") or ""

@tool
def set_eval_thread_id(thread_id: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Set/update the eval_agent remote thread id in orchestrator state."""
    return Command(update={
        "eval_agent_thread_id": thread_id,
        "messages": [ToolMessage(content=f"Set eval thread id to {thread_id}", tool_call_id=tool_call_id)]
    })

@tool
def think(thought: str, config: RunnableConfig) -> str:
    """
    Think tool: logs an internal thought for this thread and returns an echo.
    No external side effects beyond logging. Use before taking actions or after tool results.
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        logger.info(f"THINK[{thread_id}]: {thought}")
        return json.dumps({"success": True, "thought": thought})
    except Exception as e:
        return create_error_response(f"Failed to log thought: {str(e)}", e)


@tool
def save_target_expectations(
    expectations: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Save target agent expectations collected from user.
    Expectations should be a comma-separated or newline-separated list of user expectations.
    
    Args:
        expectations: String containing user's expectations (will be split into list)
    """
    try:
        # Split by commas or newlines
        expectations_list = [e.strip() for e in expectations.replace('\n', ',').split(',') if e.strip()]
        
        logger.info(f"Saving {len(expectations_list)} expectations to state")
        
        expectation_obj = TargetAgentExpectation(
            expectations=expectations_list
        )
        
        return Command(update={
            "target_expectations": expectation_obj.model_dump(),
            "messages": [ToolMessage(
                content=json.dumps({
                    "success": True,
                    "message": f"Saved {len(expectations_list)} expectations",
                    "expectations": expectations_list
                }),
                tool_call_id=tool_call_id
            )]
        })
    except Exception as e:
        logger.error(f"Failed to save expectations: {str(e)}")
        return Command(update={
            "messages": [ToolMessage(
                content=create_error_response(f"Failed to save expectations: {str(e)}", e),
                tool_call_id=tool_call_id
            )]
        })


@tool
def save_target_config(
    config_data: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Save target agent configuration collected from user.
    
    Args:
        config_data: JSON string with keys: target_agent_port, target_agent_url, 
                    target_agent_github_url, target_agent_assistant_id
    """
    try:
        agent_config = json.loads(config_data)
        
        config_obj = TargetAgentConfig(
            target_agent_port=agent_config.get("target_agent_port"),
            target_agent_url=agent_config.get("target_agent_url"),
            target_agent_github_url=agent_config.get("target_agent_github_url"),
            target_agent_assistant_id=agent_config.get("target_agent_assistant_id")
        )
        
        return Command(update={
            "target_config": config_obj.model_dump(),
            "messages": [ToolMessage(
                content=json.dumps({
                    "success": True,
                    "message": "Saved agent configuration",
                    "config": config_obj.model_dump()
                }),
                tool_call_id=tool_call_id
            )]
        })
    except Exception as e:
        return Command(update={
            "messages": [ToolMessage(
                content=create_error_response(f"Failed to save config: {str(e)}", e),
                tool_call_id=tool_call_id
            )]
        })


@tool
def check_delegation_readiness(runtime: ToolRuntime) -> str:
    """
    Check if we have collected both expectations and config for delegation.
    Returns readiness status and what's missing if not ready.
    """
    try:
        state = runtime.state
        expectations = state.get("target_expectations")
        config = state.get("target_config")
        
        ready = expectations is not None and config is not None
        
        return json.dumps({
            "success": True,
            "ready": ready,
            "has_expectations": expectations is not None,
            "has_config": config is not None,
            "message": "Ready to delegate!" if ready else "Missing data - need to collect more info",
            "expectations": expectations,
            "config": config
        })
    except Exception as e:
        return create_error_response(f"Failed to check readiness: {str(e)}", e)

      
@tool
async def message_agent(to_agent: str, message: str, config: RunnableConfig, runtime: ToolRuntime) -> str:
    """
    Generic proxy: send any message to a target agent (eval_agent, coding_agent)
    within the SAME user thread (persistent remote thread).
    
    If target_config exists in state, it will be passed as a state update to the target agent.
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        ports = {
            "eval_agent": 8002,
            "coding_agent": 8003,
        }
        port = ports.get(to_agent)
        assert port is not None, f"Invalid agent name: {to_agent}"
        base_url = f"http://127.0.0.1:{port}"
        
        # If targeting eval_agent, try to reuse stored remote thread id
        forced_remote_tid = None
        if to_agent == "eval_agent":
            forced_remote_tid = runtime.state.get("eval_agent_thread_id")

        # Extract target_config from state and pass as state update
        target_config_dict = runtime.state.get("target_config")
        state_update = None
        if target_config_dict:
            # Map orchestrator's TargetAgentConfig to eval_agent's TargetConfigDTO format
            # This will be merged into the eval_agent's state directly
            target_dto = {
                "url": target_config_dict.get("target_agent_url"),
                "port": target_config_dict.get("target_agent_port"),
                "assistant_id": target_config_dict.get("target_agent_assistant_id"),
                "github_url": target_config_dict.get("target_agent_github_url"),
            }
            # For eval_agent, nest it in current_eval.target
            state_update = {
                "current_eval": {
                    "target": target_dto
                }
            }

        text, remote_tid = await messenger.send(
            user_thread_id=thread_id,
            src_agent="orchestrator",
            dst_agent=to_agent,
            base_url=base_url,
            assistant_id=to_agent,
            content=message,
            remote_thread_id=forced_remote_tid,
            state_update=state_update,
        )
        return json.dumps({"success": True, "response": text, "remote_thread_id": remote_tid})
    except Exception as e:
        return create_error_response(f"Failed to message {to_agent}: {str(e)}", e)


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
   - If the response returns a `remote_thread_id`, immediately call `set_eval_thread_id(thread_id=...)` to persist it in state.
   - For subsequent eval_agent calls (listing/running tests), always reuse the same eval thread by storing and passing it via state using `get_eval_thread_id` and `set_eval_thread_id`.

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
    """Conversational Orchestrator agent - all data stored in state"""
    
    def build_graph(self):
        """Build the orchestrator agent using create_agent runtime"""
        model = get_llm()
        tools = [
            save_target_expectations,
            save_target_config,
            check_delegation_readiness,
            message_agent,
            think,
            get_eval_thread_id,
            set_eval_thread_id,
        ]
        return create_agent(
            model=model,
            tools=tools,
            system_prompt=ORCHESTRATOR_PROMPT,
            state_schema=OrchestratorState,
        )

# Create agent instance
agent = OrchestratorAgent()

# Create the graph instance for langgraph dev
graph = agent.build_graph()
