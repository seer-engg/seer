"""Conversational Orchestrator Agent with A2A Routing"""

import os
import json
from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from datetime import datetime

from seer.shared.base_agent import BaseAgent, BaseAgentState
from seer.shared.error_handling import create_error_response
from seer.shared.a2a_utils import send_a2a_message
from seer.shared.config import get_config

# Import orchestrator modules
from seer.agents.orchestrator.data_manager import DataManager
from seer.agents.orchestrator.agent_registry import AgentRegistry


class OrchestratorState(BaseAgentState):
    """State for Orchestrator agent"""
    registered_agents: Dict[str, Dict[str, Any]] = {}


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


class OrchestratorAgent(BaseAgent):
    """Conversational Orchestrator agent with A2A routing"""
    
    def __init__(self):
        super().__init__(
            agent_name="orchestrator",
            system_prompt=ORCHESTRATOR_PROMPT,
            tools=self._create_orchestrator_tools()
        )
        
        # Initialize modules
        self.data_manager = DataManager()
        self.agent_registry = AgentRegistry(self.data_manager)
        
        # Seed registry from deployment-config
        self._seed_registry_from_config()

    def _seed_registry_from_config(self):
        """Seed agent registry from deployment config"""
        try:
            cfg = get_config()
            for name in cfg.list_agents():
                info = cfg.get_agent_config(name)
                port = info.get("port")
                assistant_id = info.get("assistant_id")
                if port and assistant_id:
                    # Store in DB + memory
                    self.data_manager.register_agent(name, port, assistant_id, [])
                    self.agent_registry.registered_agents[name] = {
                        "port": port,
                        "assistant_id": assistant_id,
                        "capabilities": [],
                        "registered_at": datetime.now().isoformat()
                    }
        except Exception:
            pass
    
    def get_capabilities(self):
        return ["conversation", "evaluation", "code_review", "data_management", "agent_coordination"]
    
    def _create_orchestrator_tools(self):
        """Create orchestrator-specific tools"""
        from langchain_core.tools import tool
        
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
                
                # Get eval_agent info
                eval_agent_info = self.agent_registry.registered_agents.get("eval_agent")
                if not eval_agent_info:
                    return create_error_response("Eval agent not registered")
                
                # Construct message for eval agent
                message = (
                    f"Please evaluate the agent at {agent_url} (ID: {agent_id}).\n"
                    f"User expectations: {expectations}"
                )
                
                # Send to eval agent
                response = await send_a2a_message(
                    target_agent_id=eval_agent_info["assistant_id"],
                    target_port=eval_agent_info["port"],
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
                
                # Get coding_agent info
                coding_agent_info = self.agent_registry.registered_agents.get("coding_agent")
                if not coding_agent_info:
                    return create_error_response("Coding agent not registered")
                
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
                    target_agent_id=coding_agent_info["assistant_id"],
                    target_port=coding_agent_info["port"],
                    message=message,
                    thread_id=thread_id
                )
                
                return response
                
            except Exception as e:
                return create_error_response(f"Failed to delegate to coding agent: {str(e)}", e)

        return [
            delegate_to_eval_agent,
            delegate_to_coding_agent
        ]

    def build_graph(self):
        """Build the orchestrator graph with conversational pattern"""
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        ).bind_tools(self.tools)

        def _get_thread_id_from_config(state: OrchestratorState) -> Optional[str]:
            """Extract thread_id from LangGraph config"""
            try:
                # LangGraph passes thread_id in configurable
                import langgraph
                from langgraph.checkpoint.base import BaseCheckpointSaver
                
                # Try to get from pregel config (this is set by LangGraph runtime)
                config = getattr(state, "config", None) or {}
                if isinstance(config, dict):
                    configurable = config.get("configurable", {})
                    if isinstance(configurable, dict):
                        tid = configurable.get("thread_id")
                        if tid:
                            return str(tid)
            except Exception:
                pass
            return None

        def persist_message_node(state: OrchestratorState):
            """Persist inbound user messages to database - SKIP for now, rely on LangGraph state"""
            # Note: We skip manual persistence here because LangGraph handles state
            # Messages are retrieved via get_conversation_history from data service
            return {}

        def orchestrator_node(state: OrchestratorState):
            """Main conversational orchestrator node"""
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
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
        workflow.add_node("persist_message", persist_message_node)
        workflow.add_node("orchestrator", orchestrator_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.set_entry_point("persist_message")
        workflow.add_edge("persist_message", "orchestrator")
        workflow.add_conditional_edges("orchestrator", should_continue)
        workflow.add_edge("tools", "orchestrator")

        return workflow.compile()

    async def register_orchestrator(self):
        """Register the orchestrator itself"""
        try:
            from shared.config import get_assistant_id, get_port
            
            assistant_id = get_assistant_id("orchestrator")
            port = get_port("orchestrator")

            # Self-register
            self.agent_registry.register_agent(
                "orchestrator",
                port,
                assistant_id,
                ["conversation", "evaluation", "code_review", "data_management", "agent_coordination"]
            )
            
            print(f"✅ Orchestrator registered itself: {assistant_id}")
        except Exception as e:
            print(f"❌ Error registering orchestrator: {e}")


# Create agent instance
agent = OrchestratorAgent()

# Create the graph instance for langgraph dev
graph = agent.build_graph()

# Registration function for backward compatibility
async def register_orchestrator():
    await agent.register_orchestrator()
