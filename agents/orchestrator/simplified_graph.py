"""Simplified Orchestrator Agent using modular design"""

import os
import json
import time
import hashlib
import httpx
from typing import Annotated, TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from datetime import datetime

from seer.shared.base_agent import BaseAgent, BaseAgentState
from seer.shared.error_handling import create_error_response, create_success_response
from seer.shared.a2a_utils import send_a2a_message
from seer.shared.config import get_config

# Import orchestrator modules

from seer.agents.orchestrator.modules.data_manager import DataManager
from seer.agents.orchestrator.modules.message_router import MessageRouter
from seer.agents.orchestrator.modules.agent_registry import AgentRegistry


class OrchestratorState(BaseAgentState):
    """State for Orchestrator agent"""
    registered_agents: Dict[str, Dict[str, Any]] = {}
    active_conversations: Dict[str, List[Dict[str, Any]]] = {}
    pending_broadcasts: List[Dict[str, Any]] = []
    recent_broadcasts: Dict[str, Dict[str, float]] = {}



# Orchestrator Agent
ORCHESTRATOR_PROMPT = """You are the Orchestrator Agent - the central hub for all Seer agent communication.

YOUR ROLE:
1. **Message Routing**: Route messages between agents based on action type
2. **Agent Registration**: Register new agents in the network
3. **Data Management**: Handle all database operations
4. **State Management**: Maintain shared state across agents

WORKFLOW:
1. Parse input to understand requested action
2. Perform the action (register, route, store data)
3. Route messages to appropriate target agents
4. Respond with operation result

MESSAGE ROUTING:
- "user_confirmed" → eval_agent
- "eval_question" → customer_success
- "eval_results" → customer_success
- "get_eval_suites" → return stored data

You are the central message router - every message flows through you!"""

class OrchestratorAgent(BaseAgent):
    """Simplified Orchestrator agent using modular design"""
    
    def __init__(self):
        super().__init__(
            agent_name="orchestrator",
            system_prompt=ORCHESTRATOR_PROMPT,
            tools=self._create_orchestrator_tools()
        )
        
        # Initialize modules
        self.data_manager = DataManager()
        self.agent_registry = AgentRegistry(self.data_manager)
        self.message_router = MessageRouter(self.data_manager, {})
        
        # Update message router with registry
        self.message_router.update_registered_agents(self.agent_registry.registered_agents)
        # Seed registry from deployment-config
        self._seed_registry_from_config()

    def _seed_registry_from_config(self):
        try:
            cfg = get_config()
            for name in cfg.list_agents():
                info = cfg.get_agent_config(name)
                port = info.get("port")
                assistant_id = info.get("assistant_id")
                if port and assistant_id:
                    # store in DB + memory
                    self.data_manager.register_agent(name, port, assistant_id, [])
                    self.agent_registry.registered_agents[name] = {
                        "port": port,
                        "assistant_id": assistant_id,
                        "capabilities": [],
                        "registered_at": datetime.now().isoformat()
                    }
            self.message_router.update_registered_agents(self.agent_registry.registered_agents)
        except Exception:
            pass
    
    def get_capabilities(self):
        return ["message_routing", "data_management", "agent_coordination"]
    
    def _create_orchestrator_tools(self):
        """Create orchestrator-specific tools"""
        from langchain_core.tools import tool
        
        @tool
        def register_agent_tool(registration: str) -> str:
            """Register a new agent in the network"""
            try:
                data = json.loads(registration)
                agent_name = data.get("agent_name")
                port = data.get("port")
                assistant_id = data.get("assistant_id")
                capabilities = data.get("capabilities", [])

                if not all([agent_name, port, assistant_id]):
                    return create_error_response("Missing required fields: agent_name, port, assistant_id")

                result = self.agent_registry.register_agent(agent_name, port, assistant_id, capabilities)
                
                # Update message router with new agent
                self.message_router.update_registered_agents(self.agent_registry.registered_agents)
                
                return result
            except Exception as e:
                return create_error_response(f"Failed to register agent: {str(e)}", e)

        @tool
        def store_eval_suite_tool(suite_data: str) -> str:
            """Store an evaluation suite"""
            try:
                data = json.loads(suite_data)
                suite_id = self.data_manager.store_eval_suite(data)
                return create_success_response({
                    "suite_id": suite_id,
                    "message": f"Eval suite {suite_id} stored successfully"
                })
            except Exception as e:
                return create_error_response(f"Failed to store eval suite: {str(e)}", e)

        @tool
        def get_eval_suites_tool(query: str) -> str:
            """Get evaluation suites"""
            try:
                data = json.loads(query) if query else {}
                agent_url = data.get("agent_url")
                agent_id = data.get("agent_id")

                suites = self.data_manager.get_eval_suites(agent_url=agent_url, agent_id=agent_id)
                return create_success_response({
                    "suites": suites,
                    "count": len(suites)
                })
            except Exception as e:
                return create_error_response(f"Failed to get eval suites: {str(e)}", e)

        @tool
        def store_test_results_tool(results_data: str) -> str:
            """Store test results"""
            try:
                data = json.loads(results_data)
                suite_id = data.get("suite_id")
                thread_id = data.get("thread_id")
                results = data.get("results", [])

                if not all([suite_id, thread_id, results]):
                    return create_error_response("Missing required fields: suite_id, thread_id, results")

                result = self.data_manager.store_test_results(suite_id, thread_id, results)
                return create_success_response(result)
            except Exception as e:
                return create_error_response(f"Failed to store test results: {str(e)}", e)

        @tool
        def get_conversation_history_tool(thread_id: str) -> str:
            """Get conversation history for a thread"""
            try:
                messages = self.data_manager.get_conversation_history(thread_id)
                return create_success_response({
                    "thread_id": thread_id,
                    "messages": messages,
                    "count": len(messages)
                })
            except Exception as e:
                return create_error_response(f"Failed to get conversation history: {str(e)}", e)

        @tool
        def get_test_results_tool(query: str) -> str:
            """Get test results with optional filters"""
            try:
                data = json.loads(query) if query else {}
                suite_id = data.get("suite_id")
                thread_id = data.get("thread_id")

                results = self.data_manager.get_test_results(suite_id=suite_id, thread_id=thread_id)
                return create_success_response({
                    "results": results,
                    "count": len(results)
                })
            except Exception as e:
                return create_error_response(f"Failed to get test results: {str(e)}", e)

        @tool
        async def handle_user_confirmation_tool(confirmation_data: str) -> str:
            """Handle user confirmation responses from Customer Success agent"""
            return await self.message_router.route_user_confirmation(confirmation_data)

        @tool
        async def handle_eval_question_tool(question_data: str) -> str:
            """Handle evaluation questions from Eval agent to user"""
            return await self.message_router.route_eval_question(question_data)

        @tool
        async def handle_eval_results_tool(results_data: str) -> str:
            """Handle evaluation results from Eval agent to user"""
            return await self.message_router.route_eval_results(results_data)

        @tool
        def get_registered_agents_tool() -> str:
            """Get all registered agents and their assistant IDs"""
            return self.agent_registry.get_registered_agents()

        @tool
        def get_agent_status_tool() -> str:
            """Get status of all agents"""
            return self.agent_registry.get_agent_status()

        @tool
        async def send_a2a_message_to_agent(target_assistant_id: str, target_port: int, message: str, thread_id: str = None) -> str:
            """Send a message to a specific agent using LangGraph A2A protocol"""
            try:
                raw = await send_a2a_message(target_assistant_id, target_port, message, thread_id)
                try:
                    data = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except Exception:
                    data = {"success": False, "response": "", "raw": raw}
                resp_text = data.get("response") if data.get("success") else ""
                return create_success_response({"response": resp_text, "raw": data})
            except Exception as e:
                return create_error_response(f"Failed to send message: {str(e)}", e)

        @tool
        def get_all_threads_tool() -> str:
            """List all conversation threads (for UI navigation)."""
            try:
                threads = self.data_manager.get_all_threads()
                return create_success_response({
                    "threads": threads,
                    "count": len(threads)
                })
            except Exception as e:
                return create_error_response(f"Failed to list threads: {str(e)}", e)

        return [
            register_agent_tool,
            store_eval_suite_tool,
            get_eval_suites_tool,
            store_test_results_tool,
            get_conversation_history_tool,
            get_test_results_tool,
            handle_user_confirmation_tool,
            handle_eval_question_tool,
            handle_eval_results_tool,
            get_registered_agents_tool,
            get_agent_status_tool,
            send_a2a_message_to_agent,
            get_all_threads_tool
        ]

    def build_graph(self):
        """Build the orchestrator graph with standard pattern"""
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        ).bind_tools(self.tools)

        def _guess_thread_id(state: OrchestratorState, content: str) -> str:
            # Prefer an existing context value if present; else deterministic fallback
            try:
                ctx = state.get("context") or {}
                tid = ctx.get("thread_id")
            except Exception:
                tid = None
            if not tid:
                tid = f"thread_{hashlib.md5(content.encode('utf-8')).hexdigest()[:8]}"
            return tid

        def _guess_origin(last_msg) -> str:
            role = getattr(last_msg, "role", "") or getattr(last_msg, "type", "")
            name = getattr(last_msg, "name", "")
            if name:
                return name
            if role in ("user", "human"):
                return "ui_or_agent"
            if role in ("ai", "assistant"):
                return "agent"
            return "unknown"

        def broadcast_node(state: OrchestratorState):
            """Fan-out any inbound message to internal agents; persist inbound to DB."""
            if not state.get("messages"):
                return {}

            last = state["messages"][-1]
            content = getattr(last, "content", "") or ""
            if not content:
                return {}

            # Skip if this was already identified as a broadcast envelope
            if isinstance(content, str) and content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and parsed.get("broadcast") is True:
                        return {}
                except Exception:
                    pass

            role = getattr(last, "role", "") or getattr(last, "type", "")
            if role not in ("user", "human", "ai", "assistant"):
                return {}

            origin = _guess_origin(last)
            thread_id = _guess_thread_id(state, content)

            # Persist inbound message so UI threads populate
            try:
                self.data_manager.store_conversation_message(
                    thread_id=thread_id,
                    sender="ui" if origin == "ui_or_agent" else "agent",
                    content=content,
                    role="user" if origin == "ui_or_agent" else "assistant"
                )
            except Exception:
                pass

            # Per-thread dedupe within 5 seconds
            digest = hashlib.md5(content.encode("utf-8")).hexdigest()
            recent = state.get("recent_broadcasts", {})
            per_thread = recent.get(thread_id, {})
            now = time.time()
            last_ts = per_thread.get(digest)
            if last_ts and (now - last_ts) < 5.0:
                return {}
            per_thread[digest] = now
            recent[thread_id] = per_thread

            envelope = json.dumps({
                "broadcast": True,
                "origin": origin,
                "thread_id": thread_id,
                "text": content
            })

            # Restrict broadcast scope to internal agents defined in deployment-config.json (default)
            try:
                cfg = get_config()
                allowed_internal = set(cfg.list_agents())
            except Exception:
                allowed_internal = {"orchestrator", "customer_success", "eval_agent"}
            scope = os.getenv("SEER_BROADCAST_SCOPE", "internal").lower()

            try:
                with httpx.Client(timeout=3.0) as client:
                    for name, info in self.agent_registry.registered_agents.items():
                        if name == "orchestrator":
                            continue
                        if scope == "internal" and name not in allowed_internal:
                            continue
                        port = info.get("port")
                        assistant_id = info.get("assistant_id")
                        if not (port and assistant_id):
                            continue
                        url = f"http://127.0.0.1:{port}/a2a/{assistant_id}"
                        import uuid as _uuid
                        payload = {
                            "jsonrpc": "2.0",
                            "id": str(_uuid.uuid4()),
                            "method": "message/send",
                            "params": {
                                "message": {"role": "user", "parts": [{"kind": "text", "text": envelope}]},
                                "messageId": str(_uuid.uuid4()),
                                "thread": {"threadId": thread_id}
                            }
                        }
                        try:
                            client.post(url, json=payload, headers={"Accept": "application/json"})
                        except Exception:
                            pass
            except Exception:
                pass

            note = f"BROADCAST relayed: origin={origin} thread={thread_id}"
            return {"recent_broadcasts": recent, "messages": [SystemMessage(content=note)]}

        def orchestrator_node(state: OrchestratorState):
            """Main orchestrator node"""
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
        workflow.add_node("broadcast", broadcast_node)
        # Insert direct route node inline here for deterministic JSON actions
        def direct_route_node(state: OrchestratorState):
            try:
                if not state.get("messages"):
                    return {}
                last = state["messages"][-1]
                content = getattr(last, "content", "") or ""
                if not content or not isinstance(content, str):
                    return {}
                c = content.strip()
                if not (c.startswith("{") and (c.endswith("}") or c.endswith("}\n") or c.endswith("}\r\n"))):
                    return {}
                data = json.loads(c)
                action = data.get("action")
                payload = data.get("payload") or {}
                if not action:
                    return {}
                if action == "get_all_threads":
                    threads = self.data_manager.get_all_threads()
                    return {"messages": [SystemMessage(content=create_success_response({"threads": threads, "count": len(threads)}))]}
                if action == "get_conversation_history":
                    thread_id = payload.get("thread_id")
                    if not thread_id:
                        return {"messages": [SystemMessage(content=create_error_response("thread_id is required"))]}
                    msgs = self.data_manager.get_conversation_history(thread_id)
                    return {"messages": [SystemMessage(content=create_success_response({"thread_id": thread_id, "messages": msgs, "count": len(msgs)}))]}
                return {}
            except Exception as e:
                return {"messages": [SystemMessage(content=create_error_response(f"direct_route error: {str(e)}"))]}

        workflow.add_node("direct_route", direct_route_node)
        workflow.add_node("orchestrator", orchestrator_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.set_entry_point("broadcast")
        workflow.add_edge("broadcast", "direct_route")
        workflow.add_edge("direct_route", "orchestrator")
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
                ["message_routing", "data_management", "agent_coordination"]
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
