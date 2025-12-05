import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

from .state import SupervisorState
from tools.spawn_worker import spawn_worker
from tools.openai_retry_middleware import wrap_model_with_retry
from tools.think_tool import think
from langgraph.store.base import BaseStore
from models import WorkerResponse, WorkerStatus

logger = logging.getLogger(__name__)


@tool
def write_todos(todos: list[str], runtime: ToolRuntime = None) -> Command:
    """Update the todos list. Pass a list of todo items (strings).
    
    **CRITICAL GROUPING RULE:**
    Group todos by SERVICE/DOMAIN boundaries, not by micro-tasks.
    - Each todo should represent ALL work within ONE service/domain
    - Workers are domain-experts: they handle all related operations in their domain
    
    **GOOD EXAMPLES (Domain-Boundary Thinking):**
    - ["Get PR from GitHub: Find most recent merged PR from seer-engg/buggy-coder and extract all details (title, URL, author, date)", "Sync PR to Asana: Search for matching tasks, create/update task with PR details, and close it"]
    
    **BAD EXAMPLES (Micro-Task Thinking):**
    - ["Search GitHub for PR", "Extract PR details", "Search Asana", "Create task", "Close task"] ❌ Too granular
    
    Each todo should be comprehensive enough that a domain-expert worker can complete it independently.
    Group ALL GitHub operations together, ALL Asana operations together, etc.
    
    Example: write_todos(["Get PR from GitHub: Find and extract details", "Sync PR to Asana: Search, create/update, and close"])
    
    This replaces the entire todos list. Use this to create or update your plan."""
    tool_message = ToolMessage(
        content=f"✅ Todos updated: {len(todos)} items",
        tool_call_id=runtime.tool_call_id if runtime and hasattr(runtime, 'tool_call_id') else "write_todos"
    )
    return Command(update={"todos": todos, "messages": [tool_message]})


def create_supervisor(store: BaseStore = None):
    """
    Creates the Unified Supervisor Agent.
    
    Architecture:
    - Single Node ("supervisor") that handles both planning and execution.
    - Maintains a 'todos' list in state (simple list of strings).
    - Dynamically decides to update todos or spawn workers.
    - Uses Shared In-Memory Store for artifacts.
    """
    
    # 1. Define Tools
    tools = [
        think,
        write_todos,
        spawn_worker,
    ]
    
    # 2. Define System Prompt (PRUNED - ~600 chars vs ~1800)
    system_prompt_template = """You are the Supervisor Agent. Complete the user's request by creating todos and delegating to Workers.

**CURRENT TODOS:**
{todos_text}

**PROTOCOL:**
1. **PLAN**: If todos empty, call `write_todos()` to create plan.
   - **CRITICAL**: Group todos by SERVICE/DOMAIN boundaries (GitHub, Asana, Gmail, etc.)
   - Each todo = ALL work within ONE service/domain
   - Example: "Get PR from GitHub: Find most recent merged PR and extract all details" (NOT separate "search" + "extract")
   - Example: "Sync PR to Asana: Search for tasks, create/update with PR details, and close" (NOT separate "search" + "create" + "close")
   
2. **DELEGATE**: For each todo, call `spawn_worker(instruction, reasoning)` with:
   - `instruction`: CONCISE task instruction
   - `reasoning`: **MANDATORY** - WHY this worker is needed and what service/domain it handles
   - GOOD: spawn_worker("Fetch most recent merged PR from seer-engg/buggy-coder and extract details", "GitHub domain: Finding and extracting PR information")
   - BAD: spawn_worker("Search GitHub", "Need to find PR") ❌ Too vague
   
3. **REVIEW**: After worker completes, remove completed todo using `write_todos()`.
4. **FINISH**: When todos empty, respond to user.

**MANDATORY THINKING:**
After EVERY tool call (except 'think'), call `think()` to reason. ALWAYS reference CURRENT TODOS above.

**RULES:**
- Delegate heavy work to `spawn_worker`. Don't do it yourself.
- Group todos by domain/service - don't split same-service tasks.
- **ALWAYS provide reasoning when spawning workers** - explain the service/domain and why.
- Keep todos visible. Update them as you progress.
"""
    
    # 3. Define Model & Middleware
    model = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    model = wrap_model_with_retry(model, max_retries=3)  # Retry on invalid_prompt errors (total 4 attempts)
    
    middleware = [
        ToolCallLimitMiddleware(thread_limit=30, run_limit=10),
        ToolCallLimitMiddleware(tool_name="write_todos", thread_limit=5, run_limit=3),
        ToolCallLimitMiddleware(tool_name="spawn_worker", thread_limit=10, run_limit=4),  # Increased: allow one more worker spawn
    ]
    
    # 4. Create the Agent
    agent_runnable = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt_template,
        middleware=middleware
    )
    
    # 5. Define the Node
    async def supervisor_node(state: SupervisorState, config: dict = None):
        logger.info("🤖 Supervisor Node Active")
        messages = state.get("messages", [])
        
        # Format todos for display
        todos = state.get("todos", [])
        if todos:
            todos_text = "\n".join(f"  {i+1}. {todo}" for i, todo in enumerate(todos))
        else:
            todos_text = "  (No todos yet)"
        
        # Format the system prompt with current todos
        formatted_system_prompt = system_prompt_template.format(
            todos_text=todos_text
        )
        
        # Inject/Update System Prompt
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=formatted_system_prompt)] + messages
        else:
            messages = [SystemMessage(content=formatted_system_prompt)] + messages[1:]
            
        # Invoke the agent
        logger.info(f"Invoking Supervisor (Todos: {len(todos)} items)")
        
        agent_input = dict(state)
        agent_input["messages"] = messages
        
        result = await agent_runnable.ainvoke(agent_input)
        
        # Check for todos updates from Command.update
        state_updates = {}
        if "todos" in result:
            state_updates["todos"] = result["todos"]
            logger.info(f"✅ Todos update found in result: {len(result['todos'])} items")
        
        # Auto-remove completed todos based on spawn_worker responses
        agent_messages = result.get("messages", [])
        for msg in reversed(agent_messages):
            if isinstance(msg, ToolMessage) and msg.name == "spawn_worker":
                try:
                    worker_response_dict = json.loads(msg.content)
                    worker_response = WorkerResponse(**worker_response_dict)
                    
                    # Extract reasoning from the tool call if available
                    reasoning = None
                    if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                        for prev_msg in agent_messages:
                            if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                                for tc in prev_msg.tool_calls:
                                    tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                                    if tc_id == msg.tool_call_id:
                                        args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                        reasoning = args.get('reasoning') if isinstance(args, dict) else getattr(args, 'reasoning', None)
                                        break
                                if reasoning:
                                    break
                    
                    if reasoning:
                        logger.info(f"📋 Worker reasoning: {reasoning}")
                    
                    if worker_response.status == WorkerStatus.SUCCESS:
                        # Find which todo was completed (by matching instruction)
                        # For now, remove the first todo if worker succeeded
                        current_todos = state.get("todos", [])
                        if current_todos:
                            # Simple heuristic: remove first todo if worker succeeded
                            # In future, could match by instruction content
                            updated_todos = current_todos[1:]
                            if updated_todos != current_todos:
                                state_updates["todos"] = updated_todos
                                logger.info(f"✅ Auto-removed completed todo. Remaining: {len(updated_todos)}")
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug(f"Could not auto-update todos from spawn_worker: {e}")
                break
        
        if state_updates:
            result.update(state_updates)
        
        return result
    
    # 6. Define Graph
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_node)
    
    workflow.add_edge(START, "supervisor")
    
    def should_continue(state: SupervisorState):
        """Check if we need to continue or finish."""
        todos = state.get("todos", [])
        
        if not todos:
            logger.info("✅ All todos complete. Ending.")
            return END
    
        logger.info(f"🔄 Looping: {len(todos)} todos remaining.")
        return "supervisor"

    workflow.add_conditional_edges("supervisor", should_continue, {
        "supervisor": "supervisor",
        END: END
    })
    
    return workflow.compile(debug=True, store=store)
