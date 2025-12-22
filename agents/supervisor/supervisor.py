import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage, BaseMessage
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelRetryMiddleware
from langgraph.graph import StateGraph, END, START
from .state import SupervisorState
from .tools.spawn_worker import spawn_worker
from .tools.think_tool import think
from .tools.integration_tools import get_available_integrations
from .models import WorkerResponse, WorkerStatus
from .tools.user_context_store import get_user_context_store
  
logger = logging.getLogger(__name__)


@tool
def write_todos(todos: list[str]) -> str:
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
    return f"✅ Todos updated: {len(todos)} items"


def create_supervisor():
    """
    Creates the Unified Supervisor Agent.
    
    Architecture:
    - Single Node ("supervisor") that handles both planning and execution.
    - Maintains a 'todos' list in state (simple list of strings).
    - Dynamically decides to update todos or spawn workers.
    """
    
    # Get available integrations
    available_integrations = get_available_integrations()
    integrations_list = ", ".join([i.upper() for i in available_integrations])
    
    # 1. Define Tools
    tools = [
        think,
        write_todos,
        spawn_worker,
    ]
    
    # 2. Define System Prompt
    system_prompt_template = """You are the Supervisor Agent. Answer informational queries directly, or create todos and delegate actionable tasks to Workers.

**INTEGRATIONS:** {integrations_list}
**TODOS:** {todos_text}

**WORKFLOW:**
1. **PLAN**: If todos empty and request requires action, call `write_todos()` to create plan.
   - Informational queries: Answer directly (no todos/workers)
   - Actionable tasks: Group by service/domain (e.g., "Get PR from GitHub: Find and extract details")
   
2. **DELEGATE**: For each todo, call `spawn_worker(instruction, reasoning, integrations)`:
   - `instruction`: Concise task
   - `reasoning`: Required - explain service/domain
   - `integrations`: Optional - restrict tool search (e.g., ["github"], ["asana"])
   
3. **REVIEW**: After worker completes, remove todo via `write_todos()`.
4. **FINISH**: When todos empty, respond to user.

**TOOL PATTERN (MANDATORY):**
- Pattern: `think()` → action tool → `think()` → action tool...
- Always call `think()` before AND after every `write_todos()` or `spawn_worker()` call
- `think()` requires `last_tool_call` parameter (format: "Tool: <name>, Result: <what happened>")

**TOOLS:**
- `think(scratchpad, last_tool_call)` - Plan/reflect (before/after every action)
- `write_todos(todos)` - Manage task list
- `spawn_worker(instruction, reasoning, integrations)` - Delegate to workers

**RULES:**
- Delegate heavy work to workers
- Group todos by service/domain
- Match integrations to service (GitHub → ["github"], Asana → ["asana"])
"""
    
    # 3. Define Model & Middleware
    # Use config module which ensures OPENAI_API_KEY is available
    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.0,
    )
    
    # Middleware: Model retry + Tool call limits
    # ModelRetryMiddleware: Retries model calls with exponential backoff (4 total attempts: initial + 3 retries)
    middleware = [
        ModelRetryMiddleware(
            max_retries=3,  # 3 retries (4 total attempts)
            backoff_factor=2.0,  # Exponential backoff: 2s, 4s, 8s
            initial_delay=2.0,  # Initial delay of 2 seconds
        ),
        ToolCallLimitMiddleware(thread_limit=30, run_limit=10),
        ToolCallLimitMiddleware(tool_name="write_todos", thread_limit=5, run_limit=3),
        ToolCallLimitMiddleware(tool_name="spawn_worker", thread_limit=10, run_limit=4),  # Increased: allow one more worker spawn
    ]
    
    # 4. Create the Agent
    # NOTE: We don't pass system_prompt here because we manage it manually
    # in supervisor_node to dynamically update it with todos
    agent_runnable = create_agent(
        model=model,
        tools=tools,
        middleware=middleware
    )
    
    # 5. Define the Node
    async def supervisor_node(state: SupervisorState):
        logger.info("🤖 Supervisor Node Active")
        
        # Store user context from state for tools to access
        user_context_store = get_user_context_store()
        # Extract thread_id from config if available, otherwise use "default"
        # Note: thread_id is typically in config["configurable"]["thread_id"]
        # For now, store with "default" - tools will use the same
        user_context_store.store_user_context(state, thread_id="default")
        
        messages = state.get("messages", [])
        
        # Ensure all messages are proper LangChain message objects
        # CRITICAL: create_agent expects HumanMessage, AIMessage, etc., not generic BaseMessage
        normalized_messages = []
        for msg in messages:
            try:
                if isinstance(msg, dict):
                    # Convert dict to proper message object
                    msg_type = msg.get("type") or msg.get("role", "human")
                    content = msg.get("content", "") or ""  # Ensure string
                    tool_calls = msg.get("tool_calls", [])
                    id_ = msg.get("id")
                    name = msg.get("name")
                    
                    if msg_type == "human" or msg_type == "user":
                        normalized_messages.append(HumanMessage(content=content, id=id_, name=name))
                    elif msg_type == "ai" or msg_type == "assistant":
                        normalized_messages.append(AIMessage(content=content, tool_calls=tool_calls, id=id_, name=name))
                    elif msg_type == "system":
                        normalized_messages.append(SystemMessage(content=content, id=id_, name=name))
                    elif msg_type == "tool":
                        tool_call_id = msg.get("tool_call_id")
                        # Tool messages must have tool_call_id
                        if tool_call_id:
                            normalized_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, id=id_, name=name))
                        else:
                            # Fallback for missing tool_call_id
                            logger.warning(f"Tool message missing tool_call_id, treating as human message")
                            normalized_messages.append(HumanMessage(content=f"[Tool Output] {content}", id=id_, name=name))
                    else:
                        normalized_messages.append(HumanMessage(content=content, id=id_, name=name))
                elif isinstance(msg, BaseMessage):
                    # BaseMessage but not proper subclass - recreate based on type attribute
                    msg_type = getattr(msg, "type", None)
                    content = getattr(msg, "content", "") or ""
                    id_ = getattr(msg, "id", None)
                    name = getattr(msg, "name", None)
                    tool_calls = getattr(msg, "tool_calls", [])
                    
                    if msg_type == "human" or msg_type == "user":
                        normalized_messages.append(HumanMessage(content=content, id=id_, name=name))
                    elif msg_type == "ai" or msg_type == "assistant":
                        normalized_messages.append(AIMessage(content=content, tool_calls=tool_calls, id=id_, name=name))
                    elif msg_type == "system":
                        normalized_messages.append(SystemMessage(content=content, id=id_, name=name))
                    elif msg_type == "tool":
                        tool_call_id = getattr(msg, "tool_call_id", None)
                        if tool_call_id:
                            normalized_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, id=id_, name=name))
                        else:
                            normalized_messages.append(HumanMessage(content=f"[Tool Output] {content}", id=id_, name=name))
                    else:
                        # Fallback: treat as human message
                        logger.warning(f"BaseMessage with unknown type '{msg_type}', treating as human")
                        normalized_messages.append(HumanMessage(content=content, id=id_, name=name))
                else:
                    logger.warning(f"Unexpected message type: {type(msg)}, skipping")
            except Exception as e:
                logger.error(f"Error normalizing message: {e}, skipping", exc_info=True)
        
        messages = normalized_messages
        
        # DEBUG: Log the message content being processed
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                logger.debug(f"📥 Processing message: type={type(last_msg).__name__}, content='{last_msg.content[:200]}...'")
            else:
                logger.debug(f"📥 Processing message: type={type(last_msg).__name__}, content=<no content>")
        else:
            logger.warning("⚠️ Supervisor received NO messages!")
        
        # Format todos for display
        todos = state.get("todos", [])
        if todos:
            todos_text = "\n".join(f"  {i+1}. {todo}" for i, todo in enumerate(todos))
        else:
            todos_text = "  (No todos yet)"
        
        # Format the system prompt with current todos and integrations
        formatted_system_prompt = system_prompt_template.format(
            integrations_list=integrations_list,
            todos_text=todos_text
        )
        
        # Inject/Update System Prompt
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=formatted_system_prompt)] + messages
        else:
            messages = [SystemMessage(content=formatted_system_prompt)] + messages[1:]
            
        # Invoke the agent
        logger.info(f"Invoking Supervisor (Todos: {len(todos)} items)")
        
        # Extract callbacks from state if available
        callbacks = state.get("callbacks", [])
        
        agent_input = dict(state)
        agent_input["messages"] = messages
        
        # Pass callbacks to agent invocation if available
        invoke_kwargs = {}
        if callbacks:
            invoke_kwargs["config"] = {"callbacks": callbacks}
            logger.debug(f"📊 Passing {len(callbacks)} callback(s) to agent")
        
        result = await agent_runnable.ainvoke(agent_input, **invoke_kwargs)
        
        # DEBUG: Log agent response
        agent_messages = result.get("messages", [])
        logger.debug(f"📤 Agent returned {len(agent_messages)} message(s)")
        
        # Log tool calls if any
        for msg in agent_messages:
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get('name') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                    logger.info(f"🛠️  Agent called tool: {tool_name}")
        
        # Extract todos updates from write_todos tool calls
        state_updates = {}
        agent_messages = result.get("messages", [])
        
        # Look for write_todos tool calls and extract the todos argument
        # CRITICAL: Extract todos from the MOST RECENT write_todos call
        for msg in reversed(agent_messages):
            if isinstance(msg, ToolMessage) and msg.name == "write_todos":
                # Find the corresponding AIMessage with the tool call
                tool_call_id = getattr(msg, 'tool_call_id', None)
                if tool_call_id:
                    for prev_msg in agent_messages:
                        if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                            for tc in prev_msg.tool_calls:
                                tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                                if tc_id == tool_call_id:
                                    # Extract todos from tool call arguments
                                    args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                    if isinstance(args, dict) and 'todos' in args:
                                        todos_list = args['todos']
                                        # CRITICAL: Ensure todos_list is a list
                                        if isinstance(todos_list, list):
                                            state_updates["todos"] = todos_list
                                            logger.info(f"✅ Todos update found in write_todos call: {len(todos_list)} items")
                                        else:
                                            logger.warning(f"⚠️  write_todos returned non-list: {type(todos_list)}")
                                        break
                            if "todos" in state_updates:
                                break
                if "todos" in state_updates:
                    break
        
        # DEBUG: Log current todos state
        current_todos = state.get("todos", [])
        logger.debug(f"📋 Current todos in state: {len(current_todos)} items - {current_todos}")
        
        # Auto-remove completed todos based on spawn_worker responses
        # Process ALL worker completions (success or failure) to prevent infinite loops
        processed_todos_count = 0
        for msg in reversed(agent_messages):
            if isinstance(msg, ToolMessage) and msg.name == "spawn_worker":
                processed_todos_count += 1
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
                        logger.debug(f"✅ Worker completed successfully")
                    else:
                        logger.warning(f"⚠️ Worker failed or returned non-success status: {worker_response.status}")
                        
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug(f"Could not parse worker response: {e}")
        
        # Remove todos based on number of processed workers (success OR failure)
        # This ensures we don't get stuck in a loop retrying the same todo forever
        if processed_todos_count > 0:
            current_todos = state.get("todos", [])
            if len(current_todos) >= processed_todos_count:
                updated_todos = current_todos[processed_todos_count:]
                if updated_todos != current_todos:
                    state_updates["todos"] = updated_todos
                    logger.info(f"✅ Auto-removed {processed_todos_count} processed todo(s). Remaining: {len(updated_todos)}")
            else:
                # More workers than todos - clear all todos
                state_updates["todos"] = []
                logger.info(f"✅ All {len(current_todos)} todos processed by {processed_todos_count} workers")
        
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
    
    return workflow.compile()

