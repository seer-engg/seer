from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START

from agents.state import WorkerState
from tools.composio_tools import search_tools, execute_tool
from tools.openai_retry_middleware import wrap_model_with_retry
from tools.think_tool import think
from langgraph.store.base import BaseStore

def create_generic_worker(role_name: str, specific_instructions: str, store: BaseStore = None):
    """
    Creates a Generic Worker Sub-Agent.
    
    This agent is NOT hardcoded with specific tools.
    Instead, it uses `search_tools` and `execute_tool` to dynamically find what it needs.
    
    Args:
        role_name: Name of the role (e.g. "GitHub Researcher")
        specific_instructions: Task-specific system prompt additions.
        store: In-memory store for shared artifacts.
    """
    
    # 1. Generic Toolset
    # Workers can search for tools and execute them. All tool outputs are visible in worker's isolated context.
    tools = [
        think,  # Think tool for explicit reasoning after tool calls
        search_tools, 
        execute_tool, 
    ]
    
    # 2. System Prompt (PRUNED - ~500 chars vs ~1200)
    system_prompt = f"""You are {role_name}. Mission: {specific_instructions}

**PROTOCOL:**
1. Search for tools using `search_tools(query, reasoning)`.
   - **MANDATORY**: Always provide `reasoning` explaining:
     * What capability you need (e.g., "search tasks", "create task", "find PR")
     * Why you need it (context from your task/instructions)
     * What you're trying to accomplish
   - Query should describe CAPABILITIES, not include actual data values
   - GOOD: search_tools("search Asana tasks by title", "I need to find existing tasks matching the PR title to avoid duplicates")
   - BAD: search_tools("search Asana tasks by title 'Seer: Evaluate my agent'", "...") ❌ (includes actual data in query)
   - BAD: search_tools("Asana", "...") ❌ (too vague)
2. Execute tools with `execute_tool(name, params)` where params includes actual data.
3. After EVERY tool call, call `think()` with:
   - What tool was called and what it returned
   - What tool you'll call next and why
   - Any errors or issues encountered

**FAILURE HANDLING:**
- If a tool fails or hits limits, explain why in `think()` and consider alternatives
- Don't repeat failed operations - pivot to alternative approaches
- If searching fails/exhausts, consider direct creation/update instead

**REPORTING:**
- Summarize results clearly. Don't paste huge JSON.
- Report success/failure in final response.
"""
    
    # 3. Model with retry wrapper
    # We use a capable model since it needs to reason about tool discovery
    model = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    model = wrap_model_with_retry(model, max_retries=2)  # Retry on invalid_prompt errors
    
    # Tool call limits middleware - INCREASED LIMITS for better worker autonomy
    middleware = [
        ToolCallLimitMiddleware(thread_limit=20, run_limit=8),  # Increased global limit
        ToolCallLimitMiddleware(tool_name="search_tools", thread_limit=5, run_limit=3),
        ToolCallLimitMiddleware(tool_name="execute_tool", thread_limit=10, run_limit=5),
        # Note: force_reasoning_after_tools removed - think tool + prompt is sufficient
    ]
    
    # 4. Create agent using create_agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware
    )
    
    # 5. Graph Definition
    async def agent_node(state: WorkerState, config: dict = None):
        messages = state["messages"]
        
        # Prepend system prompt if not present
        if not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        else:
            # Update existing system message
            messages = [SystemMessage(content=system_prompt)] + messages[1:]
        
        result = await agent.ainvoke({"messages": messages})
        return result
        
    workflow = StateGraph(WorkerState)
    workflow.add_node("agent", agent_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)  # create_agent handles its own tool loop
    
    return workflow.compile(store=store)

