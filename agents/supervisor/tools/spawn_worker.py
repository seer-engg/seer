import hashlib
from typing import Optional, List
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from agents.supervisor.generic_worker import create_generic_worker
from agents.supervisor.models import WorkerResponse, WorkerStatus
from agents.supervisor.tools.user_context_store import get_user_context_store

import logging

logger = logging.getLogger(__name__)

@tool
async def spawn_worker(
    task_instruction: str,
    reasoning: str,
    integrations: Optional[List[str]] = None,
    runtime: ToolRuntime = None
) -> str:
    """
    Execute a task using a dynamically spawned worker.
    
    Args:
        task_instruction: CONCISE instruction for the worker.
                         Example: "Fetch most recent merged PR from seer-engg/buggy-coder and extract details"
                         NOT: "Get PR information from GitHub: Fetch the most recently merged PR from seer-engg/buggy-coder and extract all details (title, URL, author, merge date, number)"
        reasoning: **MANDATORY** - Explanation of why this worker is needed and what service/domain it handles.
                   Example: "GitHub domain: Finding and extracting PR information"
                   Must explain: (1) Which service/domain, (2) Why this worker is needed
        integrations: Optional list of integration names (lowercase, e.g., ["github"], ["asana"], ["github", "asana"]).
                      If not specified, worker searches all integrations (slower but comprehensive).
                      Specify integrations to restrict tool search for faster, more focused results.
        runtime: ToolRuntime (automatically provided by LangGraph)
    
    Returns:
        JSON string of WorkerResponse (status, message, error)
        Todos are automatically updated based on worker response (success → remove todo, failure → keep todo)
    
    Note: Callbacks are propagated from runtime if available (for tracing/debugging).
    """
    # Log reasoning (mandatory)
    logger.info(f"🤔 Worker reasoning: {reasoning}")
    if integrations:
        logger.info(f"🔗 Worker integrations: {integrations}")
    
    # Extract callbacks from runtime to propagate to worker
    thread_id = f"worker-{hashlib.md5(task_instruction.encode()).hexdigest()[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # CRITICAL: Copy Supervisor's user context to worker's thread_id BEFORE creating worker
    # Workers need access to user_id, connected_accounts, and resource_ids (workspace GID, etc.)
    user_context_store = get_user_context_store()
    
    # Get Supervisor's context (stored under "default" thread_id)
    supervisor_context = user_context_store.get_user_context(thread_id="default")
    
    # Copy Supervisor's context to worker's thread_id so worker tools can access it
    if supervisor_context.get("user_id") or supervisor_context.get("connected_accounts"):
        user_context_store._user_contexts[thread_id] = supervisor_context.copy()
        # Set thread_id in context variable so tools can access it
        user_context_store.set_current_thread_id(thread_id)
        logger.info(f"✅ Copied Supervisor context to worker thread {thread_id}: user_id={supervisor_context.get('user_id')}, connected_accounts={supervisor_context.get('connected_accounts')}, resource_ids={supervisor_context.get('resource_ids')}")
    else:
        logger.warning(f"⚠️  No Supervisor context found to copy to worker thread {thread_id}")
    
    # Create generic worker dynamically with specified integrations
    # Now that context is set, worker creation can access resource IDs
    worker = create_generic_worker("Task Executor", task_instruction, integrations=integrations)
    
    # Propagate callbacks from orchestrator to worker (if available)
    callbacks = []
    if runtime:
        # Try to get callbacks from runtime
        if hasattr(runtime, 'run_manager') and runtime.run_manager:
            if hasattr(runtime.run_manager, 'get_child'):
                callbacks = runtime.run_manager.get_child()
            elif hasattr(runtime.run_manager, 'handlers'):
                callbacks = runtime.run_manager.handlers
    
    if callbacks:
        config["callbacks"] = callbacks
    
    # CRITICAL: Set thread_id in context variable BEFORE invoking worker
    # This allows worker's tools to access the correct user context
    user_context_store.set_current_thread_id(thread_id)
    
    # Execute worker with callbacks
    try:
        result = await worker.ainvoke(
            {"messages": [HumanMessage(content=task_instruction)]},
            config=config
        )
        
        messages = result.get("messages", [])
        if messages:
            final_content = messages[-1].content
            
            # Parse worker response into structured format
            worker_response = WorkerResponse.from_message_content(final_content, messages)
            
            # Return as JSON string for orchestrator to parse
            return worker_response.model_dump_json()
        else:
            # No messages - return failure response
            worker_response = WorkerResponse(
                status=WorkerStatus.FAILURE,
                message="Worker completed but returned no message",
                error="No messages in worker result"
            )
            return worker_response.model_dump_json()
    except Exception as e:
        # Exception - return failure response
        import traceback
        worker_response = WorkerResponse(
            status=WorkerStatus.FAILURE,
            message=f"Error executing worker: {traceback.format_exc()}",
            error=str(e)
        )
        return worker_response.model_dump_json()

