import hashlib
import json
from typing import Optional, Annotated
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from agents.generic_worker import create_generic_worker
from models import WorkerResponse, WorkerStatus
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore

import logging

logger = logging.getLogger(__name__)

@tool
async def spawn_worker(
    task_instruction: str,
    reasoning: str,
    runtime: ToolRuntime = None,
    store: Annotated[BaseStore, InjectedStore] = None
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
        runtime: ToolRuntime (automatically provided by LangGraph)
        store: Shared In-Memory Store (automatically provided)
    
    Returns:
        JSON string of WorkerResponse (status, message, error)
        Todos are automatically updated based on worker response (success → remove todo, failure → keep todo)
    
    LangFuse Tracing:
    - Automatically receives callbacks from orchestrator via runtime
    - Creates nested trace for worker execution
    - Worker's trace appears as a child span under the orchestrator's trace
    """
    # Log reasoning (mandatory)
    logger.info(f"🤔 Worker reasoning: {reasoning}")
    
    # Create generic worker dynamically, passing the shared store
    worker = create_generic_worker("Task Executor", task_instruction, store=store)
    
    # Extract callbacks from runtime to propagate to worker
    thread_id = f"worker-{hashlib.md5(task_instruction.encode()).hexdigest()[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # CRITICAL: Propagate LangFuse callbacks from orchestrator to worker
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
        worker_response = WorkerResponse(
            status=WorkerStatus.FAILURE,
            message=f"Error executing worker: {e}",
            error=str(e)
        )
        return worker_response.model_dump_json()

