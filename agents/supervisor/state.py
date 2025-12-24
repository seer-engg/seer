from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class Context(TypedDict):
    """
    Context for the supervisor agent.
    """
    #integrations is a dictionary of integration names and their corresponding context eg 
    # {
    #     "sandbox": null,
    #     "github": {
    #         "id": "1098514231",
    #         "name": "seer-engg/reflexion"
    #     },
    #     "googledrive": null,
    #     "asana": null
    # }
    integrations: Dict[str, Any]
    user_id: str

class SupervisorState(TypedDict):
    """
    State for the main Supervisor agent.
    Keeps track of the high-level conversation, todos list, and delegated tasks.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    todos: List[str]  # Simple list of todo items (tasks to complete)
    tool_call_counts: Optional[Dict[str, int]]  # Track tool calls
    context: Context

class WorkerState(TypedDict):
    """
    State for a specialist worker agent.
    Isolated context for performing specific heavy-lifting tasks.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    tool_call_counts: Optional[Dict[str, int]]  # Track tool calls: {"tool_name": count, "_total": total}

