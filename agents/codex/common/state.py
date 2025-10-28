from __future__ import annotations

from typing import Annotated, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field



class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "system"]
    content: str


class TaskItem(BaseModel):
    description: str = Field(..., description="Concise action to perform")
    status: Literal["todo", "done"] = Field("todo", description="Item status")


class TaskPlan(BaseModel):
    title: str = Field(..., description="Short plan title")
    items: List[TaskItem] = Field(..., description="Ordered plan steps")

class BaseState(TypedDict, total=False):
    request: str
    repo_path: str
    repo_url: str
    branch_name: str
    sandbox_session_id: str
    messages: Annotated[list[BaseMessage], add_messages]

    taskPlan: Optional[TaskPlan]


class PlannerState(BaseState):
    autoAcceptPlan: bool
    structured_response: dict
    setup_script: str = 'pip install -r requirements.txt'


class ProgrammerState(BaseState):
    pass


class ReviewerState(BaseState):
    pass
