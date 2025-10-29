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

class BaseState(BaseModel):
    request: str = Field(..., description="The request to be fulfilled")
    repo_path: Optional[str] = Field(None, description="The path to the repository")
    repo_url: str = Field(..., description="The URL of the repository")
    branch_name: Optional[str] = Field(None, description="The name of the branch")
    sandbox_session_id: Optional[str] = Field(None, description="The ID of the sandbox session")
    messages: Annotated[list[BaseMessage], add_messages] = Field(None, description="The messages in the conversation")
    taskPlan: Optional[TaskPlan] = Field(None, description="The task plan")
    attempt_number: int = Field(0, description="The number of attempts")
    success: bool = Field(False, description="Whether the request was successful")
    max_attempts: int = Field(2, description="The maximum number of attempts")


class PlannerState(BaseState):
    autoAcceptPlan: bool = Field(True, description="Whether to automatically accept the plan")
    structured_response: Optional[dict] = Field(None, description="The structured response")
    setup_script: str = Field('pip install -r requirements.txt', description="The script to setup the project")


class ProgrammerState(BaseState):
    pass


class ReviewerState(BaseState):
    pass
