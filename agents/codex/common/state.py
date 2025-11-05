from __future__ import annotations

from typing import Annotated, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
from shared.schema import CodexInput, CodexOutput, SandboxContext, ExperimentResultContext


class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "system"]
    content: str

class TaskItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id:int = Field(..., description="The id of the task item")
    description: str = Field(..., description="Concise action to perform")
    context: str = Field(..., description="additional Context from codebase regarding the task item")
    status: Literal["todo", "done"] = Field(..., description="Item status")


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., description="Short plan title")
    items: List[TaskItem] = Field(..., description="Ordered plan steps")


class PlannerState(CodexInput, CodexOutput):
    messages: Annotated[list[BaseMessage], add_messages] = Field(None, description="The messages in the conversation")
    autoAcceptPlan: bool = Field(True, description="Whether to automatically accept the plan")
    structured_response: Optional[dict] = Field(None, description="The structured response")
    taskPlan: Optional[TaskPlan] = Field(None, description="The task plan")

    server_running: bool = Field(False, description="Whether the server is running")
    pr_summary: Optional[str] = Field(None, description="The summary of the PR")    
    



class ProgrammerState(CodexInput):
    taskPlan: TaskPlan = Field(..., description="The task plan")
    messages: Annotated[list[BaseMessage], add_messages] = Field(None, description="The messages in the conversation")
    attempt_number: int = Field(0, description="The number of attempts")
    success: bool = Field(False, description="Whether the request was successful")
    max_attempts: int = Field(2, description="The maximum number of attempts")
    latest_test_results: List[ExperimentResultContext] = Field(default_factory=list, description="Results from the most recent programmer test run")
    server_running: bool = Field(False, description="Whether the server is running")
    pr_summary: Optional[str] = Field(None, description="The summary of the PR")
    updated_sandbox_context: Optional[SandboxContext] = Field(None, description="The updated sandbox context")