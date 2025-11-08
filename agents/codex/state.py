from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
from shared.schema import CodexInput, CodexOutput, ExperimentResultContext


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


class CodexState(CodexInput, CodexOutput):
    messages: Annotated[list[BaseMessage], add_messages] = Field(None, description="The message context for the codex agent")
    planner_thread: Annotated[list[BaseMessage], add_messages] = Field(None, description="The message context for planner node")
    coder_thread: Annotated[list[BaseMessage], add_messages] = Field(None, description="The message context for coder node")
    structured_response: Optional[dict] = Field(None, description="The structured response")
    taskPlan: Optional[TaskPlan] = Field(None, description="The task plan")

    server_running: bool = Field(False, description="Whether the server is running")
    pr_summary: Optional[str] = Field(None, description="The summary of the PR")
    success: bool = Field(False, description="Whether the request was successful")    

    attempt_number: int = Field(0, description="The number of attempts")
    max_attempts: int = Field(2, description="The maximum number of attempts")
    latest_test_results: List[ExperimentResultContext] = Field(default_factory=list, description="Results from the most recent programmer test run")
