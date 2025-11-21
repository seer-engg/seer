from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
from shared.schema import CodexInput, CodexOutput, ExperimentResultContext

# TODO: move this to shared/schema.py
from agents.eval_agent.nodes.plan.filter_tools import AVAILABLE_TOOLS
from agents.eval_agent.models import ToolSelectionLog


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
    # Agent-specific threading for different nodes
    messages: Annotated[list[BaseMessage], add_messages] = Field(None, description="The message context for the codex agent")
    planner_thread: Annotated[list[BaseMessage], add_messages] = Field(None, description="The message context for planner node")
    coder_thread: Annotated[list[BaseMessage], add_messages] = Field(None, description="The message context for coder node")
    structured_response: Optional[dict] = Field(None, description="The structured response")
    taskPlan: Optional[TaskPlan] = Field(None, description="The task plan")

    # Codex-specific state
    server_running: bool = Field(False, description="Whether the server is running")
    pr_summary: Optional[str] = Field(None, description="The summary of the PR")
    success: bool = Field(False, description="Whether the request was successful")    

    attempt_number: int = Field(0, description="The number of attempts")

    #ATTENTION: This is the maximum number of attempts for the codex agent, will reflect on eval failures. default to 0.
    max_attempts: int = Field(0, description="The maximum number of attempts")
    latest_results: List[ExperimentResultContext] = Field(default_factory=list, description="Results from the most recent programmer test run")

    tool_selection_log: Optional[ToolSelectionLog] = Field(default=None, description="The tool selection log")