from __future__ import annotations

from typing import Annotated, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field



class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "system"]
    content: str


class TaskItem(BaseModel):
    id:int = Field(..., description="The id of the task item")
    description: str = Field(..., description="Concise action to perform")
    context: str = Field(..., description="additional Context from codebase regarding the task item")
    status: Literal["todo", "done"] = Field("todo", description="Item status")


class TaskPlan(BaseModel):
    title: str = Field(..., description="Short plan title")
    items: List[TaskItem] = Field(..., description="Ordered plan steps")



class PlannerIOState(BaseModel):
    request: str = Field(..., description="The request to be fulfilled")
    repo_url: str = Field(..., description="The URL of the repository")
    branch_name: Optional[str] = Field(None, description="The name of the branch")
    messages: Annotated[list[BaseMessage], add_messages] = Field(None, description="The messages in the conversation")
    setup_script: str = Field('pip install -r requirements.txt', description="The script to setup the project")

class PlannerState(PlannerIOState):
    autoAcceptPlan: bool = Field(True, description="Whether to automatically accept the plan")
    structured_response: Optional[dict] = Field(None, description="The structured response")
    repo_path: Optional[str] = Field(None, description="The path to the repository")
    sandbox_session_id: Optional[str] = Field(None, description="The ID of the sandbox session")
    taskPlan: Optional[TaskPlan] = Field(None, description="The task plan")
    

from pydantic import BaseModel, Field
from typing import Optional, List

class Failure(BaseModel):
    test_intention: str = Field(..., description="The intention of the test")
    failure_reason: str = Field(..., description="The reason for the failure")
    optinal_refrence: Optional[str] = Field(None, description="An optional refrence to the code that failed the test")

class TestResults(BaseModel):
    success: bool = Field(..., description="Whether the tests passed for the requested implementations")
    failures: Optional[List[Failure]] = Field(None, description="The failures of the tests")


class ProgrammerState(BaseModel):
    request: str = Field(..., description="The request to be fulfilled")
    taskPlan: TaskPlan = Field(..., description="The task plan")
    repo_path: str = Field(..., description="The path to the repository")
    sandbox_session_id: Optional[str] = Field(None, description="The ID of the sandbox session")
    messages: Annotated[list[BaseMessage], add_messages] = Field(None, description="The messages in the conversation")
    attempt_number: int = Field(0, description="The number of attempts")
    success: bool = Field(False, description="Whether the request was successful")
    max_attempts: int = Field(2, description="The maximum number of attempts")
    taskPlan: Optional[TaskPlan] = Field(None, description="The task plan")
    testResults: Optional[TestResults] = Field(None, description="The test results")