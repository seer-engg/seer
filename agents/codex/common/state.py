from __future__ import annotations

from typing import List, Literal, Optional, TypedDict


class Message(TypedDict, total=False):
    role: Literal["user", "assistant", "system"]
    content: str


class TaskItem(TypedDict, total=False):
    description: str
    status: Literal["todo", "done"]


class TaskPlan(TypedDict, total=False):
    title: str
    items: List[TaskItem]

class BaseState(TypedDict, total=False):
    request: str
    repo_path: str
    repo_url: str
    branch_name: str
    sandbox_session_id: str
    messages: List[Message]
    taskPlan: Optional[TaskPlan]


class ManagerState(BaseState):
    pass


class PlannerState(BaseState):
    autoAcceptPlan: bool


class ProgrammerState(BaseState):
    pass


class ReviewerState(BaseState):
    pass
