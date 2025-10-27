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


class ManagerState(TypedDict, total=False):
    request: str
    repo_path: str
    messages: List[Message]
    taskPlan: Optional[TaskPlan]
    autoAcceptPlan: bool


class PlannerState(TypedDict, total=False):
    request: str
    repo_path: str
    messages: List[Message]
    taskPlan: Optional[TaskPlan]
    autoAcceptPlan: bool


class ProgrammerState(TypedDict, total=False):
    request: str
    repo_path: str
    messages: List[Message]
    taskPlan: Optional[TaskPlan]


class ReviewerState(TypedDict, total=False):
    request: str
    repo_path: str
    messages: List[Message]
    taskPlan: Optional[TaskPlan]
