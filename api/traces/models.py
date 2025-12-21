from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RunNode(BaseModel):
    """A single run node in the trace tree."""

    id: str
    name: str
    run_type: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str
    error: Optional[str]
    inputs: Optional[Dict[str, Any]]
    outputs: Optional[Dict[str, Any]]
    children: List["RunNode"] = []


class TraceDetail(BaseModel):
    """Full trace detail with nested runs."""

    id: str
    name: str
    project: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str
    error: Optional[str]
    run_type: str
    inputs: Optional[Dict[str, Any]]
    outputs: Optional[Dict[str, Any]]
    children: List[RunNode] = []


class TraceSummary(BaseModel):
    """Summary of a trace for list view."""

    id: str
    name: str
    project: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str
    error: Optional[str]
    run_type: str
    inputs: Optional[Dict[str, Any]]
    outputs: Optional[Dict[str, Any]]


# Resolve forward references
RunNode.model_rebuild()


__all__ = ["RunNode", "TraceDetail", "TraceSummary"]


