"""
Stage 4 — Lower the validated WorkflowSpec into an execution plan.

For V1 the lowering step simply maintains the declarative ordering from the
spec since control flow nodes (if / for_each) carry their own block definitions.
The explicit plan object makes it easy to evolve towards a more sophisticated
representation without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from workflow_compiler.schema.models import Node, WorkflowSpec


@dataclass(frozen=True)
class ExecutionPlan:
    nodes: List[Node]


def build_execution_plan(spec: WorkflowSpec) -> ExecutionPlan:
    return ExecutionPlan(nodes=list(spec.nodes))

