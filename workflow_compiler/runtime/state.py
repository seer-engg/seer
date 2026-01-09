"""
Typed helpers for LangGraph state objects used by the compiler runtime.
"""

from __future__ import annotations

from typing import TypedDict


class WorkflowState(TypedDict, total=False):
    # Dynamic user-defined keys fill the mapping; inputs are passed via config.
    pass


INTERNAL_STATE_PREFIX = "__"
