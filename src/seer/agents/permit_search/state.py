"""
Permit search agent state schema.
"""

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PermitSearchState(TypedDict):
    """State for the permit search agent.

    Tracks company name resolution and WARP permit search results.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    company_name: str
    resolved_company: Optional[str]
    permit_results: Optional[dict[str, Any]]
    error: Optional[str]
