"""Dummy Coding Agent for Seer - minimal LangGraph graph that just acknowledges requests."""

from typing import Annotated, TypedDict, Optional
import json

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class CodingAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def respond_node(state: CodingAgentState):
    last_human: Optional[HumanMessage] = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m
            break

    repo_url = None
    repo_id = None
    if last_human and isinstance(last_human.content, str):
        try:
            data = json.loads(last_human.content)
            if isinstance(data, dict):
                repo_url = data.get("repo_url")
                repo_id = data.get("repo_id")
        except Exception:
            # Ignore non-JSON inputs; this dummy agent just acknowledges
            pass

    text = "Coding agent (dummy): received request. "
    if repo_url or repo_id:
        text += f"repo_url={repo_url or 'N/A'}, repo_id={repo_id or 'N/A'}. "
    text += "Returning placeholder response for end-to-end testing."
    return {"messages": [AIMessage(content=text)]}


def build_graph():
    workflow = StateGraph(CodingAgentState)
    workflow.add_node("respond", respond_node)
    workflow.set_entry_point("respond")
    workflow.add_edge("respond", END)
    return workflow.compile()


graph = build_graph()


