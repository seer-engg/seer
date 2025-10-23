"""Coding Agent for Seer - implemented via create_agent runtime."""

from typing import Annotated, TypedDict, Optional
import json

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from seer.shared.llm import get_llm
from seer.shared.logger import get_logger

logger = get_logger('coding_agent')


@tool
def think(thought: str) -> str:
    """
    Think tool for coding_agent: logs internal reflection; no side effects.
    """
    logger.info(f"THINK: {thought}")
    return json.dumps({"success": True, "thought": thought})


class CodingAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = (
    "You are the Coding Agent. Acknowledge requests briefly and, when given a JSON payload, "
    "echo any repo_url and repo_id you find. Keep responses concise."
)


def build_graph():
    model = get_llm(temperature=0.2)
    tools = [think]
    return create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


graph = build_graph()


