from __future__ import annotations

from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import Field
from shared.schema import CodexInput, CodexOutput, ExperimentResultContext
from shared.config import config


class CodexState(CodexInput, CodexOutput):
    # Agent-specific threading for different nodes
    messages: Annotated[list[BaseMessage], add_messages] = Field(None, description="The message context for the codex agent")
    developer_thread: Annotated[list[BaseMessage], add_messages] = Field(None, description="The message context for developer node")

    # Codex-specific state
    server_running: bool = Field(False, description="Whether the server is running")
    pr_summary: Optional[str] = Field(None, description="The summary of the PR") 

    attempt_number: int = Field(0, description="The number of attempts")

    #ATTENTION: This is the maximum number of attempts for the codex agent, will reflect on eval failures. default to 0.
    max_attempts: int = Field(4, description="The maximum number of attempts")
    latest_results: List[ExperimentResultContext] = Field(default_factory=list, description="Results from the most recent programmer test run")

    @property
    def success(self) -> bool:
        """Whether the codex agent was successful"""
        passed = True
        for result in self.latest_results:
            if not result.passed:
                passed = False
                break
        return passed
