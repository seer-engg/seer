"""Finalize the codex agent"""
import asyncio
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from agents.codex.state import CodexState
from agents.eval_agent.constants import LANGFUSE_CLIENT
from shared.logger import get_logger
from shared.config import config

logger = get_logger("codex.finalize")

USER_PROMPT = """
    IN the following request from the user, just update the github branch name to the new branch name, if it is present. If not present just add the new branch name to the request and only return the updated request with github branch name.
    <request>
    {request}
    </request>
    <new_branch_name>
    {new_branch_name}
    </new_branch_name>

    <NOTES>
    - only return the updated request with github branch name.
    - do not include any other text in the response.
    </NOTES>
"""


async def _handoff_to_eval(message_content:str, state: CodexState) -> dict:
    logger.warning("Eval handoff to eval agent is not implemented yet")
    return {}






async def finalize(state: CodexState) -> CodexState:
    logger.info(f"Finalizing state: {state}")
    if config.eval_agent_handoff_enabled and state.context.target_agent_version < config.eval_n_versions:
        llm = ChatOpenAI(model=config.default_llm_model)
        input_messages = []
        input_messages.append(HumanMessage(content=USER_PROMPT.format(request=state.context.user_context.raw_request, new_branch_name=state.new_branch_name)))
        response = await llm.ainvoke(input_messages)
        await _handoff_to_eval(response.content, state)
        return state