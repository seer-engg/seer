from agents.codex.common.state import PlannerState
from shared.logger import get_logger
logger = get_logger("planner.finalize")
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

USER_PROMPT = """
    IN the following request from the user, just update the github branch name to the new branch name, if it is present. If not present just add the new branch name to the request and only return the updated request with github branch name.
    <request>
    {request}
    </request>
    <new_branch_name>
    {new_branch_name}
    </new_branch_name>

    # important notes:
    - only return the updated request with github branch name.
    - do not include any other text in the response.
"""

from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from agents.eval_agent.constants import EVAL_REMOTE_URL, LANGSMITH_CLIENT, N_VERSIONS
import asyncio
from typing import Dict, Any

async def _handoff_to_eval(message_content:str, state: PlannerState) -> dict:
    eval_payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": message_content}],
        "target_agent_version": state.target_agent_version,
    }


    logger.info("Eval payload: %s", eval_payload)

    # create a new thread for the Eval agent
    eval_sync_client = get_sync_client(url=EVAL_REMOTE_URL)
    thread = await asyncio.to_thread(eval_sync_client.threads.create)

    eval_thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
    eval_remote = RemoteGraph(
        "eval_agent",
        url=EVAL_REMOTE_URL,
        client=LANGSMITH_CLIENT,
        sync_client=eval_sync_client,
        distributed_tracing=True,
    )

    # Fire-and-forget: run remote eval in a background thread and return immediately
    asyncio.create_task(
        asyncio.to_thread(
            eval_remote.invoke,
            eval_payload,
            eval_thread_cfg,
        )
    )
    logger.info("Eval handoff to eval agent completed")
    return {}






async def finalize(state: PlannerState) -> PlannerState:
    logger.info(f"Finalizing planner state: {state}")
    if os.getenv("EVAL_HANDOFF_ENABLED") == "true" and state.target_agent_version < N_VERSIONS:
        llm = ChatOpenAI(model="gpt-4o-mini")
        input_messages = []
        input_messages.append(HumanMessage(content=USER_PROMPT.format(request=state.user_context.user_raw_request, new_branch_name=state.new_branch_name)))
        response = await llm.ainvoke(state.messages)
        await _handoff_to_eval(response.content, state)
        return state