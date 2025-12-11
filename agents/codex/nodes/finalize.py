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
    eval_payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": message_content}],
        "target_agent_version": state.context.target_agent_version,
    }


    logger.info("Eval payload: %s", eval_payload)

    # create a new thread for the Eval agent
    eval_sync_client = get_sync_client(url=config.eval_remote_url)
    thread = await asyncio.to_thread(eval_sync_client.threads.create)

    eval_thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}
    
    # Generate deterministic trace ID from thread ID for distributed tracing
    trace_id = None
    langfuse_handler = None
    if LANGFUSE_CLIENT:
        trace_id = Langfuse.create_trace_id(seed=thread["thread_id"])
        # Initialize CallbackHandler with public_key (required for proper tracing)
        langfuse_handler = CallbackHandler(
            public_key=config.langfuse_public_key
        ) if config.langfuse_public_key else CallbackHandler()
        # Add trace context and metadata to config
        eval_thread_cfg["metadata"] = eval_thread_cfg.get("metadata", {})
        eval_thread_cfg["metadata"]["langfuse_trace_id"] = trace_id
        eval_thread_cfg["metadata"]["project_name"] = config.codex_project_name  # Add project_name for filtering (codex uses separate project name)
    
    eval_remote = RemoteGraph(
        "eval_agent",
        url=config.eval_remote_url,
        sync_client=eval_sync_client,
    )

    # Fire-and-forget: run remote eval in a background thread and return immediately
    # Wrap with Langfuse trace context if available
    async def invoke_with_tracing():
        if LANGFUSE_CLIENT and trace_id:
            from langfuse import propagate_attributes
            # Use propagate_attributes to ensure metadata is attached to root trace
            # This wraps the invocation to propagate metadata to all observations
            with propagate_attributes(metadata={"project_name": config.codex_project_name}):
                with LANGFUSE_CLIENT.start_as_current_observation(
                    as_type="span",
                    name="eval-remote-invocation",
                    trace_context={"trace_id": trace_id}
                ) as span:
                    span.update_trace(input=eval_payload)
                    # Pass metadata via config to ensure it's attached to the root trace
                    invoke_config = {**eval_thread_cfg}
                    if langfuse_handler:
                        invoke_config["callbacks"] = [langfuse_handler]
                    # Metadata is already in eval_thread_cfg["metadata"] from above
                    result = await asyncio.to_thread(
                        eval_remote.invoke,
                        eval_payload,
                        invoke_config,
                    )
                    span.update_trace(output=result)
        else:
            await asyncio.to_thread(
                eval_remote.invoke,
                eval_payload,
                eval_thread_cfg,
            )
    
    asyncio.create_task(invoke_with_tracing())
    logger.info("Eval handoff to eval agent completed")
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