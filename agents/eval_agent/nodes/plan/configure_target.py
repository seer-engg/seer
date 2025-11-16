import json
import asyncio
from e2b import AsyncSandbox
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from agents.eval_agent.models import EvalAgentPlannerState
from shared.schema import SandboxContext
from shared.logger import get_logger
from shared.mcp_client import get_mcp_client_and_configs
from shared.config import TARGET_AGENT_PORT

logger = get_logger("eval_agent.plan.configure")

async def configure_target_agent(state: EvalAgentPlannerState) -> dict:
    """
    Configures the Target Agent by sending it the user's request
    and the MCP resource/config map.
    """
    if not state.sandbox_context:
        raise ValueError("Sandbox context is missing, cannot configure agent.")
    
    logger.info("Connecting to sandbox to configure target agent...")
    sbx = await AsyncSandbox.connect(state.sandbox_context.sandbox_id)
    deployment_url = sbx.get_host(TARGET_AGENT_PORT)
    if not deployment_url.startswith("http"):
        deployment_url = f"https://{deployment_url}"

    # 1. Get MCP configs (to pass to the TA)
    _, mcp_configs = await get_mcp_client_and_configs(state.mcp_services)
    
    # 2. Set up the RemoteGraph client for the TA
    sync_client = get_sync_client(url=deployment_url)
    remote_graph = RemoteGraph(
        state.github_context.agent_name,
        sync_client=sync_client,
    )
    
    # 3. Create a new thread for this configuration/run
    thread = await asyncio.to_thread(sync_client.threads.create)
    thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}

    # 4. Create the config payload
    config_payload = {
        "type": "config",
        "mcp_resources": state.mcp_resources,
        "mcp_configs": mcp_configs
    }
    
    # 5. Create the user request payload
    user_request_payload = {
        "type": "invoke",
        "content": state.user_context.raw_request
    }
    
    # 6. Send the messages to the Target Agent
    try:
        logger.info(f"Sending config payload to TA thread {thread['thread_id']}...")
        await asyncio.to_thread(
            remote_graph.invoke,
            {"messages": [{"role": "user", "content": json.dumps(config_payload)}]},
            thread_cfg,
        )
        
        logger.info(f"Sending user request to TA thread {thread['thread_id']}...")
        await asyncio.to_thread(
            remote_graph.invoke,
            {"messages": [{"role": "user", "content": json.dumps(user_request_payload)}]},
            thread_cfg,
        )
        logger.info("Target agent configured and user request sent.")
        
    except Exception as e:
        logger.error(f"Failed to configure target agent: {e}")
        raise
    
    return {}
