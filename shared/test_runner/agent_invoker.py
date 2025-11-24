"""
Agent invocation module for test runner.

This module handles ACTUALLY invoking the target agent (e.g., buggy_coder)
with the input_message and capturing its tool calls.

This is Phase 2 of the 3-phase testing architecture.
"""
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from e2b import AsyncSandbox

from shared.schema import SandboxContext, GithubContext
from shared.logger import get_logger
from sandbox.constants import TARGET_AGENT_PORT


logger = get_logger("test_runner.agent_invoker")


class AgentInvocationResult(BaseModel):
    """Result of invoking the target agent."""
    
    final_output: Optional[str] = Field(
        None,
        description="The final text output from the agent, if any"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if agent invocation failed"
    )
    thread_id: str = Field(
        ...,
        description="The thread ID used for this invocation"
    )
    execution_time_seconds: float = Field(
        ...,
        description="How long the agent took to respond"
    )


async def invoke_target_agent(
    sandbox_context: SandboxContext,
    github_context: GithubContext,
    input_message: str,
    timeout_seconds: int = 300
) -> AgentInvocationResult:
    """
    Invoke the target agent with input_message and capture its tool calls.
    
    This is the critical function that ACTUALLY TESTS THE AGENT.
    
    Args:
        sandbox_context: The sandbox where the agent is running
        github_context: Context about the agent being tested
        input_message: The human-readable scenario to send to the agent
        mcp_resources: Resources provisioned in Phase 1 (e.g., {"test_pr": {...}})
        mcp_configs: MCP server configurations to send to the agent
        timeout_seconds: Maximum time to wait for agent response
    
    Returns:
        AgentInvocationResult with tool_calls, output, and any errors
        
    Raises:
        ValueError: If sandbox_context or github_context is invalid
        RuntimeError: If agent invocation fails
        asyncio.TimeoutError: If agent exceeds timeout
    """
    if not sandbox_context or not sandbox_context.sandbox_id:
        raise ValueError(
            "sandbox_context with valid sandbox_id is required. "
            "Cannot invoke agent without sandbox."
        )
    
    if not github_context or not github_context.agent_name:
        raise ValueError(
            "github_context with valid agent_name is required. "
            "Cannot invoke agent without knowing which agent to test."
        )
    
    logger.info(f"Invoking target agent '{github_context.agent_name}' with message: {input_message[:100]}...")
    start_time = datetime.now(timezone.utc)
    
    try:
        # 1. Connect to sandbox and get agent deployment URL
        logger.info(f"Connecting to sandbox {sandbox_context.sandbox_id}...")
        sbx = await asyncio.wait_for(
            AsyncSandbox.connect(sandbox_context.sandbox_id),
            timeout=30
        )
        
        deployment_url = sbx.get_host(TARGET_AGENT_PORT)
        if not deployment_url.startswith("http"):
            deployment_url = f"https://{deployment_url}"
        
        logger.info(f"Agent deployment URL: {deployment_url}")
        
        # 2. Set up RemoteGraph client
        sync_client = get_sync_client(url=deployment_url)
        remote_graph = RemoteGraph(
            github_context.agent_name,
            sync_client=sync_client,
        )
        
        # 3. Create a new thread for this test run
        logger.info("Creating new thread for agent invocation...")
        thread = await asyncio.to_thread(sync_client.threads.create)
        thread_id = thread["thread_id"]
        thread_cfg = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Thread created: {thread_id}")
        
        # 5. Invoke agent with actual input_message
        logger.info(f"Sending input_message to agent: '{input_message[:100]}...'")
        
        # Wrap the synchronous invoke call in asyncio.to_thread and add timeout
        invoke_task = asyncio.to_thread(
            remote_graph.invoke,
            {"messages": [{"role": "user", "content": input_message}]},
            thread_cfg
        )
        
        result = await asyncio.wait_for(invoke_task, timeout=timeout_seconds)
        
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Agent responded in {execution_time:.2f} seconds")

        try:
            # TODO: This is not working , probably because of the 
            final_output = result.get("messages", [])[-1].content
        except Exception as e:
            logger.error(f"Failed to extract final output from agent response: {e}")
            final_output = ""
        
        return AgentInvocationResult(
            final_output=final_output,
            error=None,
            thread_id=thread_id,
            execution_time_seconds=execution_time
        )
        
    except asyncio.TimeoutError:
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - start_time).total_seconds()
        error_msg = f"Agent invocation timed out after {timeout_seconds} seconds"
        logger.error(error_msg)
        
        raise RuntimeError(
            f"Agent '{github_context.agent_name}' exceeded timeout of {timeout_seconds}s. "
            f"The agent may be stuck, hanging, or taking too long to respond. "
            f"Actual execution time: {execution_time:.2f}s"
        )
    
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - start_time).total_seconds()
        error_msg = f"Failed to invoke agent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        raise RuntimeError(
            f"Agent invocation failed after {execution_time:.2f}s: {str(e)}. "
            f"Check that the agent is properly deployed and accessible at the sandbox URL."
        ) from e

