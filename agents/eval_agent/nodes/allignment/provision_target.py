import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI

from agents.eval_agent.models import EvalAgentPlannerState
from sandbox import (
    initialize_e2b_sandbox,
    setup_project,
    deploy_server_and_confirm_ready,
)
from shared.schema import AgentContext
from shared.schema import SandboxContext
from shared.logger import get_logger
from shared.config import config
from shared.integrations.main import get_provider
from langgraph.config import get_stream_writer

logger = get_logger("eval_agent.plan")




async def provision_target_agent(state: EvalAgentPlannerState) -> dict:
    # Skip sandbox provisioning in plan-only mode (not needed for plan generation)
    writer = get_stream_writer()
    if config.eval_plan_only_mode:
        logger.info("plan.provision: Plan-only mode enabled - skipping sandbox provisioning")
        mcp_resources = dict(state.context.mcp_resources or {})
        # Create updated context with mcp_resources but no sandbox
        updated_context = AgentContext(
            user_context=state.context.user_context,
            github_context=state.context.github_context,
            sandbox_context=None,  # No sandbox in plan-only mode
            agent_name=state.context.agent_name,
            target_agent_version=state.context.target_agent_version,
            mcp_services=state.context.mcp_services,
            mcp_resources=mcp_resources,
            functional_requirements=state.context.functional_requirements,
            tool_entries=state.context.tool_entries,
            integrations=state.context.integrations,
            user_id=state.context.user_id,
        )
        return {
            "context": updated_context,
        }
    
    if not state.context.github_context:
        raise ValueError("GitHub context is required to provision the target agent.")

    repo_url = state.context.github_context.repo_url
    branch_name = state.context.github_context.branch_name
    
    # Validate repo_url is not empty
    if not repo_url or not repo_url.strip():
        raise ValueError(
            "GitHub repository URL is required but was not found in the user's message. "
            "Please include a GitHub URL in your request (e.g., 'Evaluate my agent at https://github.com/owner/repo')."
        )
    
    mcp_resources = dict(state.context.mcp_resources or {})

    updates: Dict[str, Any] = {}

    if not state.context.sandbox_context:
        github_token = config.github_token
        logger.info(
            "plan.provision: provisioning sandbox (repo=%s branch=%s)",
            repo_url,
            branch_name,
        )
        env_vars = {
            "COMPOSIO_USER_ID": state.context.user_id,
        }
        writer({"progress":f"Provisioning sandbox (repo={repo_url} branch={branch_name}) ..."})

        sbx, repo_dir, resolved_branch = await initialize_e2b_sandbox(
            repo_url=repo_url,
            branch_name=branch_name,
            github_token=github_token,
            env_vars=env_vars,
        )
        sandbox_branch = resolved_branch or branch_name
        sandbox_id = sbx.sandbox_id
        writer({"progress":f"Sandbox created: {sandbox_id}"})
        writer({"progress":f"Setting up project (repo={repo_dir}) ..."})
        await setup_project(sandbox_id, repo_dir, "pip install -e .")
        writer({"progress":f"Project setup complete"    })
        sandbox, _ = await deploy_server_and_confirm_ready(
            cmd=config.target_agent_command,
            sb=sbx,
            cwd=repo_dir,
        )
        writer({"progress":f"Testing server deployment (cmd={config.target_agent_command}) ..."})

        deployment_url = sandbox.get_host(config.target_agent_port)
        if not deployment_url.startswith("http"):
            deployment_url = f"https://{deployment_url}"
        writer({"progress":f"Server deployment successful: {deployment_url}"})
        logger.info("plan.provision: sandbox ready at %s", deployment_url)

        sandbox_ctx = SandboxContext(
            sandbox_id=sandbox_id,
            working_directory=repo_dir,
            working_branch=sandbox_branch,
        )
        updates["sandbox_context"] = sandbox_ctx
    else:
        logger.info(
            "plan.provision: reusing existing sandbox %s",
            state.context.sandbox_context.sandbox_id,
        )

    # Update the AgentContext with new sandbox and resources
    updated_context = AgentContext(
        user_context=state.context.user_context,
        github_context=state.context.github_context,
        sandbox_context=updates.get("sandbox_context", state.context.sandbox_context),
        agent_name=state.context.agent_name,
        target_agent_version=state.context.target_agent_version,
        mcp_services=state.context.mcp_services,
        mcp_resources=mcp_resources,
        functional_requirements=state.context.functional_requirements,
        tool_entries=state.context.tool_entries,
        integrations=state.context.integrations,
        user_id=state.context.user_id,
    )
    
    return {"context": updated_context}
