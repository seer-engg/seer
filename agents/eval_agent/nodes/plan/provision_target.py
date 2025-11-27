import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI

from agents.eval_agent.models import EvalAgentPlannerState
from sandbox import (
    initialize_e2b_sandbox,
    setup_project,
    TARGET_AGENT_COMMAND,
    TARGET_AGENT_PORT,
    deploy_server_and_confirm_ready,
)
from shared.agent_context import AgentContext
from shared.schema import SandboxContext
from shared.logger import get_logger

logger = get_logger("eval_agent.plan")



def _seed_default_resources(mcp_resources: Dict[str, Any]) -> None:
    workspace_id = os.getenv("ASANA_WORKSPACE_ID") or os.getenv("ASANA_DEFAULT_WORKSPACE_GID")
    if workspace_id and "asana_workspace" not in mcp_resources:
        mcp_resources["asana_workspace"] = {"id": workspace_id, "gid": workspace_id}
        logger.info("Seeded Asana workspace from environment: %s", workspace_id)

    project_id = os.getenv("ASANA_PROJECT_ID") or os.getenv("ASANA_DEFAULT_PROJECT_GID")
    if project_id and "asana_project" not in mcp_resources:
        mcp_resources["asana_project"] = {"id": project_id, "gid": project_id}
        logger.info("Seeded Asana project from environment: %s", project_id)



async def provision_target_agent(state: EvalAgentPlannerState) -> dict:
    if not state.context.github_context:
        raise ValueError("GitHub context is required to provision the target agent.")

    repo_url = state.context.github_context.repo_url
    branch_name = state.context.github_context.branch_name
    mcp_resources = dict(state.context.mcp_resources or {})
    _seed_default_resources(mcp_resources)

    updates: Dict[str, Any] = {}

    if not state.context.sandbox_context:
        github_token = os.getenv("GITHUB_TOKEN")
        logger.info(
            "plan.provision: provisioning sandbox (repo=%s branch=%s)",
            repo_url,
            branch_name,
        )

        sbx, repo_dir, resolved_branch = await initialize_e2b_sandbox(
            repo_url=repo_url,
            branch_name=branch_name,
            github_token=github_token,
        )
        sandbox_branch = resolved_branch or branch_name
        sandbox_id = sbx.sandbox_id

        await setup_project(sandbox_id, repo_dir, "pip install -e .")

        sandbox, _ = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=repo_dir,
        )

        deployment_url = sandbox.get_host(TARGET_AGENT_PORT)
        if not deployment_url.startswith("http"):
            deployment_url = f"https://{deployment_url}"

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
        target_agent_version=state.context.target_agent_version,
        mcp_services=state.context.mcp_services,
        mcp_resources=mcp_resources,
    )
    
    return {"context": updated_context}
