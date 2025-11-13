import os
import json
import httpx
import uuid
from typing import Dict
from e2b import AsyncSandbox

from agents.eval_agent.models import EvalAgentState
from sandbox import initialize_e2b_sandbox, setup_project, TARGET_AGENT_COMMAND, TARGET_AGENT_PORT, deploy_server_and_confirm_ready
from shared.schema import SandboxContext
from shared.logger import get_logger
from shared.mcp_client import get_mcp_client_and_configs
from langchain_core.tools import BaseTool

logger = get_logger("eval_agent.plan")


async def provision_target_agent(state: EvalAgentState) -> dict:
    if not state.github_context:
        raise ValueError("GitHub context is required to provision the target agent.")
    
    repo_url = state.github_context.repo_url
    branch_name = state.github_context.branch_name
    mcp_resources = {}

    if not state.sandbox_context:
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
        
        if state.mcp_services:
            logger.info(f"Provisioning MCP resources for: {state.mcp_services}")
            mcp_client, mcp_configs = await get_mcp_client_and_configs(state.mcp_services)
            mcp_tools = await mcp_client.get_tools()
            tools_dict: Dict[str, BaseTool] = {t.name: t for t in mcp_tools}

            try:
                # 1. Create Asana Project
                if "asana.create_project" in tools_dict:
                    project_name = f"Seer Eval - {state.github_context.agent_name} - {uuid.uuid4().hex[:6]}"
                    logger.info(f"Creating Asana project: {project_name}")
                    asana_project = await tools_dict["asana.create_project"].ainvoke({"name": project_name})
                    mcp_resources["asana_project"] = asana_project # Store the whole object
                    logger.info(f"Created Asana project ID: {asana_project.get('id')}")

                # 2. Create GitHub Repo
                if "github.create_repo" in tools_dict:
                    repo_name = f"seer-eval-{state.github_context.agent_name}-{uuid.uuid4().hex[:6]}"
                    logger.info(f"Creating GitHub repo: {repo_name}")
                    # Assumes create_repo takes a name and returns the repo object
                    github_repo = await tools_dict["github.create_repo"].ainvoke({"name": repo_name, "private": True})
                    mcp_resources["github_repo"] = github_repo # Store the whole object
                    logger.info(f"Created GitHub repo: {github_repo.get('full_name')}")

            except Exception as e:
                logger.error(f"Failed to provision MCP resources: {e}")
                # We might want to raise an error here to stop the run
                raise
        # --- End MCP Logic ---

        return {
            "sandbox_context": SandboxContext(
                sandbox_id=sandbox_id,
                working_directory=repo_dir,
                working_branch=sandbox_branch,
            ),
            "mcp_resources": mcp_resources,
        }
    else:
        logger.info("plan.provision: reusing existing sandbox %s", state.sandbox_context.sandbox_id)
        # We still need to re-provision MCP resources for the new experiment
        mcp_resources = {}
        if state.mcp_services:
            logger.info(f"Provisioning MCP resources for: {state.mcp_services}")
            mcp_client, mcp_configs = await get_mcp_client_and_configs(state.mcp_services)
            mcp_tools = await mcp_client.get_tools()
            tools_dict: Dict[str, BaseTool] = {t.name: t for t in mcp_tools}
                
            try:
                # (Same as above) Create Asana Project
                if "asana.create_project" in tools_dict:
                    project_name = f"Seer Eval - {state.github_context.agent_name} - {uuid.uuid4().hex[:6]}"
                    asana_project = await tools_dict["asana.create_project"].ainvoke({"name": project_name})
                    mcp_resources["asana_project"] = asana_project
                    logger.info(f"Created Asana project ID: {asana_project.get('id')}")

                # (Same as above) Create GitHub Repo
                if "github.create_repo" in tools_dict:
                    repo_name = f"seer-eval-{state.github_context.agent_name}-{uuid.uuid4().hex[:6]}"
                    github_repo = await tools_dict["github.create_repo"].ainvoke({"name": repo_name, "private": True})
                    mcp_resources["github_repo"] = github_repo
                    logger.info(f"Created GitHub repo: {github_repo.get('full_name')}")
                
            except Exception as e:
                logger.error(f"Failed to provision MCP resources: {e}")
                raise

        return {"mcp_resources": mcp_resources}
