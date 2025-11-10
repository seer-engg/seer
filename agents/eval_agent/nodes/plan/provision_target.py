from agents.eval_agent.models import EvalAgentState
from sandbox import initialize_e2b_sandbox, setup_project, TARGET_AGENT_COMMAND, TARGET_AGENT_PORT, deploy_server_and_confirm_ready
from shared.schema import SandboxContext
from shared.logger import get_logger
import os
logger = get_logger("eval_agent.plan")


async def provision_target_agent(state: EvalAgentState) -> dict:
    repo_url = state.github_context.repo_url
    branch_name = state.github_context.branch_name

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

        return {
            "sandbox_context": SandboxContext(
                sandbox_id=sandbox_id,
                working_branch=sandbox_branch,
            ),
        }
    else:
        logger.info("plan.provision: reusing deployment url %s", state.sandbox_context.deployment_url)
        return {}
