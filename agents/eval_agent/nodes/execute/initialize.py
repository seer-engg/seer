from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.parameter_population import extract_all_context_variables


logger = get_logger("eval_agent.execute.initialize")


async def initialize_node(state: TestExecutionState) -> dict:

    updates: dict = {}

    # Initialize pending list and accumulator on first entry
    pending = list(state.dataset_examples or [])
    updates["pending_examples"] = pending
    updates["accumulated_results"] = []

    # Enrich mcp_resources once using context variables (if github context is present)
    enriched_resources = dict(state.context.mcp_resources or {})
    if state.context.github_context and state.context.github_context.repo_url:
        context_vars = extract_all_context_variables(
            user_context=state.context.user_context,
            github_context=state.context.github_context,
            mcp_resources=enriched_resources,
        )
        if "github_owner" in context_vars:
            enriched_resources["github_owner"] = {"id": context_vars["github_owner"]}
            logger.info(f"Added github_owner to mcp_resources: {context_vars['github_owner']}")
        if "github_repo" in context_vars:
            enriched_resources["github_repo"] = {"id": context_vars["github_repo"]}
            logger.info(f"Added github_repo to mcp_resources: {context_vars['github_repo']}")
    updates["mcp_resources"] = enriched_resources

    return updates