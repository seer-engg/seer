from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.integrations.main import get_provider
from coolname import generate_slug

logger = get_logger("eval_agent.execute.seed_mcp_resources")


async def seed_mcp_resources(state: TestExecutionState) -> None:
    seed = generate_slug(2)
    updates = {}
    updates["current_seed"] = seed
    updates["mcp_resources"] = state.context.mcp_resources
    
    for service in state.context.mcp_services:
        provider = await get_provider(service)
        logger.info(f"Seeding MCP resources for {service} with seed {seed}")
        resources = await provider.provision_resources(seed=seed)
        if resources:
            updates["mcp_resources"][service].update(resources)
        logger.info(f"Seeded MCP resources for {service} with seed {seed}")

    return updates