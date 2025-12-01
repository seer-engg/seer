from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.integrations.main import get_provider

logger = get_logger("eval_agent.execute.clean_mcp_resources")


async def clean_mcp_resources(state: TestExecutionState) -> None:   
    for service in state.context.mcp_services:
        provider = await get_provider(service)
        logger.info(f"Cleaning up MCP resources for {service} with seed {state.current_seed}")
        await provider.cleanup_resources(resources=state.mcp_resources[service])
        logger.info(f"Cleaned up MCP resources for {service} with seed {state.current_seed}")
    return {}