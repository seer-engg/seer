from agents.eval_agent.models import TestExecutionState
from shared.logger import get_logger
from shared.integrations.main import get_provider
from coolname import generate_slug
from langgraph.config import get_stream_writer
from langchain_core.messages import ToolMessage
import uuid


logger = get_logger("eval_agent.execute.seed_mcp_resources")


async def seed_mcp_resources(state: TestExecutionState) -> None:
    seed = generate_slug(2)
    updates = {}
    updates["current_seed"] = seed
    updates["mcp_resources"] = state.context.mcp_resources
    writer = get_stream_writer()
    
    for service in state.context.mcp_services:
        provider = await get_provider(service)
        writer({"progress": f"Seeding MCP resources for {service} with seed {seed}"})
        logger.info(f"Seeding MCP resources for {service} with seed {seed}")
        updates["mcp_resources"][service] = provider.persistent_resource
        resources = await provider.provision_resources(seed=seed, user_id=state.context.user_id)
        if resources:
            updates["mcp_resources"][service].update(resources)
        logger.info(f"Seeded MCP resources for {service} with seed {seed}")
    
    output_messages = [ToolMessage(content=f"provisioned resources for {state.context.mcp_services}", tool_call_id=str(uuid.uuid4()))]

    updates["messages"] = output_messages

    return updates