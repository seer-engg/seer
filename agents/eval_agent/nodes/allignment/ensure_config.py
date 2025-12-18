import re
from typing import Optional, Tuple, List 
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from agents.eval_agent.models import EvalAgentPlannerState
from shared.schema import AgentContext
from shared.schema import GithubContext, UserContext
from shared.logger import get_logger
from shared.tools import resolve_mcp_services
from shared.config import config
from langchain_openai import ChatOpenAI
import uuid

logger = get_logger("eval_agent.plan")

async def ensure_target_agent_config(state: EvalAgentPlannerState) -> dict:
    last_human = None
    input_context = state.input_context
    for msg in reversed(state.messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            last_human = msg
            break
    if last_human is None:
        raise ValueError("No human message to extract from")
    try:
        human_message = last_human.content[0].get('text', '')
    except Exception as e:
        human_message = last_human.content

    instruction = """
        Extract the following fields from the user's latest message about the target agent:
        - agent_name: the name of the agent
        - mcp_services: A list of external service names mentioned (e.g., 'asana', 'github', 'jira'). Return an empty list if none are mentioned.
    """

    class TargetAgentExtractionContext(BaseModel):
        """Context for extracting the target agent's GitHub and user context."""
        mcp_services: List[str] = Field(
            default_factory=list, 
            description="List of external MCP services mentioned, e.g., ['asana', 'github']"
        )
        agent_name: str = Field(
            default="agent",
            description="The name of the agent"
        )

    # using gpt-5-mini without response api  to avoid json schema error 
    # TODO: find out how to adapt schea with responses api
    extractor = ChatOpenAI(model="gpt-5-nano", temperature=0.0).with_structured_output(TargetAgentExtractionContext)
    context: TargetAgentExtractionContext = await extractor.ainvoke(f"{instruction}\n\nUSER:\n{human_message}")
    
    resolved_services = resolve_mcp_services(context.mcp_services)
    logger.info(
        f"Resolved MCP services (requested={context.mcp_services}): {resolved_services}"
    )
    
    # Create or update the AgentContext
    agent_context = state.context if state.context else AgentContext()
    
    # Update the context with extracted values
    # Ensure github_context is set (should be set above, but handle None case)
    github_ctx = GithubContext(
        repo_url=f"https://github.com/{input_context.get('integrations', {}).get('github', {}).get('name')}",
        branch_name="main"
    )

    user_context = UserContext(
        raw_request=human_message
    )
    updated_context = AgentContext(
        user_context=user_context,
        github_context=github_ctx,
        sandbox_context=agent_context.sandbox_context,  # Preserve existing sandbox
        agent_name=context.agent_name,
        target_agent_version=agent_context.target_agent_version,
        mcp_services=resolved_services,
        mcp_resources=agent_context.mcp_resources,
        functional_requirements=agent_context.functional_requirements,
        tool_entries=agent_context.tool_entries,
        integrations=input_context.get('integrations', agent_context.integrations),
        user_id=input_context.get('user_id') or config.user_id , 
    )

    output_messages = [ToolMessage(content=context, tool_call_id=str(uuid.uuid4()))]

    return {
        "context": updated_context,
        "messages": output_messages,
    }
