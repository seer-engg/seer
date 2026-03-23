from typing import Optional, Any
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)
from seer.agents.nexus.tool_call_sanitizer import ToolCallSanitizationMiddleware

from seer.config import config
from seer.logger import get_logger
from seer.llm import get_llm
from seer.agents.nexus.utils import get_workflow_tools
from seer.utilities.ml_flow import _ensure_mlflow_autologging

logger = get_logger(__name__)

if config.mlflow_enabled:
    _ensure_mlflow_autologging()


async def _get_memory_context_for_user(user_id: str, current_query: Optional[str] = None) -> str:
    """
    Get formatted memory context for injection into agent system prompt.

    Args:
        user_id: Clerk user_id for memory lookup
        current_query: Optional current query for relevance-based retrieval

    Returns:
        Formatted memory context string, or empty string if unavailable
    """
    try:
        from seer.services.memory import UserMemoryService  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import

        memory_service = UserMemoryService()
        return await memory_service.get_context_for_prompt(
            user_id=user_id,
            current_query=current_query or "",
            max_memories=config.memory_context_max_memories,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Memory is non-critical, must not block agent creation
        logger.warning("Failed to get memory context for user %s: %s", user_id, e)
        return ""

async def create_nexus_chat_agent(
    model: str = "qwen/qwen3-235b-a22b-2507",
    checkpointer: Optional[Any] = None,
    user_id: Optional[str] = None,
    current_query: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> Any:
    """
    Create a LangGraph agent for Nexus chat assistance using create_agent.

    Uses LangChain v1.0+ create_agent with middleware for summarization
    and human-in-the-loop capabilities.

    Args:
        model: Model name to use (e.g., 'qwen/qwen3-235b-a22b-2507', 'google/gemini-2.0-flash-001')
        checkpointer: Optional LangGraph checkpointer for persistence
        user_id: Optional user ID for memory context injection (Clerk user_id)
        current_query: Optional current user query for memory relevance search
        workflow_id: Optional workflow ID for pre-bound workflow tools (e.g., 'wf_abc123')

    Returns:
        LangGraph agent compiled with tools and middleware
    """


    llm = get_llm(model=model, temperature=0)

    # Load base system prompt (modular - detailed guides available via get_workflow_guide())
    from seer.mcp.tools.guides import get_started_impl  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import

    system_prompt = get_started_impl()

    # Inject user memory context if enabled
    if user_id and config.memory_enabled and config.memory_context_injection_enabled:
        memory_context = await _get_memory_context_for_user(user_id, current_query)
        if memory_context:
            # Prepend memory context to system prompt
            system_prompt = memory_context + "\n\n" + system_prompt
            logger.debug("Injected memory context for user %s (%d chars)", user_id, len(memory_context))

    # Get workflow tools (with pre-bound workflow_id if provided)
    tools = get_workflow_tools(workflow_id=workflow_id)

    # Create summarization model (use same model with lower max tokens)
    summarization_model = get_llm(
        model=model,
        temperature=0,
    )

    # Build middleware list
    # ToolCallSanitizationMiddleware runs first to sanitize tool_call_ids
    # before they're processed or stored in checkpoints
    middleware = [
        ToolCallSanitizationMiddleware(),
        SummarizationMiddleware(
            model=summarization_model,
            max_tokens_before_summary=256000/2,  #Model Limit 256k
        ),
    ]

    # Verify checkpointer is provided (required for persistence)
    if checkpointer is None:
        logger.warning("No checkpointer provided to create_nexus_chat_agent - traces will not be persisted")
    else:
        logger.debug("Creating Nexus chat agent with checkpointer: %s", type(checkpointer).__name__)

    # Create agent with middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
    )

    logger.info("Created workflow chat agent with model %s, checkpointer=%s", model, 'enabled' if checkpointer else 'disabled')
    return agent
