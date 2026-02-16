from typing import Optional, Dict, Any
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)

from seer.config import config
from seer.logger import get_logger
from seer.llm import get_llm
from seer.agents.nexus.utils import get_workflow_tools
from seer.agents.nexus.schema_context import (
    get_workflow_spec_schema_text,
    generate_primitive_blocks_guide,
    generate_graph_structure_guide,
)
from seer.prompts import get_nexus_system_prompt
from seer.utilities.ml_flow import _ensure_mlflow_autologging
logger = get_logger(__name__)

WORKFLOW_SPEC_SCHEMA = get_workflow_spec_schema_text()

if config.mlflow_enabled:
    _ensure_mlflow_autologging()

def create_nexus_chat_agent(
    model: str = "moonshotai/kimi-k2.5",
    checkpointer: Optional[Any] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a LangGraph agent for Nexus chat assistance using create_agent.

    Uses LangChain v1.0+ create_agent with middleware for summarization
    and human-in-the-loop capabilities.

    Args:
        model: Model name to use (e.g., 'moonshotai/kimi-k2.5', 'moonshotai/kimi-k2-thinking')
        checkpointer: Optional LangGraph checkpointer for persistence

    Returns:
        LangGraph agent compiled with tools and middleware
    """


    llm = get_llm(model=model, temperature=0)

    # System prompt for the workflow assistant
    # Load base system prompt from markdown file
    base_system_prompt = get_nexus_system_prompt()

    # Add dynamic content sections (schema always injected; templates/examples available via tools)
    schema_section = f"\n\nWorkflowSpec schema excerpt (trimmed):\n{WORKFLOW_SPEC_SCHEMA}"
    blocks_guide = f"\n\n{generate_primitive_blocks_guide()}"
    graph_guide = f"\n\n{generate_graph_structure_guide()}"

    # Compose full system prompt from loaded base + dynamic content
    system_prompt = base_system_prompt + blocks_guide + graph_guide + schema_section

    # Get workflow tools (with optional workflow_state injection)
    tools = get_workflow_tools(workflow_state=workflow_state)

    # Create summarization model (use same model with lower max tokens)
    summarization_model = get_llm(
        model=model,
        temperature=0,
    )

    # Build middleware list
    middleware = [
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
