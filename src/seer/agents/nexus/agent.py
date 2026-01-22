from typing import Optional, Dict, Any
from seer.logger import get_logger
from seer.config import config

logger = get_logger(__name__)


def create_nexus_chat_agent(
    model: str = "gpt-4o-mini",
    checkpointer: Optional[Any] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a LangGraph agent for Nexus chat assistance using supervisor architecture.

    Uses multi-agent supervisor architecture with specialized agents:
    - tool_discovery: Searches for relevant tools
    - trigger_discovery: Searches for relevant triggers
    - workflow_architect: Designs complete workflow structure
    - validation: Validates and submits workflow spec

    Args:
        model: Model name to use (e.g., 'gpt-4o-mini', 'gpt-4o')
        checkpointer: Optional LangGraph checkpointer for persistence
        workflow_state: Optional existing workflow state for editing

    Returns:
        LangGraph agent compiled with supervisor architecture
    """
    if not config.supervisor_mode_enabled:
        logger.warning(
            "Supervisor mode is disabled but required for Nexus. "
            "Forcing supervisor mode on. Set SUPERVISOR_MODE_ENABLED=true in config."
        )

    logger.info("Creating Nexus agent with supervisor multi-agent architecture")
    from seer.agents.nexus.supervisor.graph import create_supervisor_graph  # pylint: disable=import-outside-toplevel  # Conditional import for optional feature

    return create_supervisor_graph(model, checkpointer, workflow_state)
