from agents.eval_agent.models import EvalAgentPlannerState, ToolSelectionLog
from shared.logger import get_logger
from shared.tools import ToolEntry
from typing import List, Dict
from shared.tool_service import get_tool_service


logger = get_logger("eval_agent.plan.filter_tools")

AVAILABLE_TOOLS = [
    "GITHUB_ACTIVITY_STAR_REPO_FOR_AUTHENTICATED_USER",
    "GITHUB_CHECK_IF_A_PULL_REQUEST_HAS_BEEN_MERGED",
    "GITHUB_CREATE_A_BLOB",
    "GITHUB_CREATE_A_COMMIT",
    "GITHUB_CREATE_A_COMMIT_COMMENT",
    "GITHUB_CREATE_AN_ISSUE",
    "GITHUB_CREATE_A_PULL_REQUEST",
    "GITHUB_CREATE_A_REFERENCE",
    "GITHUB_GET_A_REFERENCE",
    "GITHUB_GET_A_COMMIT",
    "GITHUB_CREATE_OR_UPDATE_FILE_CONTENTS",

    "ASANA_ADD_TASK_TO_SECTION",
    "ASANA_CREATE_A_PROJECT",
    "ASANA_CREATE_A_TASK",
    "ASANA_CREATE_CUSTOM_FIELD",
    "ASANA_CREATE_PROJECT_STATUS_UPDATE",
    "ASANA_CREATE_SECTION_IN_PROJECT",
    "ASANA_CREATE_TASK_COMMENT",
    "ASANA_GET_A_PROJECT",
    "ASANA_GET_A_TASK",
    "ASANA_UPDATE_A_TASK",
    "ASANA_UPDATE_PROJECT"
]


async def filter_tools(state: EvalAgentPlannerState) -> dict:
    """Filter the tools for the test generation."""

    available_tools: List[str] = []
    tool_entries: Dict[str, ToolEntry] = {}


    # We'll initialize context_for_scoring here to ensure it's always defined
    context_for_scoring = ""
    for example in state.dataset_examples:
        context_for_scoring += ",".join(example.expected_output.create_test_data) + "\n" + ",".join(example.expected_output.assert_final_state)

    # if state.context.mcp_services:
    #     try:
    #         tool_service = get_tool_service()
    #         await tool_service.initialize(state.context.mcp_services)
    #         tool_entries = tool_service.get_tool_entries()
            
    #         prioritized = await tool_service.select_relevant_tools(
    #             context_for_scoring,
    #             max_total=20,  # Increased from 20 to accommodate multiple services
    #             max_per_service=10,  # Increased from 10 to ensure all relevant tools per service
    #         )
            
    #         # Convert tools to names
    #         available_tools = [tool.name for tool in prioritized]
    #         if not available_tools:
    #             available_tools = sorted({entry.name for entry in tool_entries.values()})
                
    #         logger.info(
    #             "Found %d prioritized MCP tools for test generation.",
    #             len(available_tools),
    #         )
    #     except Exception as exc:
    #         logger.error(f"Failed to load MCP tools for test generation: {exc}")
    # else:
    #     logger.info("No MCP services configured; tool prompts will be limited to system tools.")
    
    log_entry = ToolSelectionLog(
        selection_context=context_for_scoring,
        selected_tools=AVAILABLE_TOOLS
    )

    return {
        "available_tools": available_tools,
        "tool_selection_log": log_entry,
        "tool_entries": tool_entries,
    }