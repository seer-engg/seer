from langchain_openai import ChatOpenAI
from shared.mcp_client import ComposioMCPClient
from shared.config import COMPOSIO_USER_ID
from agents.eval_agent.models import TestExecutionState

COMMMON_TOOL_INSTRUCTIONS = """

# Important:
- Asana expects no offset parameter at all on the first page. Sending offset="" (empty string) is treated as an invalid pagination token, so Asana returns:
    offset: Your pagination token is invalid.

- When creating a github repository, do not pass 'team_id' parameter.
"""


llm = ChatOpenAI(
    model="gpt-5",
    # use_responses_api=True,
    # output_version="responses/v1",
    # reasoning={"effort": "medium"},
)


async def get_tools(state: TestExecutionState):
    # Ensure tools initialized (for execution) and entries loaded (for parameter completion)
    tool_service = ComposioMCPClient(["GITHUB", "ASANA"], COMPOSIO_USER_ID)
    tools = await tool_service.get_tools()
    selected_tools = state.tool_selection_log.selected_tools
    actual_tools = [tool for tool in tools if tool.name in selected_tools]
    return actual_tools