

TARGET_AGENT_GUARDRAILS = """

- the agent you are going to develop should be langgraph based agent.
- always use openai models for llm reasoning. To create a react agent prefer create_agent function from langchain.agents.
- To give the agent ability to interact with external services (like asana, github, jira, etc.) use composio tools only , we have already added the COMPOSIO_USER_ID and COMPOSIO_API_KEY in the environment variables.

"""


# TODO: Remove the following hardcoding with mcp tools or graph rag 
COMPOSIO_LANGCHAIN_INTEGRATION = """

# Example implementation showing how can we use composio tools with langchain.

The LangChain Provider transforms Composio tools into a format compatible with LangChain's function calling capabilities.

## Setup
<CodeGroup>
```bash title="Python" for="python"
pip install composio_langchain==0.8.0 langchain
```
</CodeGroup>

## Usage

<CodeGroup>
```python Python title="Python" 
from composio import Composio
from composio_langchain import LangchainProvider
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

### handle error middleware
@wrap_tool_call
async def handle_tool_errors(request, handler):
    "Handle tool execution errors with custom messages."
    try:
        return await handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
### openai client
openai_client = ChatOpenAI(model="gpt-5")

### composio user id
COMPOSIO_USER_ID = os.getenv("COMPOSIO_USER_ID")

composio = Composio(provider=LangchainProvider())

### Get tools from Composio for toolkits
tools = composio.tools.get(
    user_id=COMPOSIO_USER_ID,
    toolkits=["ASANA","GITHUB"],
    limit=1000
)

SYSTEM_PROMPT = '
You are a helpful assistant that can help with tasks related to GitHub and Asana.
'
### Define agent
agent = create_agent(openai_client, tools, middleware=[handle_tool_errors], system_prompt=SYSTEM_PROMPT)

task = "
WHta are the open PRs in the repository?
"
result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})

</CodeGroup>

# Composio Tools
asana and github toolkit contains many useful tools some of them are :
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
    "GITHUB_MERGE_A_PULL_REQUEST",
    "GITHUB_FIND_PULL_REQUESTS",

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
    "ASANA_UPDATE_PROJECT",
    "ASANA_GET_STORIES_FOR_TASK",

"""