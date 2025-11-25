

TARGET_AGENT_GUARDRAILS = """

- the agent you are going to develop should be langgraph based agent.
- always use openai models for llm reasoning. To create a react agent prefer create_agent function from langchain.agents.
- To give the agent ability to interact with external services (like asana, github, jira, etc.) use composio tools only , we have already added the COMPOSIO_USER_ID and COMPOSIO_API_KEY in the environment variables.

"""


# TODO: Remove the following hardcoding with mcp tools or graph rag 
COMPOSIO_LANGCHAIN_INTEGRATION = """

# example implementation showing how can we use composio tools with langchain.

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

### Get tools from Composio (include docs search)
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
Provision the environment for the target agent based on the instructions provided.
Resorces:
- asana_project: id=1211928407052666 (use [resource:asana_project.id])
- asana_workspace: id=1211928405122978 (use [resource:asana_workspace.id])
- github_owner: id=seer-engg (use [resource:github_owner.id])
- github_repo: id=buggy-coder (use [resource:github_repo.id])

SHORT_UUID = e1a2b3c4
Instructions:
Create a Pull Request in seer-engg/label-edgecase-repo from "buggy-coder/test-sync-e1a2b3c4" into the default branch.\n      * PR title: "Sync Asana task  - test"\n     * PR body must contain exactly the text: "Asana Task:" 
"


result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})

</CodeGroup>

"""