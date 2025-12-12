"""Context and plan step"""
from __future__ import annotations
from pathlib import Path


from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool
from langchain.tools import ToolRuntime

from shared.logger import get_logger
from shared.config import config
from shared.tools import search_composio_documentation, web_search, search_langchain_documentation, think

from agents.codex.state import CodexState
from shared.llm import get_agent_final_respone

from sandbox.tools import (
    run_command,    
    inspect_directory,
    read_files,
    grep,
    SandboxToolContext,
    create_file,
    create_directory,
    write_file,
    edit_file,
)

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents import create_agent
from agents.eval_agent.nodes.execute.utils import get_tool_hub
from shared.tools import ToolEntry
logger = get_logger("codex.nodes.developer")


SYSTEM_PROMPT_PATH = Path(__file__).parent / "developer_prompt.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

CODEBASE_VIEW_TOOLS = [
    inspect_directory, read_files,
    grep, 
    # TODO: code chunking tools
    # search_code, search_symbols, semantic_search, get_symbol_definition, find_usages, get_code_region
]

CODEBASE_EDIT_TOOLS = [
    create_file, create_directory, write_file, edit_file
]

USER_PROMPT = """
# CONTEXT

Here are the evaluation test cases and the failed thread traces from the target agent:

<eval_data>
{evals_and_thread_traces}
</eval_data>

Here is what the user expects from the target agent:

<user_expectations>
{user_raw_request}
</user_expectations>

Develop the agent to pass all these evaluation test cases.

"""

EVALS_AND_THREAD_TRACE_TEMPLATE = """
    
    <EVAL> 
    {eval}
    </EVAL>
    <THREAD TRACE>
    {thread_trace}
    </THREAD TRACE>
"""

# Intentionaly left blank to use the developer_prompt.md file as system prompt.
TODOS_TOOL_SYSTEM_PROMPT = " "

llm = ChatOpenAI(model="gpt-5.1-codex-mini", reasoning={"effort": "high"})
low_reasoning_llm = ChatOpenAI(model="gpt-5.1-codex-mini", reasoning={"effort": "medium"})
medium_reasoning_llm = ChatOpenAI(model="gpt-5-mini", reasoning={"effort": "medium"})



CODEBASE_EXPLORER_SYSTEM_PROMPT = """
You are a helpful  coding assistant that can help explore the codebase and return detailed necessary information.

You should carefully consider every aspect of the querry, thoroughly inspect the codebase and return the relevant information.

while returning the information don't miss any details, include file paths, code snippets, etc.
"""

codebase_explorer_subgraph = create_agent(
    model=low_reasoning_llm,
    tools=[
        *CODEBASE_VIEW_TOOLS,
        web_search,
        search_composio_documentation,
        search_langchain_documentation,
    ],
    system_prompt=CODEBASE_EXPLORER_SYSTEM_PROMPT,
    context_schema=SandboxToolContext,
)

@tool
async def codebase_explorer_subagent(query: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Explore the codebase and return the relevant information. This tool is an agent that can be used to explore the codebase and return the relevant information. 
    Capabilities: 
        - It has read only access to the codebase, 
        - web search 
        - composio documentation tools.
    Args:
        query: The query to explore the codebase.
    Returns:
        The relevant information from the codebase.
    """
    result = await codebase_explorer_subgraph.ainvoke({"messages": [HumanMessage(content=query)]}, config=RunnableConfig(context=runtime.context))
    return await get_agent_final_respone(result)



EXPERIMENTATION_SYSTEM_PROMPT = """
You are a SCIENTIFIC INVESTIGATIVE SOFTWARE ENGINEER working on a langchain/langgraph based agent. You will receive a user request and must approach it with rigorous scientific methodology, skepticism, and evidence-based reasoning.

## Core Scientific Principles
- TRUST NOTHING: Question every assumption, claim, and piece of existing code
- HYPOTHESIS-DRIVEN: Form explicit hypotheses before any investigation
- EVIDENCE-BASED: Every claim must be backed by experimental results or empirical evidence
- FALSIFIABLE: Design experiments that can prove or disprove your hypotheses
- REPRODUCIBLE: Document all experiments so they can be replicated
- ITERATIVE: Expect to refine hypotheses based on experimental results

## Investigation Framework
You have access to CLI commands and can create experiments. For every investigation:

1. **Form Hypothesis**: State what you believe is true and why
2. **Design Experiment**: Determine if you need:
   - Simple CLI command verification (for quick checks)
   - Full experimental setup in experiments folder (for complex testing)
3. **Execute & Document**: Run the experiment and record results
4. **Analyze Results**: Determine if hypothesis is supported or refuted
5. **Iterate**: Refine understanding based on evidence

## Experimental Guidelines
- Create experiments in an "experiments" folder (create if it doesn't exist)
- Each experiment gets its own subfolder with descriptive name
- Use CLI commands for quick verifications (file existence, simple tests, etc.)
- Use full experimental setups for complex behavior testing
- Document methodology, expected results, and actual results
- If results contradict assumptions, explicitly acknowledge and adjust

"""

experimantation_subgraph = create_agent(
    model=llm,
    tools=[
        *CODEBASE_VIEW_TOOLS,
        *CODEBASE_EDIT_TOOLS,
        run_command,
        search_composio_documentation,
        web_search,

    ],
    system_prompt=EXPERIMENTATION_SYSTEM_PROMPT,
    context_schema=SandboxToolContext,
)

@tool
async def junior_programmer_subagent(query: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Experiment with the codebase, implement small scale coding tasks and return the relevant results. This tool is an agent that can be used to experiment with the codebase or implement small scale coding tasks and return the relevant information.
    Capabilities:
        - Read and Write access to codebase 
        - run commands in the working directory of codebase
        - web search 
        - composio documentation tools.

    Args:
        query: The query to experiment with the codebase.
    Returns:
        The relevant results from the experiment.
    """
    result = await experimantation_subgraph.ainvoke({"messages": [HumanMessage(content=query)]}, config=RunnableConfig(context=runtime.context))
    return await get_agent_final_respone(result)


SEARCH_DOCUMENTATION_SYSTEM_PROMPT = """
You are a helpful python  assistant that can help search the relevant information for python.

You should carefully consider every aspect of the querry.

# CONTEXT:
Composio is a pythan package that offers langchain based tools for all external services like github, asana, etc. Composio provides a langchain provider to fetch tools that can be directly used with langchain/langgraph agents . There are thousands of tools avialble in composio , you can search for tools using search_composio_tools tool.
```python
# composio user id
COMPOSIO_USER_ID = os.getenv("COMPOSIO_USER_ID")

composio = Composio(provider=LangchainProvider())

# Get tools from Composio
tools = composio.tools.get(
    user_id=COMPOSIO_USER_ID,
    tools=[
        "ASANA_CREATE_A_TASK",
        "ASANA_CREATE_TASK_COMMENT"
    ],
)

```


# Important:
- while returning the information don't miss any details, include code snippets, etc.
- Do not include any web urls in the response.
- Only search for python documentations and in your response only include python code snippets, not any other code snippets.

# TOOL USAGE:
- search_composio_tools: Use this tool to search for tools from composio.
- get_tool_schema_from_composio: use this tool to get the schema of a specific tool from composio.
- search_composio_documentation: Use this tool to search general documentation from composio about how we integrate composio tools with the codebase. DO NOT use this to search for composio tools.
- web_search: Use this tool to search the web .
- search_langchain_documentation: Use this tool to search the langchain/langgraph specific documentation(this does not include composio documentation).
- think: Use this tool to think about the current task/plan.


"""
from shared.tools import ComposioMCPClient

import asyncio
@tool()
async def get_tool_schema_from_composio(tool_names: list[str]) -> dict:
    """
    Tool to get the schema of a specific tools from composio.
    Args:
        tool_names: The names of the tools to get the schema for.
    Returns:
        The schema of the tools in a dictionary format.
    """
    #TODO: replcae with tool hub search 
    # tool_service = ComposioMCPClient([], config.composio_user_id)
    # client = tool_service.get_client()
    # tools = await asyncio.to_thread(client.tools.get,
    #     user_id=config.composio_user_id,
    #     tools=tool_names,
    # )
    # tools  = [{'name': t.name,'schema': t.args_schema.model_json_schema()} for t in tools]
    tools = []
    return tools

from typing import Dict

@tool()
async def search_composio_tools(query_text: str) -> Dict[str, ToolEntry]:
    """
    Tool to search for tools from composio. Use when you have to find tools in composio for specific intents.

    Args:
        query_text: The query to search the tools for.
    Returns:
        The relevant tools from composio.
    """
    hub = await get_tool_hub()
    relevant_tool_dicts = await asyncio.to_thread(hub.query, query_text, top_k=20)
    
    # 4. Convert to ToolEntry format expected by the agent state
    tool_entries: Dict[str, ToolEntry] = {}
    for t_dict in relevant_tool_dicts:
        name = t_dict.get("name")
        if not name:
            continue
            
        # Infer service from name (e.g. 'github_create_issue' -> 'github')
        # This is a heuristic; ideally ToolHub would return this metadata
        service = name.split("_")[0] if "_" in name else "general"
        
        tool_entries[name] = ToolEntry(
            name=name,
            description=t_dict.get("description", ""),
            service=service,
        )
    return str(tool_entries)

SEARCH_DOCUMENTATION_USER_PROMPT = """
# QUERIES:
 {query_list}

Return the relevant information for each query in the list. Do not include any web urls in the response.
"""

search_documentation_subgraph = create_agent(
    model=medium_reasoning_llm,
    tools=[
        search_composio_documentation,
        web_search,
        search_composio_tools,
        get_tool_schema_from_composio,
        search_langchain_documentation,
        think,
    ],
    system_prompt=SEARCH_DOCUMENTATION_SYSTEM_PROMPT,
    # NO tool is using runtime context, so we don't need to set the context schema
    # context_schema=SandboxToolContext,
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)

from pydantic import BaseModel, Field
from typing import Literal

class SearchQuery(BaseModel):
    """A search query."""
    query: str = Field(description="Detailed search query to search the documentation for. Write detailed queries that you have, instead of few keywords.")
    meta_context: str = Field(description="The meta context of the search. Why you are searching for this information.")
    expected_output: str = Field(description="The expected output of the search. What do you expect to find in the documentation.")

class SearchDocumentationInput(BaseModel):
    """Input for weather queries."""
    query_list: list[SearchQuery] = Field(description="List of detailed queries to search the documentation for.")

@tool(args_schema=SearchDocumentationInput)
async def search_documentation_subagent(query_list: list[SearchQuery]) -> str:
    """
    Tool to search for python package documentations (langchain, composio, etc), composio tools for external services. Use this when you need to find information about a specific python package's implementation details, usage, etc.  Use it to search for specific tools available from composio for each external service.
    
    Args:
        query_list: List of detailed queries to search the documentation for. Write detailed queries that you have, instead of few keywords.
    Returns:
        The relevant information from the documentation.
    """
    user_prompt = SEARCH_DOCUMENTATION_USER_PROMPT.format(query_list=query_list)
    # not invoked with runtime context, since no tools are using it 
    result = await search_documentation_subgraph.ainvoke({"messages": [HumanMessage(content=user_prompt)]})
    return await get_agent_final_respone(result)

PACKAGE_INSTALLER_SYSTEM_PROMPT = """
You are a helpful python package installer that can help install python packages in the codebase.

"""
package_installer_subgraph = create_agent(
    model=low_reasoning_llm,
    tools=[
        run_command,
        web_search,
    ],
    system_prompt=PACKAGE_INSTALLER_SYSTEM_PROMPT,
    context_schema=SandboxToolContext,
)

PACKAGE_INSTALLER_USER_PROMPT = """
Install the following packages: {packages}
"""
@tool()
async def package_installer(packages: list[str], runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Tool to install python packages in the codebase.
    Args:
        packages: The list of python packages to install.
    Returns:
        The output of the package installation.
    """
    user_prompt = PACKAGE_INSTALLER_USER_PROMPT.format(packages=packages)
    result = await package_installer_subgraph.ainvoke({"messages": [HumanMessage(content=user_prompt)]}, config=RunnableConfig(context=runtime.context))
    return await get_agent_final_respone(result)

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
    model=llm,
    tools=[
        codebase_explorer_subagent,
        *CODEBASE_VIEW_TOOLS,
        *CODEBASE_EDIT_TOOLS,        
        search_documentation_subagent,
        package_installer,
        think,  
    ],
    system_prompt=SYSTEM_PROMPT,
    context_schema=SandboxToolContext,  # Add context schema for sandbox tools
    middleware=[
        TodoListMiddleware(system_prompt=TODOS_TOOL_SYSTEM_PROMPT),
        # TODO: Add summarization middleware
        # SummarizationMiddleware(
        #     model="gpt-5-mini",
        #     trigger=("tokens",300000)
        # ),
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)

async def developer(state: CodexState) -> CodexState:
    """Single ReAct agent that gathers repo context and returns a concrete plan."""

    # Extract sandbox context for tools
    sandbox_context = state.context.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")

    input_messages = list[BaseMessage](state.developer_thread or [])

    # Pass context along with state
    result = await agent.ainvoke(
        input={"messages": input_messages},
        config=RunnableConfig(recursion_limit=150),
        context=SandboxToolContext(sandbox_context=sandbox_context)  # Pass sandbox context
    )
    logger.info(f"developer completed successfully")

    return {
        "developer_thread": result.get("messages"),
    }
