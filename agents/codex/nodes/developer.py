"""Context and plan step"""
from __future__ import annotations
from pathlib import Path


from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.tools import tool
from langchain.tools import ToolRuntime

from shared.logger import get_logger
from shared.config import config
from shared.tools import LANGCHAIN_DOCS_TOOLS, search_composio_documentation, web_search

from agents.codex.state import CodexState
from agents.codex.format_thread import fetch_thread_timeline_as_string
from agents.codex.utils import get_agent_final_respone

from sandbox.tools import (
    run_command,    
    inspect_directory,
    read_file,
    grep,
    search_code,
    search_symbols,
    semantic_search,
    get_symbol_definition,
    find_usages,
    get_code_region,
    SandboxToolContext,
    create_file,
    create_directory,
    write_file,
    edit_file,
)

logger = get_logger("codex.nodes.developer")


SYSTEM_PROMPT_PATH = Path(__file__).parent / "developer_prompt.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

CODEBASE_VIEW_TOOLS = [
    inspect_directory, read_file, grep, search_code, search_symbols, semantic_search, get_symbol_definition, find_usages, get_code_region
]

CODEBASE_EDIT_TOOLS = [
    create_file, create_directory, write_file, edit_file
]

USER_PROMPT = """
    user exectation with the agent is : {user_raw_request}
    Analyse the following eval test cases and corresponding  thread trace of the agent .
    <EVALS AND THREAD TRACES>
    {evals_and_thread_traces}
    </EVALS AND THREAD TRACES>

    Develop the agent to pass all the eval cases.
"""

EVALS_AND_THREAD_TRACE_TEMPLATE = """
    
    <EVAL> 
    {eval}
    </EVAL>
    <THREAD TRACE>
    {thread_trace}
    </THREAD TRACE>
"""

TODOS_TOOL_SYSTEM_PROMPT = """
## `write_todos`
You have access to the `write_todos` tool to help you manage and plan agent development objectives into smaller steps.

-It is critical that you mark todos as completed as soon as you are done with a step.
-For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.
"""

llm = ChatOpenAI(model="gpt-5-codex", reasoning={"effort": "high"})


CODEBASE_EXPLORER_SYSTEM_PROMPT = """
You are a helpful  coding assistant that can help explore the codebase and return detailed necessary information.

You should carefully consider every aspect of the querry, thoroughly inspect the codebase and return the relevant information.

while returning the information don't miss any details, include file paths, code snippets, etc.
"""

codebase_explorer_subgraph = create_agent(
    model=llm,
    tools=[
        *CODEBASE_VIEW_TOOLS,
        web_search,
        search_composio_documentation,
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
    return get_agent_final_respone(result)



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
async def experimentaion_subagent(query: str, runtime: ToolRuntime[SandboxToolContext]) -> str:
    """
    Experiment with the codebase and return the relevant results. This tool is an agent that can be used to experiment with the codebase and return the relevant information.
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
    return get_agent_final_respone(result)

agent = create_agent(
    model=llm,
    tools=[
        run_command,
        web_search,
        codebase_explorer_subagent,
        experimentaion_subagent,
        *CODEBASE_VIEW_TOOLS,
        *CODEBASE_EDIT_TOOLS,
        *LANGCHAIN_DOCS_TOOLS,
        search_composio_documentation,
    ],
    system_prompt=SYSTEM_PROMPT,
    context_schema=SandboxToolContext,  # Add context schema for sandbox tools
    middleware=[
        TodoListMiddleware(system_prompt=TODOS_TOOL_SYSTEM_PROMPT),
    ],
)

async def developer(state: CodexState) -> CodexState:
    """Single ReAct agent that gathers repo context and returns a concrete plan."""

    # Extract sandbox context for tools
    sandbox_context = state.context.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")
    user_raw_request = state.context.user_context.raw_request
    
    experiment_results = state.experiment_context.results


    input_messages = list[BaseMessage](state.developer_thread or [])

    if not state.latest_results:
        evals_and_thread_traces=[] 
        for eval in experiment_results:
            if eval.passed:
                continue
            x={
                "INPUT:": eval.dataset_example.input_message,
                "EXPECTED OUTPUT:": eval.dataset_example.expected_output.expected_action,
                "ACTUAL OUTPUT:": eval.actual_output,
                "SCORE:": eval.score,
                "JUDGE FEEDBACK:": eval.judge_reasoning
            }
            thread_trace = await fetch_thread_timeline_as_string(eval.thread_id, config.target_agent_langsmith_project)
            evals_and_thread_traces.append(
                EVALS_AND_THREAD_TRACE_TEMPLATE.format(
                    eval=x,
                    thread_trace=thread_trace
                )
            )
        
        task_message = HumanMessage(content=USER_PROMPT.format(user_raw_request=user_raw_request, evals_and_thread_traces=evals_and_thread_traces))
        input_messages.append(task_message)

    # Pass context along with state
    result = await agent.ainvoke(
        input={"messages": input_messages},
        config=RunnableConfig(recursion_limit=200),
        context=SandboxToolContext(sandbox_context=sandbox_context)  # Pass sandbox context
    )
    logger.info(f"developer completed successfully")


    return {
        "developer_thread": result.get("messages"),
    }