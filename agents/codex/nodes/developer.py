"""Context and plan step"""
from __future__ import annotations
from langchain_core.messages.base import BaseMessage


from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

from shared.logger import get_logger
from shared.tools import web_search
from shared.llm import get_llm
from agents.codex.state import CodexState, TaskPlan
from agents.codex.format_thread import fetch_thread_timeline_as_string
from shared.config import TARGET_AGENT_LANGSMITH_PROJECT
from deepagents import create_deep_agent, CompiledSubAgent
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

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
    run_command,
    read_file,
    grep,
    inspect_directory,
    _inspect_directory_impl,
    create_file,
    create_directory,
    write_file,
    patch_file,
    apply_patch,    
)
from agents.codex.common_instructions import TARGET_AGENT_GUARDRAILS , COMPOSIO_LANGCHAIN_INTEGRATION
from shared.tools.docs_tools import docs_tools
logger = get_logger("codex.nodes.developer")


SYSTEM_PROMPT = """
    You are an Expert Software Engineer specializing in LLM based Agent development.
    Your task is to understand current state of the agent , based on failed eval threads develop the agent to pass all the eval cases.
    
# IMPORTANT:
    - for searching of packages, use the web_search tool, do not use pip search.
    - after adding any new package to pyproject.toml, always run command `pip install -e .` to install the new package.
    - relative imports often results in errors, use absolute imports whenever possible.
    - For complex tasks, delegate to your subagents using the task() tool.
    This keeps your context clean and improves results.
""" + TARGET_AGENT_GUARDRAILS + COMPOSIO_LANGCHAIN_INTEGRATION

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


codebase_explorer_subgraph = create_agent(
    model=ChatOpenAI(model="gpt-5-codex", reasoning={"effort": "high"}),
    tools=[
        inspect_directory,
        read_file,
        grep,
        search_code,
        search_symbols,
        semantic_search,
        get_symbol_definition
    ],
    system_prompt=""" You are a great codebase explorer. based on the user request, you need to explore the codebase and provide the answer to the question, you are provided with the tools to explore the codebase""",
    context_schema=SandboxToolContext,

)
codebase_explorer_subagent = CompiledSubAgent(
    name="codebase-explorer-agent",
    description="Used to explore the codebase and query the codebase",
    runnable=codebase_explorer_subgraph,
)

documentation_explorer_subgraph = create_agent(
    model=ChatOpenAI(model="gpt-5-codex", reasoning={"effort": "high"}),
    tools=[
        web_search,
        *docs_tools,
    ],
    system_prompt="""Used to explore the documentation of langchain/langgraph , composio and any other relevant documentation, You are a great documentation explorer, you are given a question and you need to explore the documentation and provide the answer to the question""",
    context_schema=SandboxToolContext,
)
documentation_explorer_subagent = CompiledSubAgent(
    name="documentation-explorer-agent",
    description="Used to explore the documentation of langchain/langgraph , composio and any other relevant documentation",
    runnable=documentation_explorer_subgraph,
)

subagents = [codebase_explorer_subagent, documentation_explorer_subagent]

async def developer(state: CodexState) -> CodexState:
    """Single ReAct agent that gathers repo context and returns a concrete plan."""

    # Extract sandbox context for tools
    sandbox_context = state.context.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")
    user_raw_request = state.context.user_context.raw_request
    
    experiment_results = state.experiment_context.results

    llm = ChatOpenAI(model="gpt-5-codex", reasoning={"effort": "high"})

    agent = create_deep_agent(
        model=llm,
        tools=[
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
            web_search,
            apply_patch,
            patch_file,
            create_file,
            create_directory,
            write_file, 
            *docs_tools,
        ],
        system_prompt=SYSTEM_PROMPT,
        context_schema=SandboxToolContext,  # Add context schema for sandbox tools
        subagents=subagents,
    )
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
            thread_trace = await fetch_thread_timeline_as_string(eval.thread_id, TARGET_AGENT_LANGSMITH_PROJECT)
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