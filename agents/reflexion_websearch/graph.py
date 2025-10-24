import os
import traceback
from typing import List, Literal, Dict, Any

from langchain.messages import AnyMessage
from langchain.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from tavily import TavilyClient
from pydantic import BaseModel
import requests

from shared.llm import get_llm
from shared.logger import get_logger

llm = get_llm(model="gpt-4.1", temperature=0.0)
logger = get_logger('reflexion_websearch')
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Mem0 configuration
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
MEM0_BASE_URL = "https://api.mem0.ai"
MEM0_DEFAULT_USER_ID = os.getenv("MEM0_USER_ID", "user_akshay")


def _mem0_headers() -> Dict[str, str]:
    if not MEM0_API_KEY:
        raise RuntimeError("MEM0_API_KEY not set in environment")
    return {
        "Authorization": f"Token {MEM0_API_KEY}",
        "Content-Type": "application/json",
    }


def mem0_add_memory(messages: List[Dict[str, str]], user_id: str | None = None) -> Dict[str, Any]:
    payload = {
        "messages": messages,
        "user_id": user_id or MEM0_DEFAULT_USER_ID,
    }
    print('XXX', payload, flush=True)
    resp = requests.post(f"{MEM0_BASE_URL}/v1/memories/", headers=_mem0_headers(), json=payload, timeout=30)
    print('XXX', resp.json(), flush=True)
    resp.raise_for_status()
    return resp.json()


def mem0_search_memories(query: str, user_id: str | None = None) -> List[Dict[str, Any]]:
    payload = {
        "query": query,
        "filters": {
            "OR": [
                {"user_id": (user_id or MEM0_DEFAULT_USER_ID)}
            ]
        }
    }
    resp = requests.post(f"{MEM0_BASE_URL}/v2/memories/search/", headers=_mem0_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Normalize results into a list of dicts
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        if "memories" in data and isinstance(data["memories"], list):
            return data["memories"]
    return [data]


class InputState(BaseModel):
    messages: list[AnyMessage]

class OutputState(BaseModel):
    trajectory: list[AnyMessage] = []
    latest_answer: str = None
    tries_remaining: int = 3

class OverallState(InputState, OutputState):
    pass


@tool
def think(thought: str) -> str:
    """
    Think tool for reflexion_websearch: logs internal thoughts; not logged in memory; no side effects.
    """
    logger.info(f"THINK: {thought}")
    return f"Thought: {thought}"


@tool
def reflect(reflection: str) -> str:
    """Use `reflect` tool to look back on past experience (actions and results) to analyze and understand them.
    Use this tool more intentionally and deliberately, focused on meaning making and learning from experiences.
    Args:
        reflection: The reflection to produce.
    Returns:
        A string containing the reflection.
    """
    logger.info(f"Reflection: {reflection}")
    return f"Reflection: {reflection}"

@tool
def web_search_qna(query: str) -> str:
    """Web search QnA tool. Use this tool to search the web and get accurate and concise answers to a question.
    Args:
        query: The query to search the web for.
    Returns:
        A string containing the answer to the question.
    """
    print('XXX', query, flush=True)
    response = tavily_client.qna_search(query, search_depth='basic', max_results=3)
    print('XXX', response, flush=True)
    return response

@tool
def search_in_memory(query: str) -> List[str]:
    """Search reflections using Mem0 semantic search for the current user.
    Args:
        query: The query to search for in memory.
    Returns:
        A list of textual memory snippets.
    """
    try:
        results = mem0_search_memories(query, user_id=MEM0_DEFAULT_USER_ID)
        texts: List[str] = []
        for item in results:
            # Attempt common fields
            if isinstance(item, dict):
                if "content" in item and isinstance(item["content"], str):
                    texts.append(item["content"])
                elif "memory" in item and isinstance(item["memory"], str):
                    texts.append(item["memory"])
                elif "text" in item and isinstance(item["text"], str):
                    texts.append(item["text"])
                else:
                    texts.append(str(item))
            else:
                texts.append(str(item))
        return texts
    except Exception as e:
        logger.error(traceback.format_exc())
        return []


# nodes
def reflect_node(state: OverallState):
    _REFLECTOR_SYSTEM_PROMPT = f"""
    # who are you?
    You are reflector node in reflexion.
    
    # whats your goal?
    Your goal is to determine what you have been doing so far, what you have been learning so far, and decide if it makes sense to continue or if we should stop and call for help.

    # whats not your job?
    - actually finding the information, that's the job of the actor node.
    - actually acting with the environment, that's the job of the actor node.
    - actually evaluating the actions, that's the job of the evaluator node.

    # what you can do?
    - you can collect the user conversation, your past reflections so far.
    - you can reflect on your past reflections so far.

    # when to stop?
    - if you've been running in circles for last 3 searches, it's a better idea to stop and call for help. 

    # when to continue?
    - if you've been learning something new, it's a better idea to continue.

    # what happens to your reflections?
    - They are persistent and stored in memory. 
    - They are used by an actor agent to act with the environment. 
    - They are evaluated by an evaluator agent. 

    # internal trajectory so far
    {state.trajectory}

    # user conversation so far
    {state.messages}
    """
    # run LLM with tools
    llm_with_tools = llm.bind_tools([think, reflect, search_in_memory], strict=True)
    response = llm_with_tools.invoke(_REFLECTOR_SYSTEM_PROMPT)

    tool_message = ToolMessage(
        content=response.text, 
        tool_call_id = response.tool_calls[0]['id'],
    )

    # mandatory: store the reflection in Mem0
    try:
        mem0_add_memory(
            messages=state.trajectory + state.messages + [{"role": "assistant", "content": response.text}],
            user_id=MEM0_DEFAULT_USER_ID,
        )
    except Exception as e:
        logger.error(traceback.format_exc())

    # move to next node
    return Command(update={'trajectory': state.trajectory + [response, tool_message]})


def act_node(state: OverallState):
    _ACTOR_SYSTEM_PROMPT = f"""
    # who are you?
    You are actor node in reflexion.

    # whats your goal?
    Your goal is acting with the environment based on the reflections produced by the reflect node and user conversation so far.

    # what you can do?
    - you can use the web_search_qna tool to search the web and get accurate and concise answers to a question.

    # what happens to your actions?
    - Based on it, decide which tool you should use to act with the environment. 
    - Your actions will be later evaluated by the evaluator node in reflexion.

    # internal trajectory so far
    {state.trajectory}

    # user conversation so far
    {state.messages}

    # latest answer so far
    {state.latest_answer}

    # task
    come up with a query to search the web for the answer to the user's question.
    """
    # run LLM with tools
    llm_with_tools = llm.bind_tools([web_search_qna], strict=True)
    response = llm_with_tools.invoke(_ACTOR_SYSTEM_PROMPT)

    if not response.tool_calls:
        return Command(update={'trajectory': state.trajectory})

    tool_message = ToolMessage(
        content=response.text, 
        tool_call_id = response.tool_calls[0]['id'],
    )

    # move to next node
    return Command(update={
        'trajectory': state.trajectory + [response, tool_message], 
        'tries_remaining': state.tries_remaining - 1,
        'latest_answer': response.text,
    })

@tool
def evaluate(evaluation: str) -> str:
    """Use `evaluate` tool to evaluate the actions produced by the actor node based on the reflections produced by the reflect node and user conversation so far.
    Args:
        evaluation: The evaluation to produce.
    Returns:
        A string containing the evaluation.
    """
    return f"Evaluation: {evaluation}"


def eval_node(state: OverallState):
    _EVALUATOR_SYSTEM_PROMPT = f"""
    # who are you?
    You are evaluator node in reflexion.

    # whats your goal?
    Your goal is evaluating the actions produced by the actor node based on the reflections produced by the reflect node and user conversation so far.

    # how should you evaluate the actions?
    - be intellectually honest
    - be concise and to the point
    - be objective
    - critically evaluate the actions and the reflections

    # internal trajectory so far
    {state.trajectory}

    # user conversation so far
    {state.messages}
    """
    # run LLM with tools
    llm_with_tools = llm.bind_tools([think, evaluate], strict=True)
    response = llm_with_tools.invoke(_EVALUATOR_SYSTEM_PROMPT)

    tool_message = ToolMessage(
        content=response.text, 
        tool_call_id = response.tool_calls[0]['id'],
    )

    # move to next node
    return Command(update={'trajectory': state.trajectory + [response, tool_message]})


def route(state: OverallState) -> Literal["act", END]:
    if state.tries_remaining > 0:
        return "act"
    else:
        return Command(goto=END, update={'messages': state.messages + [state.latest_answer]})

# build graph
builder = StateGraph(OverallState, input=InputState, output=OutputState)
# add nodes
builder.add_node("reflect", reflect_node)
builder.add_node("act", act_node)
builder.add_node("eval", eval_node)
# define edges
builder.add_edge(START, "reflect")
builder.add_conditional_edges("reflect", route)
builder.add_edge("act", "eval")
builder.add_edge("eval", 'reflect')
_agent = builder.compile()