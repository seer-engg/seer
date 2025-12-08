"""
FastAPI application that exposes the Eval Agent LangGraph.

The eval agent graph is stateful and already configured with a Sqlite
checkpointer (`agents/eval_agent/graph.py`). This service wraps that graph
behind a simple HTTP interface so clients only need to provide a natural
language message. Thread management is handled via the LangGraph
checkpointer—clients can optionally provide a `thread_id` to continue a
previous run, otherwise a new thread is created automatically.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional

import aiosqlite
from fastapi import FastAPI, HTTPException
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from agents.eval_agent.graph import build_graph
from agents.eval_agent.models import EvalAgentState
from shared.config import config
from shared.logger import get_logger
from shared.schema import AgentContext, GithubContext, SandboxContext, UserContext
from shared.tools.loader import resolve_mcp_services


logger = get_logger("api.eval_agent")


app = FastAPI(
    title="Seer Eval Agent API",
    version="0.1.0",
    description="Serve the evaluation agent LangGraph over HTTP.",
)


_graph_lock = asyncio.Lock()
_async_graph = None
_async_checkpointer = None


async def _get_or_create_graph():
    """Lazily build the eval graph with an AsyncSqliteSaver-backed checkpointer."""
    global _async_graph, _async_checkpointer
    if _async_graph is not None:
        return _async_graph

    async with _graph_lock:
        if _async_graph is None:
            conn = await aiosqlite.connect("memory.sqlite")
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            _async_checkpointer = AsyncSqliteSaver(conn)
            _async_graph = build_graph(checkpointer=_async_checkpointer)
    return _async_graph


@app.on_event("shutdown")
async def _shutdown_graph():
    """Ensure the underlying SQLite connection is closed on application shutdown."""
    global _async_graph, _async_checkpointer
    if _async_checkpointer is not None:
        await _async_checkpointer.conn.close()
        _async_checkpointer = None
        _async_graph = None


class EvalAgentRequest(BaseModel):
    """Incoming request schema for running the eval agent."""

    message: str = Field(..., description="User request for the eval agent.")
    thread_id: Optional[str] = Field(
        None,
        description="Existing thread to resume. Leave empty to create a new run.",
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier. Defaults to config.user_id.",
    )
    agent_name: Optional[str] = Field(
        None,
        description="Name of the target agent being evaluated.",
    )
    repo_url: Optional[str] = Field(
        None,
        description="Optional GitHub repository URL for context.",
    )
    branch_name: Optional[str] = Field(
        default="main",
        description="Optional GitHub branch name. Defaults to 'main'.",
    )
    sandbox_id: Optional[str] = Field(
        None,
        description="Optional sandbox id if an environment is already provisioned.",
    )
    sandbox_working_directory: Optional[str] = Field(
        None,
        description="Working directory inside the sandbox.",
    )
    sandbox_working_branch: Optional[str] = Field(
        None,
        description="Working branch within the sandbox. Defaults to repo branch.",
    )
    target_agent_version: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional target agent version counter to seed into context.",
    )
    mcp_services: Optional[List[str]] = Field(
        default=None,
        description="Explicit MCP services to load. Defaults to config defaults.",
    )


class EvalAgentResponse(BaseModel):
    """Response schema for eval agent runs."""

    thread_id: str = Field(description="Thread identifier that stores state in SQLite.")
    state: Dict[str, Any] = Field(
        description="Serialized eval agent state returned from LangGraph."
    )


def _serialize_value(value: Any) -> Any:
    """Convert LangChain/LangGraph objects into JSON-serializable forms."""
    if isinstance(value, BaseModel):
        return _serialize_value(value.model_dump())
    if isinstance(value, BaseMessage):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    return value


def _serialize_state(state: Any) -> Dict[str, Any]:
    """Normalize LangGraph state output into a JSON-friendly dict."""
    if isinstance(state, EvalAgentState):
        raw_state: Dict[str, Any] = state.model_dump()
    elif isinstance(state, BaseModel):
        raw_state = state.model_dump()
    elif isinstance(state, dict):
        raw_state = state
    else:
        raw_state = {"data": state}
    return {key: _serialize_value(val) for key, val in raw_state.items()}


def _build_agent_context(payload: EvalAgentRequest) -> AgentContext:
    """Construct AgentContext defaults using only the incoming message."""
    user_identifier = payload.user_id or config.user_id or "seer-user"
    user_context = UserContext(user_id=user_identifier, raw_request=payload.message)

    github_context = (
        GithubContext(repo_url=payload.repo_url, branch_name=payload.branch_name)
        if payload.repo_url
        else None
    )

    sandbox_context = (
        SandboxContext(
            sandbox_id=payload.sandbox_id,
            working_directory=payload.sandbox_working_directory or "",
            working_branch=payload.sandbox_working_branch
            or payload.branch_name
            or "main",
        )
        if payload.sandbox_id
        else None
    )

    services = resolve_mcp_services(payload.mcp_services or [])

    return AgentContext(
        user_context=user_context,
        github_context=github_context,
        sandbox_context=sandbox_context,
        agent_name=payload.agent_name or "target_agent",
        target_agent_version=payload.target_agent_version or 0,
        mcp_services=services,
    )


def _build_initial_state(payload: EvalAgentRequest) -> EvalAgentState:
    """Create the initial EvalAgentState when a thread starts."""
    context = _build_agent_context(payload)
    return EvalAgentState(
        context=context,
        messages=[HumanMessage(content=payload.message)],
    )


def _thread_config(thread_id: str) -> Dict[str, Dict[str, str]]:
    """Helper for constructing LangGraph configurable settings."""
    return {"configurable": {"thread_id": thread_id}}


async def _has_existing_state(graph_instance, thread_cfg: Dict[str, Dict[str, str]]) -> bool:
    """Check whether the Sqlite checkpointer already stores state for this thread."""
    get_state = getattr(graph_instance, "aget_state", None)
    if not callable(get_state):
        return False

    try:
        snapshot = await get_state(thread_cfg)
    except Exception:
        logger.debug(
            "Unable to fetch existing state for config=%s", thread_cfg, exc_info=True
        )
        return False

    if snapshot is None:
        return False

    values = getattr(snapshot, "values", None)
    if isinstance(values, dict):
        return bool(values)

    state = getattr(snapshot, "state", None)
    if state is None:
        return False

    state_values = getattr(state, "values", None)
    if isinstance(state_values, dict):
        return bool(state_values)

    return False


@app.get("/healthz", tags=["health"])
async def healthcheck() -> Dict[str, str]:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/eval-agent/run", response_model=EvalAgentResponse, tags=["eval-agent"])
async def run_eval_agent(payload: EvalAgentRequest) -> EvalAgentResponse:
    """
    Run the eval agent with a natural language message.

    If `thread_id` is omitted a new conversation thread is created
    and returned in the response. Reusing the same `thread_id`
    lets clients resume multi-step eval workflows without having to resend
    the entire state themselves.
    """
    eval_graph = await _get_or_create_graph()
    thread_id = payload.thread_id or str(uuid.uuid4())
    thread_cfg = _thread_config(thread_id)
    logger.info("Starting eval agent run (thread_id=%s)", thread_id)

    existing_state = await _has_existing_state(eval_graph, thread_cfg)

    if payload.thread_id and not existing_state:
        logger.warning(
            "Thread %s was requested but no saved state was found; creating a new run.",
            thread_id,
        )

    # Use a full EvalAgentState for brand-new threads; otherwise only append a message
    if not existing_state:
        state_input: Any = _build_initial_state(payload)
    else:
        state_input = {"messages": [HumanMessage(content=payload.message)]}

    try:
        result_state = await eval_graph.ainvoke(state_input, config=thread_cfg)
    except Exception as exc:
        logger.exception("Eval agent invocation failed (thread_id=%s)", thread_id)
        raise HTTPException(
            status_code=500,
            detail=f"Eval agent failed to run: {exc}",
        ) from exc

    serialized = _serialize_state(result_state)
    return EvalAgentResponse(thread_id=thread_id, state=serialized)



