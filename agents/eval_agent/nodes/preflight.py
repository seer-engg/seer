"""
Config preflight utilities for eval-agent subgraphs.

Goal: fail-fast with a clean early-exit when required config/env is missing,
instead of crashing mid-graph. Optionally uses LangGraph interrupts to ask
the human for missing values.
"""

from __future__ import annotations

import os
from typing import Callable, List, Sequence, TypeVar, Union, Literal, Any

from langchain_core.messages import AIMessage

from shared.config import config
from shared.logger import get_logger

logger = get_logger("eval_agent.preflight")

TState = TypeVar("TState")

try:
    # LangGraph human-in-the-loop API
    from langgraph.types import interrupt  # type: ignore
except Exception:  # pragma: no cover
    interrupt = None  # type: ignore


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


RequiredSpec = Union[Sequence[str], Callable[[TState], Sequence[str]]]

_ENV_BY_CONFIG_FIELD = {
    "openai_api_key": "OPENAI_API_KEY",
    "github_token": "GITHUB_TOKEN",
    "composio_api_key": "COMPOSIO_API_KEY",
    "langfuse_secret_key": "LANGFUSE_SECRET_KEY",
    "langfuse_public_key": "LANGFUSE_PUBLIC_KEY",
    "langfuse_base_url": "LANGFUSE_BASE_URL",
    "tavily_api_key": "TAVILY_API_KEY",
    "database_uri": "DATABASE_URI",
}


def _apply_runtime_config(field_name: str, value: str) -> None:
    """
    Apply a config override at runtime.

    We set BOTH:
    - `shared.config.config.<field_name>` (for codepaths that read config fields)
    - `os.environ[ENV_VAR]` (for libraries that read env vars directly)
    """
    if not hasattr(config, field_name):
        raise AttributeError(f"Unknown config field: {field_name}")
    setattr(config, field_name, value)
    env_var = _ENV_BY_CONFIG_FIELD.get(field_name, field_name.upper())
    os.environ[env_var] = value


def _parse_human_reply(reply: Any) -> tuple[str, str | None]:
    """
    Normalize interrupt resume payload.

    Returns: (action, value)
    - action: "set" | "exit"
    - value: str | None
    """
    if reply is None:
        return ("exit", None)
    if isinstance(reply, str):
        s = reply.strip()
        if s.lower() in {"exit", "quit", "stop", "cancel"}:
            return ("exit", None)
        return ("set", s if s != "" else None)
    if isinstance(reply, dict):
        action = str(reply.get("action", "")).strip().lower()
        if action in {"exit", "quit", "stop", "cancel"}:
            return ("exit", None)
        # default to set
        value = reply.get("value")
        if value is None:
            value = reply.get("input")
        if value is None:
            value = reply.get("text")
        if value is None:
            return ("set", None)
        return ("set", str(value).strip() or None)
    # Fallback: stringify
    s = str(reply).strip()
    if s.lower() in {"exit", "quit", "stop", "cancel"}:
        return ("exit", None)
    return ("set", s if s != "" else None)


def make_config_preflight_node(
    *,
    subgraph_name: str,
    required: RequiredSpec,
    interactive: bool = True,
) -> Callable[[TState], "dict"]:
    """
    Create an async node that checks required `shared.config.config` fields.

    - On failure: sets `should_exit=True`, `missing_config=[...]`, and appends an AIMessage.
    - On success: clears `should_exit` and `missing_config`.
    """

    async def _node(state: TState) -> dict:
        required_fields: Sequence[str] = required(state) if callable(required) else required
        missing: List[str] = []
        logger.info(f"Checking config for {subgraph_name}: {required_fields}")

        for field_name in required_fields or []:
            if not hasattr(config, field_name):
                # Treat unknown fields as missing (helps catch typos early)
                missing.append(field_name)
                continue
            value = getattr(config, field_name)
            if _is_missing(value):
                missing.append(field_name)

        if missing:
            # Interactive mode: ask human for each missing key (requires checkpointer to resume)
            logger.info(f"Interactive mode: asking human for missing config: {missing}")
            can_interrupt = bool(interactive and interrupt is not None)
            if can_interrupt:
                logger.info(f"Interactive mode: can interrupt: {can_interrupt}")
                for field_name in list(missing):
                    # Re-check as earlier prompts may have set it
                    if hasattr(config, field_name) and not _is_missing(getattr(config, field_name)):
                        continue

                    env_var = _ENV_BY_CONFIG_FIELD.get(field_name, field_name.upper())
                    prompt_payload = {
                        "type": "missing_config",
                        "subgraph": subgraph_name,
                        "field": field_name,
                        "env_var": env_var,
                        "instructions": (
                            f"Provide a value for `{env_var}` (config field `{field_name}`), "
                            "or reply `exit` to stop."
                        ),
                    }
                    logger.info(f"Interactive mode: asking human for missing config: {prompt_payload}")
                    reply = interrupt(prompt_payload)  # pauses until human responds
                    action, value = _parse_human_reply(reply)
                    if action == "exit":
                        msg = (
                            f"Exiting `{subgraph_name}` due to missing required config: "
                            f"{', '.join(sorted(set(missing)))}."
                        )
                        logger.warning(msg)
                        return {
                            "should_exit": True,
                            "missing_config": sorted(set(missing)),
                            "messages": [AIMessage(content=msg)],
                        }
                    if not value:
                        # Still missing; keep going but it will exit below.
                        continue
                    try:
                        _apply_runtime_config(field_name, value)
                        logger.info(f"Config provided interactively for {field_name} ({env_var})")
                    except Exception as e:
                        logger.warning(f"Failed to apply interactive config for {field_name}: {e}")

                # Recompute missing after applying overrides
                missing = []
                for field_name in required_fields or []:
                    if not hasattr(config, field_name) or _is_missing(getattr(config, field_name, None)):
                        missing.append(field_name)

            # If still missing after interactive prompts, exit
            if missing:
                msg = (
                    f"Missing required config for `{subgraph_name}`: {', '.join(sorted(set(missing)))}. "
                    f"Set the corresponding environment variables (or .env) and retry."
                )
                logger.warning(msg)
                return {
                    "should_exit": True,
                    "missing_config": sorted(set(missing)),
                    "messages": [AIMessage(content=msg)],
                }
            
            # All config provided interactively - continue!
            logger.info(f"All required config for `{subgraph_name}` provided interactively. Continuing...")

        return {
            "should_exit": False,
            "missing_config": [],
        }

    return _node


def route_after_preflight(state) -> Literal["continue", "exit"]:
    """Route helper for conditional edges after preflight."""
    return "exit" if getattr(state, "should_exit", False) else "continue"


