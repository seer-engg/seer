"""Utilities for loading and prioritizing MCP tools."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from langchain_core.tools import BaseTool

from shared.logger import get_logger
from shared.mcp_client import get_mcp_client_and_configs


logger = get_logger("shared.tool_catalog")


DEFAULT_MCP_SERVICES: Sequence[str] = ("asana", "github", "langchain_docs")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_mcp_services(requested_services: Iterable[str]) -> List[str]:
    """Normalize and optionally augment requested services."""

    normalized: List[str] = []
    for service in requested_services or []:
        if not service:
            continue
        normalized_name = service.strip().lower()
        if normalized_name and normalized_name not in normalized:
            normalized.append(normalized_name)

    if not _env_flag("EVAL_AGENT_LOAD_DEFAULT_MCPS", default=True):
        return normalized

    combined: List[str] = list(DEFAULT_MCP_SERVICES)
    for service in normalized:
        if service not in combined:
            combined.append(service)
    return combined


@dataclass
class ToolEntry:
    name: str
    description: str
    service: str


async def load_tool_entries(service_names: Sequence[str]) -> Dict[str, ToolEntry]:
    """Return lightweight metadata for the requested MCP tools keyed by name."""

    if not service_names:
        return {}

    mcp_client, _ = await get_mcp_client_and_configs(list(service_names))
    tools: List[BaseTool] = await mcp_client.get_tools()
    entries: Dict[str, ToolEntry] = {}
    for tool in tools:
        service = tool.name.split(".", 1)[0] if "." in tool.name else "misc"
        entry = ToolEntry(
            name=tool.name,
            description=getattr(tool, "description", "") or "",
            service=service,
        )
        entries[tool.name.lower()] = entry
    logger.info(
        "Loaded %d MCP tool entries for services: %s",
        len(entries),
        ", ".join(service_names),
    )
    return entries


def select_relevant_tools(
    entries: Dict[str, ToolEntry],
    context: str,
    *,
    max_total: int = 20,
    max_per_service: int = 5,
) -> List[str]:
    """Score tools against the provided context string and return a limited list."""

    if not entries:
        return []

    keywords = set(re.findall(r"[a-z0-9_]+", context.lower()))
    service_buckets: Dict[str, List[tuple[int, str]]] = defaultdict(list)

    for entry in entries.values():
        haystack = f"{entry.name} {entry.description}".lower()
        score = sum(1 for kw in keywords if kw and kw in haystack)
        service_buckets[entry.service].append((score, entry.name))

    prioritized: List[str] = []
    for items in service_buckets.values():
        items.sort(key=lambda pair: (-pair[0], pair[1]))
        limited = [name for score, name in items[:max_per_service]]
        prioritized.extend(limited)
        if len(prioritized) >= max_total:
            break
    return prioritized[:max_total]


def canonicalize_tool_name(raw_tool: str, service_hint: str | None = None) -> str:
    """Normalize various tool name spellings into lookup-friendly keys."""

    if not raw_tool:
        return raw_tool

    normalized = raw_tool.strip()
    if not normalized:
        return normalized

    lowered = normalized.lower()
    if lowered.startswith("system."):
        return lowered

    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".", "_")
    normalized = normalized.strip("_")
    normalized = normalized.lower()

    if "_" in normalized:
        return normalized

    if service_hint:
        prefix = service_hint.strip().lower()
        if prefix:
            return f"{prefix}_{normalized}"

    return normalized


def build_tool_name_set(entries: Dict[str, ToolEntry]) -> Dict[str, str]:
    """Return canonical tool keys mapped to their original names."""

    return {
        canonicalize_tool_name(entry.name): entry.name
        for entry in entries.values()
    }
