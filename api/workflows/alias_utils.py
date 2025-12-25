"""
Helpers for deriving workflow alias metadata and template hints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import re

_ALIAS_SANITIZE_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def sanitize_block_alias(value: Optional[str]) -> Optional[str]:
    """Convert arbitrary labels/tool names into safe template identifiers."""
    if not value:
        return None
    alias = _ALIAS_SANITIZE_PATTERN.sub("_", str(value).strip()).strip("_").lower()
    if not alias:
        return None
    if alias[0].isdigit():
        alias = f"_{alias}"
    return alias


def derive_block_aliases(graph_data: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a mapping of block_id -> list of sanitized aliases usable in templates.
    
    Preference order:
      1. Explicit block label provided by the user
      2. tool_name/toolName (for tool blocks)
      3. variable_name (for input blocks)
      4. Sanitized ReactFlow node id (guaranteed fallback)
    """
    nodes = (graph_data or {}).get("nodes") or []
    alias_map: Dict[str, List[str]] = {}
    used_aliases: Set[str] = set()
    
    for node in nodes:
        block_id = node.get("id")
        if not block_id:
            continue
        
        data = node.get("data") or {}
        config = data.get("config") or {}
        candidates: Tuple[Optional[str], ...] = (
            data.get("label"),
            config.get("tool_name") or config.get("toolName"),
            config.get("variable_name"),
            block_id,
        )
        
        alias_list: List[str] = []
        for candidate in candidates:
            alias = sanitize_block_alias(candidate)
            if not alias or alias in used_aliases:
                continue
            alias_list.append(alias)
            used_aliases.add(alias)
        
        if alias_list:
            alias_map[block_id] = alias_list
    
    return alias_map


def collect_input_variables(graph_data: Optional[Dict[str, Any]]) -> Set[str]:
    """Collect simple variable names defined by input blocks (fields, variable_name)."""
    nodes = (graph_data or {}).get("nodes") or []
    variables: Set[str] = set()
    
    for node in nodes:
        data = node.get("data") or {}
        if data.get("type") != "input":
            continue
        config = data.get("config") or {}
        
        variable_name = config.get("variable_name")
        if variable_name:
            variables.add(str(variable_name))
        
        fields = config.get("fields")
        if isinstance(fields, list):
            for field in fields:
                if isinstance(field, dict):
                    field_name = field.get("id") or field.get("name")
                    if field_name:
                        variables.add(str(field_name))
    
    return variables


def build_template_reference_examples(alias_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Provide simple {{alias.*}} examples per block for UI/agent hints."""
    examples: Dict[str, List[str]] = {}
    
    for block_id, aliases in alias_map.items():
        if not aliases:
            continue
        alias = aliases[0]
        examples[block_id] = [
            f"{{{{{alias}.output}}}}",
            f"{{{{{alias}.structured_output}}}}",
        ]
    
    return examples


__all__ = [
    "build_template_reference_examples",
    "collect_input_variables",
    "derive_block_aliases",
    "sanitize_block_alias",
]

