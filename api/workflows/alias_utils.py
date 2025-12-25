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


def extract_block_alias_info(node: Dict[str, Any], existing_aliases: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Extract alias information for a single node.
    
    Args:
        node: Node dictionary with id, type, data (label, config), etc.
        existing_aliases: Optional set of already-used aliases to avoid collisions
        
    Returns:
        Dictionary with:
        - alias: Primary sanitized alias (first available)
        - aliases: List of all sanitized aliases for this block
        - references: List of template reference examples like ["{{alias.output}}"]
    """
    if existing_aliases is None:
        existing_aliases = set()
    
    block_id = node.get("id")
    if not block_id:
        return {"alias": None, "aliases": [], "references": []}
    
    data = node.get("data") or {}
    config = data.get("config") or {}
    
    # Same preference order as derive_block_aliases
    candidates: Tuple[Optional[str], ...] = (
        data.get("label"),
        config.get("tool_name") or config.get("toolName"),
        config.get("variable_name"),
        block_id,
    )
    
    alias_list: List[str] = []
    for candidate in candidates:
        alias = sanitize_block_alias(candidate)
        if not alias or alias in existing_aliases:
            continue
        alias_list.append(alias)
        existing_aliases.add(alias)
    
    if not alias_list:
        return {"alias": None, "aliases": [], "references": []}
    
    primary_alias = alias_list[0]
    references = [
        f"{{{{{primary_alias}.output}}}}",
        f"{{{{{primary_alias}.structured_output}}}}",
    ]
    
    return {
        "alias": primary_alias,
        "aliases": alias_list,
        "references": references,
    }


def refresh_workflow_state_aliases(workflow_state: Dict[str, Any]) -> None:
    """
    Rebuild alias maps in workflow_state after nodes/edges change.
    
    Mutates workflow_state to update:
    - block_aliases: Dict[block_id, List[aliases]]
    - template_reference_examples: Dict[block_id, List[reference_strings]]
    - input_variables: Sorted list of input variable names
    """
    graph_snapshot = {
        "nodes": workflow_state.get("nodes", []),
        "edges": workflow_state.get("edges", []),
    }
    
    block_aliases = derive_block_aliases(graph_snapshot)
    workflow_state["block_aliases"] = block_aliases
    workflow_state["template_reference_examples"] = build_template_reference_examples(block_aliases)
    workflow_state["input_variables"] = sorted(collect_input_variables(graph_snapshot))


__all__ = [
    "build_template_reference_examples",
    "collect_input_variables",
    "derive_block_aliases",
    "extract_block_alias_info",
    "refresh_workflow_state_aliases",
    "sanitize_block_alias",
]

