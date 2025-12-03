"""
Helper functions for formatting messages sent to target agents with different context levels.

This module provides utilities to enrich target agent messages with varying levels of context,
enabling experiments on how context comprehensiveness impacts evaluation quality.
"""
from typing import Optional
from shared.schema import DatasetExample, AgentContext
from shared.logger import get_logger

logger = get_logger("shared.message_formatter")


def format_target_agent_message(
    example: DatasetExample,
    context: AgentContext,
    context_level: int = 0
) -> str:
    """
    Format a message for the target agent with specified context level.
    
    Context Levels:
    - 0 (Minimal): Only input_message from dataset example
    - 1 (System Goal): input_message + system goal description
    - 2 (System Goal + Action): Level 1 + expected action
    - 3 (Full Context): Level 2 + MCP services + resource hints
    
    Args:
        example: The dataset example containing input_message and expected_output
        context: The agent context containing system goal, MCP services, etc.
        context_level: The level of context to include (0-3)
    
    Returns:
        Formatted message string to send to target agent
    """
    base_message = example.input_message
    
    if context_level == 0:
        # Minimal: just the input message
        return base_message
    
    # Build enriched message
    parts = [base_message]
    
    if context_level >= 1:
        # Add MCP services and resource hints
        if context.mcp_services:
            parts.append("\n\n--- Available Services ---")
            parts.append(", ".join(context.mcp_services))
        
        if context.mcp_resources:
            resource_hints = str(context.mcp_resources)
            if resource_hints and resource_hints != "None provided. Prefer using [var:...] or [resource:...] tokens for runtime values.":
                parts.append("\n\n--- Available Resources ---")
                parts.append(resource_hints)
        
    
    if context_level >= 2:
        # Add system goal
        if context.user_context and context.user_context.raw_request:
            parts.append("\n\n--- System Goal ---")
            parts.append(context.user_context.raw_request)
        
    
    if context_level >= 3:
        # Add expected action
        if example.expected_output and example.expected_output.expected_action:
            parts.append("\n\n--- Expected Action ---")
            parts.append(example.expected_output.expected_action)
        
    
    enriched_message = "\n".join(parts)
    
    logger.info(
        f"Formatted target agent message with context_level={context_level} "
        f"(base_length={len(base_message)}, enriched_length={len(enriched_message)})"
    )
    
    return enriched_message

