from typing import Optional, Dict, Any
from shared.logger import get_logger
from shared.llm import get_llm_without_responses_api
from agents.workflow_agent.utils import get_workflow_tools
from shared.logger import get_logger
logger = get_logger(__name__)

from langchain.agents import create_agent
from langchain.agents.middleware import (
        SummarizationMiddleware,
    )
import json
# Autolog LangChain
import mlflow
mlflow.langchain.autolog()

def create_workflow_chat_agent(
    model: str = "gpt-4o-mini",
    checkpointer: Optional[Any] = None,
    workflow_state: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a LangGraph agent for workflow chat assistance using create_agent.
    
    Uses LangChain v1.0+ create_agent with middleware for summarization
    and human-in-the-loop capabilities.
    
    Args:
        model: Model name to use (e.g., 'gpt-5.2', 'gpt-5-mini')
        checkpointer: Optional LangGraph checkpointer for persistence
        
    Returns:
        LangGraph agent compiled with tools and middleware
    """
    
    llm = get_llm_without_responses_api(model=model, temperature=0.7, api_key=None)
    
    # System prompt for the workflow assistant
    workflow_context = ""
    template_hint_section = ""
    if workflow_state:
        workflow_context = f"\n\nCurrent workflow state:\n{json.dumps(workflow_state, indent=2)}\n\nUse this information when calling tools. Tools automatically access workflow state from thread context via runtime configuration."
        alias_examples = workflow_state.get("template_reference_examples") or {}
        if alias_examples:
            alias_lines = []
            for block_id, examples in alias_examples.items():
                if not examples:
                    continue
                alias_lines.append(f"- {block_id}: {', '.join(examples)}")
            if alias_lines:
                template_hint_section = "\nTemplate reference hints (use these names when writing {{alias.output}} expressions):\n" + "\n".join(alias_lines)
    
    system_prompt = f"""You are an intelligent workflow assistant that helps users build and edit workflows. Your role is to understand user intent, discover appropriate tools, and create workflows that achieve their goals.
**Core Principles:**
- Focus on what users want to achieve, not technical implementation details
- Ask questions in everyday language - avoid jargon and technical terms
- Always plan changes first, get approval, then apply - never modify workflows directly
- Use search_tools to discover tools dynamically - let the LLM reason about tool selection
- When referencing other blocks, use the alias hints returned by add_workflow_block/modify_workflow_block tools
- Each tool response includes "variable_references" showing the exact {{alias.output}} format to use
- Always use the aliases from tool responses - never invent template variable names
- Remember alias names are always between double curly braces like {{alias.output}} 
**Creating Workflows:**
1. Understand user intent - what outcome do they want?
2. If unclear, ask 1-2 clarifying questions in plain language (e.g., "What should happen if no emails are found?")
3. Search for relevant tools using search_tools(query, reasoning)
4. Build the workflow step-by-step: add blocks, connect them via tool add_workflow_edge, validate intent
5. remember to connect blocks to each other via tool add_workflow_edge after you have added all blocks
**Modifying Workflows:**
- Call planning tools (add_workflow_block, modify_workflow_block, etc.) to build a complete proposal
- Each tool call must describe the change as a patch operation (add_node, update_node, remove_node, add_edge, remove_edge)
- Never apply edits yourself; simply describe the full change-set so the user can accept or reject it
- Include a concise natural-language justification for the proposal
**Asking Questions:**
- Ask only when essential information is missing - make reasonable assumptions otherwise
- Limit to 1-2 questions at a time to avoid overwhelming users
- Adapt language to user's apparent technical level
- Ask about goals and outcomes, not technical details
**Error Handling:**
- If tools fail, explain the error in plain terms and suggest alternatives
- If workflow state is invalid, identify what's missing and ask for clarification
- When multiple tools could work, choose the simplest that meets the requirement
**Tool Discovery:**
- Use search_tools(query, reasoning) for semantic search by capability
- Use list_available_tools(integration_type) to see all tools, optionally filtered
- Explain why selected tools are appropriate for the user's request
Always think through your reasoning and provide clear explanations for suggestions.{template_hint_section}{workflow_context}"""

    # Get workflow tools (with optional workflow_state injection)
    tools = get_workflow_tools(workflow_state=workflow_state)
    
    # Create summarization model (use same model with lower max tokens)
    summarization_model = get_llm_without_responses_api(
        model=model,
        temperature=0.3,
        api_key=None,
    )
    
    # Build middleware list
    middleware = [
        SummarizationMiddleware(
            model=summarization_model,
            max_tokens_before_summary=256000,  # 256k token limit
        ),
    ]
    
    # Verify checkpointer is provided (required for persistence)
    if checkpointer is None:
        logger.warning("No checkpointer provided to create_workflow_chat_agent - traces will not be persisted")
    else:
        logger.debug(f"Creating workflow chat agent with checkpointer: {type(checkpointer).__name__}")
    
    # Create agent with middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
    )
    
    logger.info(f"Created workflow chat agent with model {model}, checkpointer={'enabled' if checkpointer else 'disabled'}")
    return agent

