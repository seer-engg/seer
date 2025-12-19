"""Supervisor agent node with database tools and subagents."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.tools import tool
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field

from shared.logger import get_logger
from shared.config import config
from shared.llm import get_agent_final_respone
from shared.tools import think
from shared.tools.postgres import PostgresClient, get_postgres_tools

from agents.supervisor.state import SupervisorState, FileAttachment

logger = get_logger("supervisor.nodes.supervisor")

# Load system prompt
SYSTEM_PROMPT_PATH = Path(__file__).parent / "supervisor_prompt.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

# LLM configuration
# llm = ChatOpenAI(
#     model="gpt-5-mini",
#     reasoning={"effort": "medium"}
# )
# low_reasoning_llm = ChatOpenAI(
#     model="gpt-5-mini",
#     reasoning={"effort": "low"}
# )
import os
llm = ChatOpenAI(
    model="gpt-oss-120b",
    api_key=os.environ["CEREBRAS_API_KEY"],
    base_url="https://api.cerebras.ai/v1",
    temperature=0.2,
)
low_reasoning_llm = llm

# ============================================================================
# Database Explorer Subagent
# ============================================================================

DATABASE_EXPLORER_SYSTEM_PROMPT = """
You are a specialized Database Explorer agent. Your job is to thoroughly explore and understand PostgreSQL database structures and data.

## Capabilities
- Explore database schemas, tables, columns, and relationships
- Run exploratory queries to understand data patterns
- Analyze foreign key relationships and data dependencies
- Provide insights about table structures and data distributions

## Guidelines
1. Start with schema exploration to understand the database structure
2. Use appropriate queries to explore data patterns
3. Pay attention to foreign keys and relationships between tables
4. Provide clear, structured reports of your findings
5. Always respect query limits to avoid overwhelming results

## Available Tools
- `postgres_query`: Execute SELECT queries
- `postgres_get_schema`: Get schema information for tables

When exploring:
1. First get the overall schema view
2. Then dive into specific tables as needed
3. Look for patterns and relationships
4. Summarize findings clearly
"""

# Get postgres tools for the explorer subagent
# We'll create tools bound to a client at runtime
_explorer_postgres_client: Optional[PostgresClient] = None


def _get_explorer_tools(connection_string: Optional[str] = None) -> list:
    """Get postgres tools for the database explorer, using connection string from config if not provided."""
    global _explorer_postgres_client
    
    conn_str = connection_string or config.database_uri
    if not conn_str:
        raise ValueError("No database connection string provided. Set DATABASE_URI or pass connection_string.")
    
    if _explorer_postgres_client is None or _explorer_postgres_client._connection_string != conn_str:
        _explorer_postgres_client = PostgresClient(conn_str)
    
    # Only include read-only tools for the explorer
    tools = _explorer_postgres_client.get_tools()
    # Filter to only read tools (postgres_query, postgres_get_schema)
    read_only_tools = [t for t in tools if t.name in ("postgres_query", "postgres_get_schema")]
    return read_only_tools


def _create_database_explorer_subgraph(connection_string: Optional[str] = None):
    """Create the database explorer subagent graph."""
    explorer_tools = _get_explorer_tools(connection_string)
    
    return create_agent(
        model=low_reasoning_llm,
        tools=explorer_tools,
        system_prompt=DATABASE_EXPLORER_SYSTEM_PROMPT,
    )


class DatabaseExplorerInput(BaseModel):
    """Input schema for database explorer tool."""
    query: str = Field(description="The exploration query or task for the database explorer. Be specific about what you want to explore.")


@tool(args_schema=DatabaseExplorerInput)
async def database_explorer_subagent(query: str) -> str:
    """
    Delegate database exploration tasks to a specialized database explorer agent.
    
    Use this tool when you need to:
    - Explore database schema comprehensively
    - Understand relationships between multiple tables
    - Analyze data patterns across the database
    - Get detailed information about database structure
    
    Args:
        query: The exploration query or task. Be specific about what you want to explore.
    
    Returns:
        Detailed exploration results and insights about the database.
    """
    try:
        explorer = _create_database_explorer_subgraph()
        result = await explorer.ainvoke({"messages": [HumanMessage(content=query)]})
        return await get_agent_final_respone(result)
    except Exception as e:
        logger.error(f"Database explorer error: {e}")
        return f"Database exploration failed: {str(e)}"


# ============================================================================
# File Request Tool (uses interrupt)
# ============================================================================

class FileRequestInput(BaseModel):
    """Input schema for file request tool."""
    description: str = Field(description="Description of the files you need and why they would help")
    file_types: List[str] = Field(
        default=["image", "pdf"],
        description="Types of files you're requesting (e.g., 'image', 'pdf', 'document')"
    )


@tool(args_schema=FileRequestInput)
def request_files(description: str, file_types: List[str] = ["image", "pdf"]) -> str:
    """
    Request additional files (images, PDFs) from the user via interrupt.
    
    Use this tool when you need visual context like:
    - Schema diagrams or ER diagrams
    - Screenshots of data or reports
    - PDF documentation
    - Any other visual/document context
    
    Args:
        description: Description of what files you need and why
        file_types: Types of files you're requesting
    
    Returns:
        The user's response with file information or rejection
    """
    file_types_str = ", ".join(file_types)
    
    request_payload = {
        "type": "file_request",
        "description": description,
        "accepted_types": file_types,
        "message": (
            f"📎 **File Request**\n\n"
            f"The supervisor agent is requesting additional files:\n\n"
            f"**Reason**: {description}\n\n"
            f"**Accepted types**: {file_types_str}\n\n"
            f"Please attach the requested files or reply 'skip' to continue without them."
        )
    }
    
    response = interrupt(request_payload)
    return str(response)


# ============================================================================
# Main Supervisor Agent
# ============================================================================

_supervisor_postgres_client: Optional[PostgresClient] = None


def _get_supervisor_tools(connection_string: Optional[str] = None) -> list:
    """Get all tools for the supervisor agent."""
    global _supervisor_postgres_client
    
    conn_str = connection_string or config.database_uri
    
    tools = [
        database_explorer_subagent,
        request_files,
        think,
    ]
    
    # Add postgres tools if connection string is available
    if conn_str:
        if _supervisor_postgres_client is None or _supervisor_postgres_client._connection_string != conn_str:
            _supervisor_postgres_client = PostgresClient(conn_str)
        
        postgres_tools = _supervisor_postgres_client.get_tools()
        tools.extend(postgres_tools)
    else:
        logger.warning("No database connection string provided. Postgres tools will not be available.")
    
    return tools


def _create_supervisor_agent(connection_string: Optional[str] = None):
    """Create the main supervisor agent."""
    tools = _get_supervisor_tools(connection_string)
    
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
        ],
    )


def _format_messages_with_files(
    messages: List[BaseMessage],
    files: List[FileAttachment]
) -> List[BaseMessage]:
    """
    Format messages to include file attachments as multimodal content.
    
    For images, converts to the format expected by vision-capable models.
    For PDFs, includes text description with base64 reference.
    """
    if not files:
        return messages
    
    # Get the last human message to attach files to
    formatted_messages = list(messages)
    
    # Build multimodal content blocks for files
    file_content_blocks = []
    
    for file in files:
        if file.is_image:
            # Image content block for vision models
            file_content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{file.content_type};base64,{file.data}",
                    "detail": "auto"
                }
            })
        elif file.is_pdf:
            # For PDFs, add as text with base64 reference
            # Note: Most models can't directly process PDFs, so we note it's attached
            file_content_blocks.append({
                "type": "text",
                "text": f"[PDF Attached: {file.filename}]\n(PDF content is base64 encoded and attached for reference)"
            })
        else:
            # Other file types
            file_content_blocks.append({
                "type": "text",
                "text": f"[File Attached: {file.filename} ({file.content_type})]"
            })
    
    if file_content_blocks:
        # Find the last HumanMessage and enhance it with file content
        for i in range(len(formatted_messages) - 1, -1, -1):
            if isinstance(formatted_messages[i], HumanMessage):
                original_content = formatted_messages[i].content
                
                # Convert to multimodal format
                if isinstance(original_content, str):
                    new_content = [{"type": "text", "text": original_content}]
                else:
                    new_content = list(original_content)
                
                new_content.extend(file_content_blocks)
                formatted_messages[i] = HumanMessage(content=new_content)
                break
    
    return formatted_messages


async def supervisor(state: SupervisorState) -> dict:
    """
    Main supervisor node that orchestrates database operations.
    
    Handles:
    - Multimodal input (text, images, PDFs)
    - PostgreSQL operations via tools
    - Delegation to database explorer subagent
    - Human-in-the-loop file requests via interrupt
    """
    logger.info("Supervisor agent processing request...")
    
    # Get connection string from state or config
    connection_string = state.database_connection_string or config.database_uri
    
    # Create agent with appropriate tools
    agent = _create_supervisor_agent(connection_string)
    
    # Format messages with any attached files
    input_messages = _format_messages_with_files(
        list(state.messages or []),
        state.files or []
    )
    
    if not input_messages:
        logger.warning("No input messages provided to supervisor")
        return {
            "response": "No input provided. Please send a message to get started.",
        }
    
    # Invoke the agent
    try:
        result = await agent.ainvoke(
            input={"messages": input_messages},
            config=RunnableConfig(recursion_limit=50),
        )
        
        # Extract final response
        final_response = await get_agent_final_respone(result)
        logger.info("Supervisor completed successfully")
        
        return {
            "messages": result.get("messages", []),
            "response": final_response,
        }
        
    except GraphInterrupt:
        # Re-raise GraphInterrupt - LangGraph needs this for human-in-the-loop
        # The interrupt() function raises this exception which LangGraph catches
        # to pause execution and wait for user input
        raise
    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        return {
            "response": f"An error occurred: {str(e)}",
        }

