"""
Agent Traces API - List and detail endpoints for agent conversation traces.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from seer.database import Workflow, parse_workflow_public_id
from seer.logger import get_logger

from .checkpointer import get_checkpointer

logger = get_logger("api.agents.traces")

router = APIRouter(prefix="/agents/traces", tags=["Agent Traces"])


# =============================================================================
# Pydantic Models
# =============================================================================

class AgentMessage(BaseModel):
    """Agent message model matching frontend interface."""
    id: int = Field(..., description="Sequential message ID")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    thinking: Optional[str] = Field(None, description="AI reasoning/thinking")
    suggested_edits: Optional[Dict[str, Any]] = Field(None, description="Suggested edits")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    created_at: str = Field(..., description="ISO timestamp")


class AgentTraceSummary(BaseModel):
    """Agent trace summary for list view."""
    thread_id: str = Field(..., description="Thread ID")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    workflow_name: Optional[str] = Field(None, description="Associated workflow name")
    message_count: int = Field(..., description="Number of messages")
    created_at: str = Field(..., description="ISO timestamp")
    updated_at: str = Field(..., description="ISO timestamp")
    title: Optional[str] = Field(None, description="Trace title")


class AgentTraceListResponse(BaseModel):
    """Response model for trace list endpoint."""
    traces: List[AgentTraceSummary] = Field(..., description="List of traces")
    total: int = Field(..., description="Total number of traces")


class AgentTraceDetail(BaseModel):
    """Agent trace detail with messages."""
    thread_id: str = Field(..., description="Thread ID")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    workflow_name: Optional[str] = Field(None, description="Associated workflow name")
    created_at: str = Field(..., description="ISO timestamp")
    updated_at: str = Field(..., description="ISO timestamp")
    title: Optional[str] = Field(None, description="Trace title")
    messages: List[AgentMessage] = Field(..., description="List of messages")


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_message_type_and_content(msg: Any) -> tuple[str, str, dict, dict]:
    """
    Extract type, content, and kwargs from various message formats.

    Handles:
    - Dict format (serialized checkpoints)
    - LangChain message objects (HumanMessage, AIMessage, etc.)
    - Unknown formats (fallback to string conversion)

    Returns:
        Tuple of (msg_type, content, additional_kwargs, metadata)
    """
    if isinstance(msg, dict):
        msg_type = msg.get("type", "") or msg.get("role", "")
        content = msg.get("content", "")
        additional_kwargs = msg.get("additional_kwargs", {})
        metadata = msg.get("metadata", {})
    elif hasattr(msg, "content") and hasattr(msg, "type"):
        msg_type = msg.type
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        additional_kwargs = getattr(msg, "additional_kwargs", {})
        metadata = getattr(msg, "metadata", {})
    else:
        msg_type = "unknown"
        content = str(msg)
        additional_kwargs = {}
        metadata = {}

    return msg_type, content, additional_kwargs, metadata


def _determine_message_role(msg_type: str) -> str:
    """
    Map LangChain message type to API role.

    Args:
        msg_type: LangChain message type ("human", "ai", "user", "assistant", etc.)

    Returns:
        API role ("user" or "assistant")
    """
    if msg_type in ("human", "user"):
        return "user"
    if msg_type in ("ai", "assistant"):
        return "assistant"
    return "user"  # Default fallback


def _extract_optional_field(
    field_name: str,
    additional_kwargs: dict,
    metadata: dict,
    aliases: Optional[List[str]] = None
) -> Optional[Any]:
    """
    Extract optional field with fallback priority chain.

    Priority: additional_kwargs[field] > additional_kwargs[alias] > metadata[field]

    Args:
        field_name: Primary field name to extract
        additional_kwargs: LangChain additional_kwargs dict
        metadata: LangChain metadata dict
        aliases: Alternative field names to check

    Returns:
        Field value or None
    """
    value = None

    if isinstance(additional_kwargs, dict):
        value = additional_kwargs.get(field_name)
        if not value and aliases:
            for alias in aliases:
                value = additional_kwargs.get(alias)
                if value:
                    break

    if not value and isinstance(metadata, dict):
        value = metadata.get(field_name)

    return value


def _extract_created_at(checkpoint_ts: Optional[str], metadata: dict) -> str:
    """
    Extract created_at timestamp with fallback priority.

    Priority: metadata.created_at > checkpoint_ts > current UTC time

    Returns:
        ISO format timestamp string
    """
    created_at = checkpoint_ts or datetime.utcnow().isoformat()
    if isinstance(metadata, dict) and "created_at" in metadata:
        created_at = metadata["created_at"]
    return created_at


def _convert_message_to_agent_message(
    msg: Any,
    msg_id: int,
    checkpoint_ts: Optional[str] = None
) -> AgentMessage:
    """Convert LangChain message to AgentMessage format."""
    # Extract message type and content
    msg_type, content, additional_kwargs, metadata = _extract_message_type_and_content(msg)

    # Determine role
    role = _determine_message_role(msg_type)

    # Extract optional fields
    thinking = _extract_optional_field(
        "thinking", additional_kwargs, metadata, aliases=["reasoning"]
    )
    suggested_edits = _extract_optional_field(
        "suggested_edits", additional_kwargs, metadata
    )
    created_at = _extract_created_at(checkpoint_ts, metadata)

    return AgentMessage(
        id=msg_id,
        role=role,
        content=content,
        thinking=thinking,
        suggested_edits=suggested_edits,
        metadata=metadata if isinstance(metadata, dict) else None,
        created_at=created_at,
    )


def _extract_workflow_id_from_thread_id(thread_id: str) -> Optional[str]:
    """Extract workflow_id from thread_id pattern: workflow-wf_X-{uuid}"""
    if thread_id.startswith("workflow-"):
        # Pattern: workflow-wf_1-{uuid}
        parts = thread_id.split("-", 2)  # Split into ['workflow', 'wf_1', '{uuid}']
        if len(parts) >= 2 and parts[1].startswith("wf_"):
            return parts[1]  # Return 'wf_1'
    return None


async def _get_workflow_name(workflow_id: Optional[str]) -> Optional[str]:
    """Lookup workflow name from database."""
    if not workflow_id:
        return None

    try:
        pk = parse_workflow_public_id(workflow_id)
    except ValueError as exc:
        logger.debug(f"Invalid workflow public id '{workflow_id}': {exc}")
        return None

    try:
        workflow = await Workflow.get_or_none(id=pk)
        if workflow:
            return workflow.name
    except Exception as exc:
        logger.debug(f"Could not lookup workflow name for {workflow_id}: {exc}")

    return None


def _extract_title_from_message(msg: Any) -> str:
    """Extract content from a message."""
    if isinstance(msg, dict):
        content = msg.get("content", "")
    elif hasattr(msg, "content"):
        content = str(msg.content)
    else:
        content = str(msg)
    return content[:100].strip() if content else ""


def _extract_metadata_from_checkpoint(
    checkpoint: Dict[str, Any],
    channel_values: Dict[str, Any],
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """Extract metadata (workflow_id, title) from checkpoint."""
    metadata = {}

    checkpoint_metadata = checkpoint.get("metadata", {})
    if isinstance(checkpoint_metadata, dict):
        metadata["workflow_id"] = checkpoint_metadata.get("workflow_id")
        metadata["title"] = checkpoint_metadata.get("title")

    if "metadata" in channel_values and isinstance(channel_values["metadata"], dict):
        if not metadata.get("workflow_id"):
            metadata["workflow_id"] = channel_values["metadata"].get("workflow_id")
        if not metadata.get("title"):
            metadata["title"] = channel_values["metadata"].get("title")

    if not metadata.get("workflow_id") and thread_id:
        extracted_id = _extract_workflow_id_from_thread_id(thread_id)
        if extracted_id:
            metadata["workflow_id"] = extracted_id

    if not metadata.get("title"):
        messages = channel_values.get("messages", [])
        if messages and isinstance(messages, list):
            metadata["title"] = _extract_title_from_message(messages[0])

    return metadata


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("", response_model=AgentTraceListResponse)
def _parse_checkpoint_timestamp(ts_str: str) -> datetime:
    """Parse checkpoint timestamp."""
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except Exception:
        return datetime.utcnow()


def _should_skip_thread(thread_id: Optional[str]) -> bool:
    """Check if thread should be skipped."""
    return not thread_id or thread_id.startswith("run_")


def _update_thread_timestamps(thread_data: Dict[str, Any], timestamp: datetime):
    """Update earliest and latest timestamps for thread."""
    if timestamp < thread_data["earliest_ts"]:
        thread_data["earliest_ts"] = timestamp
    if timestamp > thread_data["latest_ts"]:
        thread_data["latest_ts"] = timestamp


async def _build_trace_summary(thread_id: str, data: Dict[str, Any]) -> AgentTraceSummary:
    """Build trace summary from thread data."""
    latest_checkpoint = max(data["checkpoints"], key=lambda c: c["timestamp"])["checkpoint"]
    channel_values = latest_checkpoint.get("channel_values", {})

    messages = channel_values.get("messages", [])
    message_count = len(messages) if isinstance(messages, list) else 0

    metadata = _extract_metadata_from_checkpoint(latest_checkpoint, channel_values, thread_id)
    workflow_id = metadata.get("workflow_id")
    workflow_name = await _get_workflow_name(workflow_id)

    return AgentTraceSummary(
        thread_id=thread_id,
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        message_count=message_count,
        created_at=data["earliest_ts"].isoformat(),
        updated_at=data["latest_ts"].isoformat(),
        title=metadata.get("title"),
    )


async def list_agent_traces(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AgentTraceListResponse:
    """
    List all agent traces (excluding workflow executions).

    Returns paginated list of agent conversation traces.
    """
    checkpointer = await get_checkpointer()
    if not checkpointer:
        logger.warning("Checkpointer not available, returning empty trace list")
        return AgentTraceListResponse(traces=[], total=0)

    try:
        thread_data: Dict[str, Dict[str, Any]] = {}

        async for checkpoint_tuple in checkpointer.alist({}):
            thread_id = checkpoint_tuple.config.get("configurable", {}).get("thread_id")
            if _should_skip_thread(thread_id):
                continue

            checkpoint = checkpoint_tuple.checkpoint
            timestamp = _parse_checkpoint_timestamp(checkpoint.get("ts", ""))

            if thread_id not in thread_data:
                thread_data[thread_id] = {
                    "thread_id": thread_id,
                    "checkpoints": [],
                    "earliest_ts": timestamp,
                    "latest_ts": timestamp,
                }

            thread_data[thread_id]["checkpoints"].append({
                "checkpoint": checkpoint,
                "timestamp": timestamp,
            })
            _update_thread_timestamps(thread_data[thread_id], timestamp)

        traces = []
        for thread_id, data in thread_data.items():
            trace = await _build_trace_summary(thread_id, data)
            traces.append(trace)

        traces.sort(key=lambda t: t.updated_at, reverse=True)

        total = len(traces)
        paginated_traces = traces[offset:offset + limit]

        return AgentTraceListResponse(traces=paginated_traces, total=total)

    except Exception as e:
        logger.error(f"Error listing agent traces: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list traces: {str(e)}") from e


@router.get("/{thread_id}", response_model=AgentTraceDetail)
async def _find_earliest_timestamp(checkpointer, config: dict, default: datetime) -> datetime:
    """Find earliest timestamp from all checkpoints."""
    earliest_ts = default
    async for checkpoint_tuple in checkpointer.alist(config):
        cp_ts_str = checkpoint_tuple.checkpoint.get("ts", "")
        try:
            cp_ts = datetime.fromisoformat(cp_ts_str.replace('Z', '+00:00'))
            earliest_ts = min(earliest_ts, cp_ts)
        except Exception:
            pass
    return earliest_ts


def _convert_messages(messages_raw: Any, ts_str: str) -> List[AgentMessage]:
    """Convert raw messages to AgentMessage objects."""
    messages = []
    for idx, msg in enumerate(messages_raw if isinstance(messages_raw, list) else []):
        agent_msg = _convert_message_to_agent_message(msg, idx, ts_str)
        messages.append(agent_msg)
    return messages


async def get_agent_trace(
    request: Request,
    thread_id: str,
) -> AgentTraceDetail:
    """
    Get detailed agent trace with messages.

    Returns full trace detail including all messages.
    """
    checkpointer = await get_checkpointer()
    if not checkpointer:
        raise HTTPException(status_code=503, detail="Checkpointer not available")

    try:
        config = {"configurable": {"thread_id": thread_id}}
        state_tuple = await checkpointer.aget_tuple(config)

        if not state_tuple:
            raise HTTPException(
                status_code=404,
                detail=f"Trace not found for thread_id: {thread_id}"
            )

        checkpoint = state_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        timestamp = _parse_checkpoint_timestamp(checkpoint.get("ts", ""))

        earliest_ts = await _find_earliest_timestamp(checkpointer, config, timestamp)
        messages = _convert_messages(channel_values.get("messages", []), checkpoint.get("ts", ""))

        metadata = _extract_metadata_from_checkpoint(checkpoint, channel_values, thread_id)
        workflow_name = await _get_workflow_name(metadata.get("workflow_id"))

        return AgentTraceDetail(
            thread_id=thread_id,
            workflow_id=metadata.get("workflow_id"),
            workflow_name=workflow_name,
            created_at=earliest_ts.isoformat(),
            updated_at=timestamp.isoformat(),
            title=metadata.get("title"),
            messages=messages,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent trace: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get trace: {str(e)}") from e
