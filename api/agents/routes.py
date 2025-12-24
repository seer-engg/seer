from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from .checkpointer import get_checkpointer
from .models import ThreadCreate, ThreadResponse, ThreadState, RunInput
from shared.logger import get_logger
from datetime import datetime
import uuid
from typing import Dict, Any, AsyncGenerator
import json
from fastapi import APIRouter, Response, status, Request
from .graphs import get_compiled_graph, get_available_graphs


logger = get_logger("api.agents.routes")
# =============================================================================
# Thread Endpoints
# =============================================================================

router = APIRouter(prefix="/agents", tags=["Agents"])

@router.post("/threads", response_model=ThreadResponse, tags=["Threads"])
async def create_thread(request: ThreadCreate = None) -> ThreadResponse:
    """
    Create a new thread.
    
    A thread represents a conversation session. The thread_id is used to
    maintain state across multiple runs.
    
    Example:
        curl -X POST http://localhost:2024/threads
    """
    thread_id = str(uuid.uuid4())
    metadata = request.metadata if request else None
    
    logger.info(f"Created thread: {thread_id}")
    
    return ThreadResponse(
        thread_id=thread_id,
        created_at=datetime.utcnow(),
        metadata=metadata,
    )


@router.get("/threads/{thread_id}/state", response_model=ThreadState, tags=["Threads"])
async def get_thread_state(
    thread_id: str,
    graph_name: str = Query(
        default="eval_agent",
        description="Name of the graph to get state for"
    )
) -> ThreadState:
    """
    Get the latest state of a thread.
    
    Returns the current checkpoint state for the given thread and graph.
    
    Example:
        curl http://localhost:2024/threads/{thread_id}/state?graph_name=eval_agent
    """
    try:
        checkpointer = await get_checkpointer()
        
        # Get the latest checkpoint for this thread
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get the compiled graph to access state
        graph = await get_compiled_graph(graph_name)
        
        # Get state from the graph
        state = await graph.aget_state(config)
        
        if state is None or state.values is None:
            raise HTTPException(
                status_code=404,
                detail=f"No state found for thread {thread_id}"
            )
        
        # Convert state values to serializable format
        values = _serialize_state(state.values)
        
        return ThreadState(
            thread_id=thread_id,
            values=values,
            next=list(state.next) if state.next else [],
            checkpoint_id=state.config.get("configurable", {}).get("checkpoint_id") if state.config else None,
            created_at=state.created_at if hasattr(state, 'created_at') else None,
            parent_checkpoint_id=state.parent_config.get("configurable", {}).get("checkpoint_id") if state.parent_config else None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Run Endpoints
# =============================================================================

@router.post("/threads/{thread_id}/runs/stream", tags=["Runs"])
async def create_run_stream(
    thread_id: str,
    request: RunInput,
) -> StreamingResponse:
    """
    Create a run and stream the output.
    
    Executes the specified graph with the given input and streams
    events back to the client using Server-Sent Events (SSE).
    
    Example:
        curl -X POST http://localhost:2024/threads/{thread_id}/runs/stream \\
            -H "Content-Type: application/json" \\
            -d '{"graph_name": "eval_agent", "input": {"messages": [{"role": "user", "content": "Hello"}]}}'
    """
    try:
        # Validate graph exists
        graph = await get_compiled_graph(request.graph_name)
        
        # Build config with thread_id
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        # Merge additional config if provided
        if request.config:
            config.update(request.config)
        
        # Prepare input - convert message dicts to LangChain messages if needed
        input_data = _prepare_input(request.input)
        
        logger.info(f"Starting run for thread {thread_id} on graph {request.graph_name}")
        
        return StreamingResponse(
            _stream_graph_events(graph, input_data, config, request.stream_mode),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_graph_events(
    graph,
    input_data: Dict[str, Any],
    config: Dict[str, Any],
    stream_mode: str,
) -> AsyncGenerator[str, None]:
    """
    Stream graph execution events as SSE.
    
    Yields events in the format:
        event: {event_type}
        data: {json_data}
    """
    try:
        if stream_mode == "events":
            # Use astream_events for full event stream
            async for event in graph.astream_events(input_data, config=config, version="v2"):
                yield _format_sse_event(event["event"], event)
        
        elif stream_mode == "messages":
            # Stream LLM tokens
            async for event in graph.astream_events(input_data, config=config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        yield _format_sse_event("token", {"content": chunk.content})
                elif event["event"] in ("on_chain_start", "on_chain_end"):
                    yield _format_sse_event(event["event"], {
                        "name": event.get("name"),
                        "run_id": event.get("run_id"),
                    })
        
        elif stream_mode == "updates":
            # Stream state updates
            async for chunk in graph.astream(input_data, config=config, stream_mode="updates"):
                for node_name, update in chunk.items():
                    serialized = _serialize_state(update)
                    yield _format_sse_event("update", {
                        "node": node_name,
                        "data": serialized,
                    })
        
        else:  # stream_mode == "values"
            # Stream full state values
            async for state in graph.astream(input_data, config=config, stream_mode="values"):
                serialized = _serialize_state(state)
                yield _format_sse_event("state", serialized)
        
        # Send completion event
        yield _format_sse_event("done", {"status": "completed"})
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        yield _format_sse_event("error", {"detail": str(e)})


def _format_sse_event(event_type: str, data: Any) -> str:
    """Format data as a Server-Sent Event."""
    try:
        json_data = json.dumps(data, default=str)
    except (TypeError, ValueError) as e:
        json_data = json.dumps({"error": f"Serialization error: {e}"})
    
    return f"event: {event_type}\ndata: {json_data}\n\n"


def _prepare_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare input data by converting message dicts to LangChain messages.
    """
    if "messages" in input_data:
        messages = input_data["messages"]
        converted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    converted.append(SystemMessage(content=content))
                elif role == "assistant":
                    converted.append(AIMessage(content=content))
                else:
                    converted.append(HumanMessage(content=content))
            else:
                # Already a LangChain message
                converted.append(msg)
        
        input_data["messages"] = converted
    
    return input_data


def _serialize_state(state: Any) -> Dict[str, Any]:
    """
    Serialize state to JSON-compatible format.
    
    Handles LangChain messages and Pydantic models.
    """
    if state is None:
        return {}
    
    if isinstance(state, dict):
        result = {}
        for key, value in state.items():
            result[key] = _serialize_value(value)
        return result
    
    if hasattr(state, "model_dump"):
        return _serialize_state(state.model_dump())
    
    if hasattr(state, "dict"):
        return _serialize_state(state.dict())
    
    return {"value": str(state)}


def _serialize_value(value: Any) -> Any:
    """Serialize a single value."""
    if value is None:
        return None
    
    if isinstance(value, (str, int, float, bool)):
        return value
    
    if isinstance(value, datetime):
        return value.isoformat()
    
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    
    # Handle LangChain messages
    if hasattr(value, "content") and hasattr(value, "type"):
        return {
            "type": value.type,
            "content": value.content,
            "additional_kwargs": getattr(value, "additional_kwargs", {}),
        }
    
    # Handle Pydantic models
    if hasattr(value, "model_dump"):
        return _serialize_value(value.model_dump())
    
    if hasattr(value, "dict"):
        return _serialize_value(value.dict())
    
    # Fallback to string
    return str(value)



@router.get("/graphs", tags=["System"])
async def list_graphs():
    """List available graphs."""
    return {"graphs": get_available_graphs()}

