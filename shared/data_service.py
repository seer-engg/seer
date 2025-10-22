"""
FastAPI Data Service for Seer
Provides simple REST APIs for data operations without LLM overhead
"""

import os
import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import traceback
from datetime import datetime

from seer.agents.orchestrator.data_manager import DataManager
from seer.shared.config import get_seer_config
from seer.shared.a2a_utils import send_a2a_message
from seer.shared.logger import get_logger
import uuid as _uuid
from langgraph_sdk import get_client

# Get logger for data service
logger = get_logger('data_service')

app = FastAPI(title="Seer Data Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data manager
data_manager = DataManager()


# Request/Response Models
class GetEvalSuitesRequest(BaseModel):
    agent_url: Optional[str] = None
    agent_id: Optional[str] = None


class GetTestResultsRequest(BaseModel):
    suite_id: Optional[str] = None
    thread_id: Optional[str] = None


class StoreEvalSuiteRequest(BaseModel):
    suite_id: str
    spec_name: str
    spec_version: str
    test_cases: List[Dict[str, Any]]
    target_agent_url: str
    target_agent_id: str
    thread_id: Optional[str] = None
    langgraph_thread_id: Optional[str] = None


class StoreTestResultsRequest(BaseModel):
    suite_id: str
    thread_id: str
    results: List[Dict[str, Any]]


class SendMessageRequest(BaseModel):
    """Request to send a message to an agent"""
    content: str
    thread_id: Optional[str] = None
    target_agent: str = "orchestrator"  # orchestrator, eval_agent, coding_agent


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "seer-data-service"}


# Thread operations
@app.get("/threads")
async def get_all_threads():
    """Get all conversation threads"""
    try:
        threads = data_manager.get_all_threads()
        return {
            "success": True,
            "threads": threads,
            "count": len(threads)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/messages")
async def get_conversation_history(thread_id: str):
    """Get conversation history for a specific thread"""
    try:
        messages = data_manager.get_conversation_history(thread_id)
        return {
            "success": True,
            "thread_id": thread_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/config")
async def get_thread_config(thread_id: str):
    """Get target agent configuration for a specific thread"""
    try:
        config = data_manager.get_target_agent_config(thread_id)
        return {
            "success": True,
            "thread_id": thread_id,
            "config": config
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/expectations")
async def get_thread_expectations(thread_id: str):
    """Get target agent expectations for a specific thread"""
    try:
        expectations = data_manager.get_target_agent_expectation(thread_id)
        return {
            "success": True,
            "thread_id": thread_id,
            "expectations": expectations
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/eval-suites")
async def get_thread_eval_suites(thread_id: str):
    """Get evaluation suites linked to a specific thread"""
    try:
        # Query eval suites with thread_id filter
        from seer.shared.models import EvalSuite, db
        db.connect(reuse_if_open=True)
        suites = list(EvalSuite.select().where(EvalSuite.thread == thread_id).dicts())
        return {
            "success": True,
            "thread_id": thread_id,
            "suites": suites,
            "count": len(suites)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Eval suite operations
@app.get("/eval-suites")
async def get_eval_suites(agent_url: Optional[str] = None, agent_id: Optional[str] = None):
    """Get evaluation suites with optional filters"""
    try:
        suites = data_manager.get_eval_suites(agent_url=agent_url, agent_id=agent_id)
        return {
            "success": True,
            "suites": suites,
            "count": len(suites)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval-suites")
async def store_eval_suite(request: StoreEvalSuiteRequest):
    """Store an evaluation suite"""
    try:
        suite_id = data_manager.store_eval_suite({
            "suite_id": request.suite_id,
            "spec_name": request.spec_name,
            "spec_version": request.spec_version,
            "test_cases": request.test_cases,
            "target_agent_url": request.target_agent_url,
            "target_agent_id": request.target_agent_id,
            "thread_id": request.thread_id,
            "langgraph_thread_id": request.langgraph_thread_id
        })
        return {
            "success": True,
            "suite_id": suite_id,
            "message": f"Eval suite {suite_id} stored successfully"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Test results operations
@app.get("/test-results")
async def get_test_results(suite_id: Optional[str] = None, thread_id: Optional[str] = None):
    """Get test results with optional filters"""
    try:
        results = data_manager.get_test_results(suite_id=suite_id, thread_id=thread_id)
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test-results")
async def store_test_results(request: StoreTestResultsRequest):
    """Store test results"""
    try:
        result = data_manager.store_test_results(
            suite_id=request.suite_id,
            thread_id=request.thread_id,
            results=request.results
        )
        return {
            "success": True,
            "message": "Test results stored successfully",
            "results_count": result.get("results_count", len(request.results))
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Agent registry operations
@app.get("/agents")
async def get_registered_agents():
    """Get all registered agents"""
    try:
        from seer.shared.database import get_db
        db = get_db()
        agents = db.get_subscribers()
        return {
            "success": True,
            "agents": agents,
            "count": len(agents)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Message proxy operations (UI → Agents)
@app.post("/messages/send")
async def send_message_to_agent(request: SendMessageRequest):
    """
    Proxy endpoint: UI sends messages through this FastAPI backend
    This ensures proper thread creation, message persistence, and data flow
    """
    try:
        config = get_seer_config()
        
        # Get agent configuration
        from seer.shared.config import get_config
        agent_config = get_config()
        
        # Determine target agent port
        if request.target_agent == "orchestrator":
            target_port = config.orchestrator_port
            graph_id = agent_config.get_graph_name("orchestrator")
        elif request.target_agent == "eval_agent":
            target_port = config.eval_agent_port
            graph_id = agent_config.get_graph_name("eval_agent")
        elif request.target_agent == "coding_agent":
            target_port = config.coding_agent_port
            graph_id = agent_config.get_graph_name("coding_agent")
        else:
            raise ValueError(f"Unknown target agent: {request.target_agent}")
        
        # 1. Get thread_id from request (None for new conversations)
        thread_id = request.thread_id
        
        # For orchestrator: Use LangGraph Client SDK (proper thread management)
        # For eval/coding agents: Use A2A protocol (agent-to-agent communication)
        
        final_response = ""

        # 2. Send to target agent
        if request.target_agent == "orchestrator":
            # Use LangGraph Client SDK for UI → Orchestrator (not A2A!)
            client = get_client(url=f"http://127.0.0.1:{target_port}")
            
            # Create or get thread
            if not thread_id:
                # First message - create new thread
                thread_response = await client.threads.create()
                thread_id = thread_response["thread_id"]
            
            # Send message and stream response
            final_response = ""
            async for chunk in client.runs.stream(
                thread_id=thread_id,
                assistant_id=graph_id,
                input={"messages": [{"role": "user", "content": request.content}]},
                stream_mode="values"
            ):
                if chunk.event == "values":
                    # Extract assistant message from last message in state
                    messages = chunk.data.get("messages", [])
                    # LangGraph messages use 'type' field: 'human' for user, 'ai' for assistant
                    if messages and messages[-1].get("type") == "ai":
                        final_response = messages[-1].get("content", "")
        else:
            # A2A for eval/coding by graph_name; pass thread_id (None for new, UUID for existing)
            a2a_resp = await send_a2a_message(
                target_agent_id=graph_id,
                target_port=target_port,
                message=request.content,
                thread_id=thread_id  # Can be None for first message
            )
            try:
                a2a_data = json.loads(a2a_resp)
            except Exception:
                a2a_data = {"success": False, "error": "Invalid A2A response"}
            if not a2a_data.get("success"):
                raise HTTPException(status_code=502, detail=a2a_data.get("error", "A2A failed"))
            final_response = a2a_data.get("response", "") or a2a_data.get("result", {}).get("response", "")
            
            # Get thread_id from A2A response (created on first message, same on subsequent)
            server_tid = a2a_data.get("thread_id")
            if server_tid:
                thread_id = server_tid  # Use the thread_id from A2A
        
        # 3. NOW that we have the thread_id from A2A, persist messages to database
        if not thread_id:
            raise HTTPException(status_code=500, detail="No thread_id returned from agent")
        
        # Ensure thread exists in database
        from seer.shared.database import get_db
        db = get_db()
        db.create_thread(thread_id)
        
        # Persist user message
        message_id = str(uuid.uuid4())
        db.add_message(
            thread_id=thread_id,
            message_id=message_id,
            timestamp=datetime.now().isoformat(),
            role="user",
            sender="ui",
            content=request.content,
            message_type="conversation"
        )
        
        # Persist assistant response
        if final_response:
            assistant_message_id = str(uuid.uuid4())
            db.add_message(
                thread_id=thread_id,
                message_id=assistant_message_id,
                timestamp=datetime.now().isoformat(),
                role="assistant",
                sender=request.target_agent,
                content=final_response,
                message_type="conversation"
            )

        # 4. Return response with thread_id
        return {
            "success": True,
            "response": final_response or "No response received",
            "thread_id": thread_id,
            "message_id": message_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def start_server(port: int = 8500):
    """Start the data service server"""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.getenv("DATA_SERVICE_PORT", "8500"))
    try:
        start_server(port)
    except Exception as e:
        logger.error(f"Error starting data service: {str(e)}")
        logger.error(traceback.format_exc())
