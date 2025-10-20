"""
FastAPI Data Service for Seer
Provides simple REST APIs for data operations without LLM overhead
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import traceback

from seer.agents.orchestrator.modules.data_manager import DataManager

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


def start_server(port: int = 8500):
    """Start the data service server"""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.getenv("DATA_SERVICE_PORT", "8500"))
    try:
        start_server(port)
    except Exception as e:
        print(f"Error starting data service: {str(e)}")
        traceback.print_exc()
