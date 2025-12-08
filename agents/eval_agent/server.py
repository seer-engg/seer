#!/usr/bin/env python
"""
LangServe server for deploying the Eval Agent.

This wraps the LangGraph eval agent in a FastAPI server using LangServe,
providing the same streaming and API capabilities as LangGraph Cloud, but with
full control over the infrastructure.

Usage:
    python agents/eval_agent/server.py

Or with uvicorn:
    uvicorn agents.eval_agent.server:app --host 0.0.0.0 --port 8000

Environment Variables:
    OPENAI_API_KEY: Required - OpenAI API key
    E2B_API_KEY: Required - E2B sandbox API key
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD: Optional - Neo4j connection
    LANGSMITH_API_KEY, LANGSMITH_PROJECT: Optional - LangSmith tracing
    PORT: Optional - Server port (default: 8000)
    EVAL_PLAN_ONLY_MODE: Optional - Enable plan-only mode (default: True for server)
      Set to "false" to enable execution mode (requires GitHub URL in requests)
"""
import os
import sys
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

# Add project root to path so we can import seer modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Import config FIRST to load and validate environment variables
# This must happen before any other imports that depend on env vars
from shared.config import config  # noqa: E402

# Enable plan-only mode by default for server (frontend use case)
# Can be overridden via EVAL_PLAN_ONLY_MODE env var
if not os.getenv("EVAL_PLAN_ONLY_MODE"):
    config.eval_plan_only_mode = True
    print("📋 Plan-only mode enabled by default (server mode)")
else:
    print(f"📋 Plan-only mode: {config.eval_plan_only_mode} (from env var EVAL_PLAN_ONLY_MODE)")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the supervisor graph (new Supervisor pattern)
try:
    from agents.eval_agent.supervisor import graph
    logger.info("✅ Successfully imported eval agent supervisor graph")
except Exception as e:
    logger.error(f"❌ Failed to import supervisor graph: {e}", exc_info=True)
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Eval Agent API",
    version="1.0.0",
    description="Seer Eval Agent - Multi-Agent System for Evaluating AI Agents",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add LangServe routes
# This automatically creates endpoints:
# - POST /agent/invoke - Single invocation
# - POST /agent/stream - Streaming invocation
# - POST /agent/batch - Batch invocation
# - GET /agent/playground - Interactive playground (if enabled)

def per_req_config_modifier(config_dict: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """
    Modify the config for each request.
    Ensures thread_id exists in configurable to prevent Checkpointer errors.
    Also initializes state with default values for Supervisor pattern.
    """
    if "configurable" not in config_dict:
        config_dict["configurable"] = {}
    
    # Check if thread_id is missing
    if "thread_id" not in config_dict["configurable"]:
        # Generate a new ID if missing
        generated_id = str(uuid.uuid4())
        config_dict["configurable"]["thread_id"] = generated_id
        logger.warning(f"⚠️ thread_id missing in request config. Generated fallback ID: {generated_id}")
    else:
        logger.info(f"Using provided thread_id: {config_dict['configurable']['thread_id']}")
    
    # Initialize state with defaults for Supervisor pattern if input is provided
    if "input" in config_dict:
        input_data = config_dict["input"]
        if isinstance(input_data, dict):
            # Ensure todos and plan_only_mode are set
            if "todos" not in input_data:
                input_data["todos"] = []
            if "plan_only_mode" not in input_data:
                input_data["plan_only_mode"] = config.eval_plan_only_mode
            if "tool_call_counts" not in input_data:
                input_data["tool_call_counts"] = {}
            if "current_phase" not in input_data:
                input_data["current_phase"] = None
        
    return config_dict

add_routes(
    app,
    graph,
    path="/agent",
    # Enable playground for testing (disable in production)
    playground_type="chat",
    per_req_config_modifier=per_req_config_modifier,
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with basic status."""
    # Check critical environment variables
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "E2B_API_KEY": os.getenv("E2B_API_KEY"),
    }
    
    optional_vars = {
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
    }
    
    all_required_set = all(required_vars.values())
    status = "healthy" if all_required_set else "unhealthy"
    
    response = {
        "status": status,
        "service": "eval-agent",
        "version": "1.0.0",
        "required_variables": {k: bool(v) for k, v in required_vars.items()},
        "optional_variables": {k: bool(v) for k, v in optional_vars.items()},
    }
    
    if not all_required_set:
        missing = [var for var, val in required_vars.items() if not val]
        response["missing_variables"] = missing
        response["error"] = f"Missing required environment variables: {', '.join(missing)}"
    
    return response

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Eval Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "agent_endpoint": "/agent",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port and host from environment (Railway sets PORT automatically)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting Eval Agent API server on {host}:{port}")
    logger.info(f"📚 API docs available at http://{host}:{port}/docs")
    logger.info(f"🤖 Agent endpoint: http://{host}:{port}/agent")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

