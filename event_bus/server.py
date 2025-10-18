"""
Event Bus Server - FastAPI-based message broker

Simple in-memory pub/sub system for agent communication.
"""

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import asyncio
from collections import defaultdict
from datetime import datetime
import uvicorn
import os
import logging
from pathlib import Path

from .schemas import EventMessage

# Setup event bus logger
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)
event_bus_logger = logging.getLogger("event_bus")
event_bus_logger.setLevel(logging.INFO)

# File handler for detailed event log
file_handler = logging.FileHandler(logs_dir / "event_bus.log", mode='w')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
event_bus_logger.addHandler(file_handler)

# Console handler for visibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
event_bus_logger.addHandler(console_handler)

app = FastAPI(
    title="Seer Event Bus",
    description="Message broker for multi-agent communication",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API key authentication
API_KEY = os.getenv("SEER_API_KEY", "seer-default-key")
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for protected endpoints"""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials


# In-memory storage
class EventBusState:
    """Global state for the event bus"""
    def __init__(self):
        self.messages: list[EventMessage] = []  # All messages ever published
        self.subscribers: dict[str, dict] = {}  # agent_name -> {last_poll_time, filters}
        self.message_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.evals: dict[str, list[dict]] = {}  # agent_key -> [eval_suites]
        # agent_key format: "url:port/agent_id" e.g. "http://localhost:2024/deep_researcher"
        
    def add_message(self, message: EventMessage):
        """Add a message to history and route to subscribers"""
        self.messages.append(message)
        # Route to all non-blocked subscribers (they'll filter on their end)
        for agent_name, subscriber_info in self.subscribers.items():
            if not subscriber_info.get("blocked", False):
                self.message_queues[agent_name].put_nowait(message)
    
    def register_subscriber(self, agent_name: str, filters: Optional[dict] = None):
        """Register an agent as a subscriber"""
        self.subscribers[agent_name] = {
            "registered_at": datetime.now(),
            "filters": filters or {},
            "last_poll": datetime.now(),
            "message_count": 0,
            "publish_count": 0,
            "suspicious_score": 0
        }
        if agent_name not in self.message_queues:
            self.message_queues[agent_name] = asyncio.Queue()
    
    def get_history(
        self, 
        since: Optional[datetime] = None,
        event_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 100
    ) -> list[EventMessage]:
        """Get message history with optional filters"""
        filtered = self.messages
        
        if since:
            filtered = [m for m in filtered if m.timestamp > since]
        if event_type:
            filtered = [m for m in filtered if m.event_type == event_type]
        if thread_id:
            filtered = [m for m in filtered if m.thread_id == thread_id]
        
        return filtered[-limit:]
    
    def detect_rogue_agents(self) -> list[dict]:
        """Detect potentially rogue agents based on behavior patterns"""
        suspicious_agents = []
        
        for agent_name, info in self.subscribers.items():
            suspicious_score = 0
            reasons = []
            
            # Check message frequency (too many messages)
            if info.get("message_count", 0) > 100:  # Threshold
                suspicious_score += 30
                reasons.append("High message volume")
            
            # Check publish frequency (too many publishes)
            if info.get("publish_count", 0) > 50:  # Threshold
                suspicious_score += 40
                reasons.append("High publish rate")
            
            # Check for unusual agent names
            if any(keyword in agent_name.lower() for keyword in ["test", "hack", "rogue", "malicious"]):
                suspicious_score += 50
                reasons.append("Suspicious agent name")
            
            # Check for agents that never poll (might be passive listeners)
            last_poll = info.get("last_poll")
            if last_poll and isinstance(last_poll, str):
                try:
                    last_poll_dt = datetime.fromisoformat(last_poll.replace('Z', '+00:00'))
                    time_since_poll = datetime.now() - last_poll_dt
                    if time_since_poll.total_seconds() > 300:  # 5 minutes
                        suspicious_score += 20
                        reasons.append("Inactive polling")
                except:
                    pass
            
            if suspicious_score > 50:  # Threshold for flagging
                suspicious_agents.append({
                    "agent_name": agent_name,
                    "suspicious_score": suspicious_score,
                    "reasons": reasons,
                    "info": info
                })
        
        return suspicious_agents
    
    def store_eval_suite(self, agent_url: str, agent_id: str, eval_suite: dict):
        """Store an eval suite for a target agent"""
        agent_key = f"{agent_url}/{agent_id}"
        if agent_key not in self.evals:
            self.evals[agent_key] = []
        self.evals[agent_key].append(eval_suite)
    
    def get_evals_by_agent(self, agent_url: str, agent_id: str) -> list[dict]:
        """Get all eval suites for a specific agent"""
        agent_key = f"{agent_url}/{agent_id}"
        return self.evals.get(agent_key, [])
    
    def get_eval_by_id(self, eval_suite_id: str) -> Optional[dict]:
        """Get a specific eval suite by ID"""
        for agent_key, suites in self.evals.items():
            for suite in suites:
                if suite.get("id") == eval_suite_id:
                    return suite
        return None


state = EventBusState()


@app.get("/")
async def root():
    """Health check and stats"""
    return {
        "status": "running",
        "total_messages": len(state.messages),
        "active_subscribers": len(state.subscribers),
        "subscribers": list(state.subscribers.keys())
    }


@app.post("/publish")
async def publish_message(message: EventMessage):
    """
    Publish a message to the event bus.
    All subscribers will receive it.
    """
    # Track publish count for sender
    if message.sender in state.subscribers:
        state.subscribers[message.sender]["publish_count"] = state.subscribers[message.sender].get("publish_count", 0) + 1
    
    state.add_message(message)
    
    # Log the publish event
    event_bus_logger.info(f"✉️  PUBLISH | {message.sender} → [{message.event_type}] | thread={message.thread_id or 'None'}")
    
    return {
        "status": "published",
        "message_id": message.message_id,
        "timestamp": message.timestamp
    }


@app.post("/subscribe/{agent_name}")
async def subscribe(
    agent_name: str,
    filters: Optional[dict] = None
):
    """
    Register an agent as a subscriber.
    
    Args:
        agent_name: Unique identifier for the agent
        filters: Optional filters (not used in v1, agents filter themselves)
    """
    state.register_subscriber(agent_name, filters)
    
    # Log the subscription
    event_bus_logger.info(f"🔌 SUBSCRIBE | {agent_name} joined the event bus")
    
    return {
        "status": "subscribed",
        "agent_name": agent_name,
        "message": f"Agent '{agent_name}' subscribed to event bus"
    }


@app.get("/poll/{agent_name}")
async def poll_messages(
    agent_name: str,
    timeout: int = Query(default=30, ge=1, le=60, description="Polling timeout in seconds"),
    batch_size: int = Query(default=10, ge=1, le=100)
):
    """
    Long-polling endpoint for agents to receive messages.
    
    Returns:
        List of messages for this agent (up to batch_size)
    """
    if agent_name not in state.subscribers:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not subscribed. Call /subscribe first."
        )
    
    state.subscribers[agent_name]["last_poll"] = datetime.now()
    state.subscribers[agent_name]["message_count"] = state.subscribers[agent_name].get("message_count", 0) + 1
    queue = state.message_queues[agent_name]
    
    messages = []
    try:
        # Long polling: wait up to timeout seconds for first message
        first_message = await asyncio.wait_for(queue.get(), timeout=timeout)
        messages.append(first_message)
        
        # Then grab any other messages immediately available (up to batch_size)
        for _ in range(batch_size - 1):
            try:
                msg = queue.get_nowait()
                messages.append(msg)
            except asyncio.QueueEmpty:
                break
                
    except asyncio.TimeoutError:
        # No messages within timeout - return empty list
        pass
    
    # Log poll activity (only if messages received)
    if messages:
        event_bus_logger.info(f"📥 POLL | {agent_name} received {len(messages)} message(s)")
    
    return {
        "messages": [m.model_dump() for m in messages],
        "count": len(messages)
    }


@app.get("/history")
async def get_history(
    since: Optional[str] = Query(default=None, description="ISO timestamp"),
    event_type: Optional[str] = None,
    thread_id: Optional[str] = None,
    limit: int = Query(default=100, le=1000)
):
    """
    Get message history with optional filters.
    Useful for debugging and UI display.
    """
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid ISO timestamp format")
    
    messages = state.get_history(
        since=since_dt,
        event_type=event_type,
        thread_id=thread_id,
        limit=limit
    )
    
    return {
        "messages": [m.model_dump() for m in messages],
        "count": len(messages),
        "total_in_system": len(state.messages)
    }


@app.get("/threads")
async def get_threads():
    """Get all active conversation threads"""
    threads = {}
    for msg in state.messages:
        if msg.thread_id:
            if msg.thread_id not in threads:
                threads[msg.thread_id] = {
                    "thread_id": msg.thread_id,
                    "message_count": 0,
                    "first_message": msg.timestamp,
                    "last_message": msg.timestamp
                }
            threads[msg.thread_id]["message_count"] += 1
            threads[msg.thread_id]["last_message"] = max(
                threads[msg.thread_id]["last_message"],
                msg.timestamp
            )
    
    return {"threads": list(threads.values())}


@app.delete("/clear")
async def clear_history():
    """Clear all message history (for testing)"""
    state.messages.clear()
    for queue in state.message_queues.values():
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    return {"status": "cleared", "message": "All message history cleared"}


@app.get("/subscribers")
async def get_subscribers(api_key: str = Depends(verify_api_key)):
    """Get all active subscribers with their details"""
    return {
        "subscribers": state.subscribers,
        "count": len(state.subscribers)
    }


@app.delete("/subscribers/{agent_name}")
async def remove_subscriber(agent_name: str, api_key: str = Depends(verify_api_key)):
    """Remove a subscriber from the event bus"""
    if agent_name not in state.subscribers:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )
    
    # Remove from subscribers
    del state.subscribers[agent_name]
    
    # Clear their message queue
    if agent_name in state.message_queues:
        queue = state.message_queues[agent_name]
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        del state.message_queues[agent_name]
    
    return {
        "status": "removed",
        "agent_name": agent_name,
        "message": f"Agent '{agent_name}' removed from event bus"
    }


@app.post("/subscribers/{agent_name}/block")
async def block_subscriber(agent_name: str, api_key: str = Depends(verify_api_key)):
    """Block a subscriber (prevent them from receiving new messages)"""
    if agent_name not in state.subscribers:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )
    
    # Mark as blocked
    state.subscribers[agent_name]["blocked"] = True
    state.subscribers[agent_name]["blocked_at"] = datetime.now()
    
    return {
        "status": "blocked",
        "agent_name": agent_name,
        "message": f"Agent '{agent_name}' blocked from receiving messages"
    }


@app.post("/subscribers/{agent_name}/unblock")
async def unblock_subscriber(agent_name: str, api_key: str = Depends(verify_api_key)):
    """Unblock a subscriber"""
    if agent_name not in state.subscribers:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )
    
    # Remove block
    state.subscribers[agent_name].pop("blocked", None)
    state.subscribers[agent_name].pop("blocked_at", None)
    
    return {
        "status": "unblocked",
        "agent_name": agent_name,
        "message": f"Agent '{agent_name}' unblocked"
    }


@app.get("/rogue-detection")
async def detect_rogue_agents(api_key: str = Depends(verify_api_key)):
    """Detect potentially rogue agents"""
    suspicious_agents = state.detect_rogue_agents()
    return {
        "suspicious_agents": suspicious_agents,
        "count": len(suspicious_agents),
        "detection_time": datetime.now().isoformat()
    }


@app.post("/evals")
async def store_eval_suite(payload: dict):
    """
    Store an eval suite for a target agent.
    
    Payload should include:
    - eval_suite: The eval suite object with test cases
    - target_agent_url: URL of the agent being evaluated
    - target_agent_id: ID of the agent being evaluated
    """
    eval_suite = payload.get("eval_suite")
    target_agent_url = payload.get("target_agent_url")
    target_agent_id = payload.get("target_agent_id")
    
    if not eval_suite or not target_agent_url or not target_agent_id:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: eval_suite, target_agent_url, target_agent_id"
        )
    
    state.store_eval_suite(target_agent_url, target_agent_id, eval_suite)
    
    event_bus_logger.info(
        f"📋 EVAL_STORED | {target_agent_url}/{target_agent_id} | suite_id={eval_suite.get('id')}"
    )
    
    return {
        "status": "stored",
        "eval_suite_id": eval_suite.get("id"),
        "agent_key": f"{target_agent_url}/{target_agent_id}"
    }


@app.get("/evals")
async def get_eval_suites(
    agent_url: Optional[str] = Query(default=None, description="Target agent URL"),
    agent_id: Optional[str] = Query(default=None, description="Target agent ID")
):
    """
    Get eval suites for a specific agent.
    If no agent specified, returns all evals.
    """
    if agent_url and agent_id:
        evals = state.get_evals_by_agent(agent_url, agent_id)
        return {
            "agent_key": f"{agent_url}/{agent_id}",
            "eval_suites": evals,
            "count": len(evals)
        }
    else:
        # Return all evals
        all_evals = []
        for agent_key, suites in state.evals.items():
            for suite in suites:
                all_evals.append({
                    "agent_key": agent_key,
                    "eval_suite": suite
                })
        return {
            "eval_suites": all_evals,
            "count": len(all_evals)
        }


@app.get("/evals/{eval_suite_id}")
async def get_eval_suite_by_id(eval_suite_id: str):
    """Get a specific eval suite by its ID"""
    eval_suite = state.get_eval_by_id(eval_suite_id)
    
    if not eval_suite:
        raise HTTPException(
            status_code=404,
            detail=f"Eval suite '{eval_suite_id}' not found"
        )
    
    return {
        "eval_suite": eval_suite
    }


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the event bus server"""
    event_bus_logger.info("=" * 60)
    event_bus_logger.info(f"🚌 Event Bus Starting on http://{host}:{port}")
    event_bus_logger.info(f"📝 Logging to: {logs_dir / 'event_bus.log'}")
    event_bus_logger.info("=" * 60)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()

