"""
Event Bus Server - FastAPI-based message broker

Pub/sub system for agent communication with SQLite persistence.
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
import sys

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .schemas import EventMessage
from shared.database import get_db, init_db

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


# Initialize database
db = init_db()

# In-memory storage (for real-time message queues)
class EventBusState:
    """Global state for the event bus"""
    def __init__(self):
        self.messages: list[EventMessage] = []  # Recent messages for queue routing
        self.subscribers: dict[str, dict] = {}  # agent_name -> {last_poll_time, filters}
        self.message_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        
    def add_message(self, message: EventMessage):
        """Add a message to history and route to subscribers"""
        self.messages.append(message)
        
        # Persist to database
        db.add_event(
            event_id=message.message_id,
            thread_id=message.thread_id,
            timestamp=message.timestamp.isoformat() if isinstance(message.timestamp, datetime) else message.timestamp,
            event_type=message.event_type,
            sender=message.sender,
            payload=message.payload
        )
        
        # Also persist as message if it's a MessageFromUser or MessageToUser
        if message.event_type in ["MessageFromUser", "MessageToUser"]:
            content = message.payload.get("content", "")
            role = "user" if message.event_type == "MessageFromUser" else "assistant"
            db.add_message(
                thread_id=message.thread_id,
                message_id=message.message_id,
                timestamp=message.timestamp.isoformat() if isinstance(message.timestamp, datetime) else message.timestamp,
                role=role,
                sender=message.sender,
                content=content,
                message_type=message.payload.get("message_type"),
                metadata=message.payload
            )
        
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
        
        # Persist to database
        db.register_subscriber(agent_name, filters)
    
    def get_history(
        self, 
        since: Optional[datetime] = None,
        event_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """Get message history with optional filters from database"""
        # Get from database instead of in-memory
        since_str = since.isoformat() if since else None
        events = db.get_recent_events(limit=limit, event_type=event_type, since=since_str)
        
        # Filter by thread_id if provided
        if thread_id:
            events = [e for e in events if e.get('thread_id') == thread_id]
        
        # Convert to EventMessage-like dict format
        result = []
        for event in events:
            import json
            result.append({
                "message_id": event['event_id'],
                "event_type": event['event_type'],
                "sender": event['sender'],
                "thread_id": event['thread_id'],
                "timestamp": event['timestamp'],
                "payload": json.loads(event['payload']) if isinstance(event['payload'], str) else event['payload']
            })
        
        return list(reversed(result))  # Return in chronological order
    
    def detect_rogue_agents(self) -> list[dict]:
        """Detect potentially rogue agents based on behavior patterns"""
        suspicious_agents = []
        
        # Get subscribers from database
        subscribers = db.get_subscribers()
        
        for subscriber in subscribers:
            agent_name = subscriber['agent_name']
            suspicious_score = 0
            reasons = []
            
            # Check message frequency (too many messages)
            if subscriber.get("message_count", 0) > 100:  # Threshold
                suspicious_score += 30
                reasons.append("High message volume")
            
            # Check publish frequency (too many publishes)
            if subscriber.get("publish_count", 0) > 50:  # Threshold
                suspicious_score += 40
                reasons.append("High publish rate")
            
            # Check for unusual agent names
            if any(keyword in agent_name.lower() for keyword in ["test", "hack", "rogue", "malicious"]):
                suspicious_score += 50
                reasons.append("Suspicious agent name")
            
            # Check for agents that never poll (might be passive listeners)
            last_poll = subscriber.get("last_poll")
            if last_poll:
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
                    "info": subscriber
                })
        
        return suspicious_agents


state = EventBusState()


@app.get("/")
async def root():
    """Health check and stats"""
    subscribers = db.get_subscribers()
    events = db.get_recent_events(limit=1)
    return {
        "status": "running",
        "total_messages": len(events) if events else 0,
        "active_subscribers": len(subscribers),
        "subscribers": [s['agent_name'] for s in subscribers]
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
    
    # Update in database
    db.update_subscriber_publish(message.sender)
    
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
    
    # Update in database
    db.update_subscriber_poll(agent_name)
    
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
    
    # Messages are already dicts from database, no need to call model_dump()
    return {
        "messages": messages,
        "count": len(messages),
        "total_in_system": len(db.get_recent_events(limit=1))
    }


@app.get("/threads")
async def get_threads():
    """Get all active conversation threads"""
    threads_list = db.list_threads(limit=100)
    
    # Enhance with statistics
    result = []
    for thread in threads_list:
        stats = db.get_thread_statistics(thread['thread_id'])
        result.append({
            "thread_id": thread['thread_id'],
            "message_count": stats['message_count'],
            "first_message": thread['created_at'],
            "last_message": thread['updated_at']
        })
    
    return {"threads": result}


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
    - thread_id: Optional thread ID
    """
    eval_suite = payload.get("eval_suite")
    target_agent_url = payload.get("target_agent_url")
    target_agent_id = payload.get("target_agent_id")
    thread_id = payload.get("thread_id")
    
    if not eval_suite or not target_agent_url or not target_agent_id:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: eval_suite, target_agent_url, target_agent_id"
        )
    
    # Save to database
    db.save_eval_suite(
        suite_id=eval_suite.get("id"),
        spec_name=eval_suite.get("spec_name"),
        spec_version=eval_suite.get("spec_version"),
        test_cases=eval_suite.get("test_cases", []),
        thread_id=thread_id,
        target_agent_url=target_agent_url,
        target_agent_id=target_agent_id
    )
    
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
    evals = db.get_eval_suites(
        target_agent_url=agent_url,
        target_agent_id=agent_id
    )
    
    if agent_url and agent_id:
        return {
            "agent_key": f"{agent_url}/{agent_id}",
            "eval_suites": evals,
            "count": len(evals)
        }
    else:
        # Format for all evals
        all_evals = []
        for suite in evals:
            agent_key = f"{suite.get('target_agent_url', '')}/{suite.get('target_agent_id', '')}"
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
    eval_suite = db.get_eval_suite(eval_suite_id)
    
    if not eval_suite:
        raise HTTPException(
            status_code=404,
            detail=f"Eval suite '{eval_suite_id}' not found"
        )
    
    return {
        "eval_suite": eval_suite
    }


@app.post("/test_results")
async def store_test_results(payload: dict):
    """
    Store test results for an eval suite.
    
    Payload should include:
    - suite_id: The eval suite ID
    - thread_id: The thread ID
    - results: List of test results
    """
    suite_id = payload.get("suite_id")
    thread_id = payload.get("thread_id")
    results = payload.get("results", [])
    
    if not suite_id or not thread_id or not results:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: suite_id, thread_id, results"
        )
    
    # Save each result
    import uuid
    for result in results:
        result_id = str(uuid.uuid4())
        db.save_test_result(
            result_id=result_id,
            suite_id=suite_id,
            thread_id=thread_id,
            test_case_id=result.get("test_case_id"),
            input_sent=result.get("input_sent"),
            actual_output=result.get("actual_output"),
            expected_behavior=result.get("expected_behavior"),
            passed=result.get("passed"),
            score=result.get("score"),
            judge_reasoning=result.get("judge_reasoning")
        )
    
    event_bus_logger.info(
        f"📊 TEST_RESULTS_STORED | suite_id={suite_id} | count={len(results)}"
    )
    
    return {
        "status": "stored",
        "suite_id": suite_id,
        "results_count": len(results)
    }


@app.get("/test_results/{suite_id}")
async def get_test_results(suite_id: str):
    """Get test results for a specific eval suite"""
    results = db.get_test_results(suite_id=suite_id)
    
    return {
        "suite_id": suite_id,
        "results": results,
        "count": len(results)
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

