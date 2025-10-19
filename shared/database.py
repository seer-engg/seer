"""
Database module for Seer - Tortoise ORM persistence layer

Stores:
- Chat threads and messages
- Event bus events  
- Agent activities
- Eval suites and test results
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from tortoise import Tortoise
from tortoise.transactions import in_transaction

from .models import (
    Thread, Message, Event, AgentActivity,
    EvalSuite, TestResult, Subscriber
)


class Database:
    """Database manager for Seer using Tortoise ORM"""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Default to project root / data / seer.db
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "seer.db")
        
        self.db_path = db_path
        self.db_url = f"sqlite://{db_path}"
        self._initialized = False
    
    async def init(self):
        """Initialize Tortoise ORM connection"""
        if self._initialized:
            return
        
        await Tortoise.init(
            db_url=self.db_url,
            modules={'models': ['shared.models']},
        )
        await Tortoise.generate_schemas(safe=True)
        self._initialized = True
    
    async def close(self):
        """Close database connections"""
        await Tortoise.close_connections()
        self._initialized = False
    
    # Thread operations
    async def create_thread(self, thread_id: str, user_id: str = None, 
                           metadata: Dict[str, Any] = None):
        """Create a new thread"""
        await Thread.get_or_create(
            thread_id=thread_id,
            defaults={
                'user_id': user_id,
                'metadata': metadata
            }
        )
    
    async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get thread by ID"""
        thread = await Thread.filter(thread_id=thread_id).first()
        if thread:
            return {
                'thread_id': thread.thread_id,
                'created_at': thread.created_at.isoformat() if thread.created_at else None,
                'updated_at': thread.updated_at.isoformat() if thread.updated_at else None,
                'user_id': thread.user_id,
                'status': thread.status,
                'metadata': thread.metadata,
                'github_url': thread.github_url,
                'agent_host': thread.agent_host,
                'agent_port': thread.agent_port,
                'agent_id': thread.agent_id
            }
        return None
    
    async def list_threads(self, limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
        """List all threads"""
        query = Thread.all()
        if status:
            query = query.filter(status=status)
        
        threads = await query.order_by('-updated_at').limit(limit)
        
        return [
            {
                'thread_id': t.thread_id,
                'created_at': t.created_at.isoformat() if t.created_at else None,
                'updated_at': t.updated_at.isoformat() if t.updated_at else None,
                'user_id': t.user_id,
                'status': t.status,
                'metadata': t.metadata
            }
            for t in threads
        ]
    
    async def update_thread_timestamp(self, thread_id: str):
        """Update thread's updated_at timestamp"""
        await Thread.filter(thread_id=thread_id).update(updated_at=datetime.now())
    
    async def update_thread_config(self, thread_id: str, github_url: str = None,
                                   agent_host: str = None, agent_port: int = None,
                                   agent_id: str = None):
        """Update thread configuration (GitHub URL and agent details)"""
        # Ensure thread exists
        await self.create_thread(thread_id)
        
        # Build update dict with only non-None values
        update_data = {}
        if github_url is not None:
            update_data['github_url'] = github_url
        if agent_host is not None:
            update_data['agent_host'] = agent_host
        if agent_port is not None:
            update_data['agent_port'] = agent_port
        if agent_id is not None:
            update_data['agent_id'] = agent_id
        
        if update_data:
            await Thread.filter(thread_id=thread_id).update(**update_data)
    
    async def get_thread_config(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get thread configuration (GitHub URL and agent details)"""
        thread = await Thread.filter(thread_id=thread_id).first()
        if thread:
            return {
                'github_url': thread.github_url,
                'agent_host': thread.agent_host,
                'agent_port': thread.agent_port,
                'agent_id': thread.agent_id
            }
        return None
    
    # Message operations
    async def add_message(self, thread_id: str, message_id: str, timestamp: str, 
                         role: str, sender: str, content: str, 
                         message_type: str = None, metadata: Dict[str, Any] = None):
        """Add a message to a thread"""
        # Ensure thread exists
        await self.create_thread(thread_id)
        
        # Parse timestamp
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Create or get message
        await Message.get_or_create(
            message_id=message_id,
            defaults={
                'thread_id': thread_id,
                'timestamp': timestamp,
                'role': role,
                'sender': sender,
                'content': content,
                'message_type': message_type,
                'metadata': metadata
            }
        )
        
        # Update thread timestamp
        await self.update_thread_timestamp(thread_id)
    
    async def get_thread_messages(self, thread_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all messages in a thread"""
        messages = await Message.filter(thread_id=thread_id).order_by('timestamp').limit(limit)
        
        return [
            {
                'thread_id': m.thread_id,
                'message_id': m.message_id,
                'timestamp': m.timestamp.isoformat() if m.timestamp else None,
                'role': m.role,
                'sender': m.sender,
                'content': m.content,
                'message_type': m.message_type,
                'metadata': m.metadata
            }
            for m in messages
        ]
    
    # Event operations
    async def add_event(self, event_id: str, thread_id: str, timestamp: str,
                       event_type: str, sender: str, payload: Dict[str, Any],
                       metadata: Dict[str, Any] = None):
        """Add an event to the database"""
        # Ensure thread exists if thread_id is provided
        if thread_id:
            await self.create_thread(thread_id)
        
        # Parse timestamp
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        await Event.get_or_create(
            event_id=event_id,
            defaults={
                'thread_id': thread_id,
                'timestamp': timestamp,
                'event_type': event_type,
                'sender': sender,
                'payload': payload,
                'metadata': metadata
            }
        )
        
        # Update thread timestamp if applicable
        if thread_id:
            await self.update_thread_timestamp(thread_id)
    
    async def get_thread_events(self, thread_id: str, event_type: str = None, 
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get all events for a thread"""
        query = Event.filter(thread_id=thread_id)
        if event_type:
            query = query.filter(event_type=event_type)
        
        events = await query.order_by('timestamp').limit(limit)
        
        return [
            {
                'event_id': e.event_id,
                'thread_id': e.thread_id,
                'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                'event_type': e.event_type,
                'sender': e.sender,
                'payload': e.payload,
                'metadata': e.metadata
            }
            for e in events
        ]
    
    async def get_recent_events(self, limit: int = 100, event_type: str = None,
                               since: str = None) -> List[Dict[str, Any]]:
        """Get recent events across all threads"""
        query = Event.all()
        
        if event_type:
            query = query.filter(event_type=event_type)
        
        if since:
            if isinstance(since, str):
                since = datetime.fromisoformat(since.replace('Z', '+00:00'))
            query = query.filter(timestamp__gt=since)
        
        events = await query.order_by('-timestamp').limit(limit)
        
        return [
            {
                'event_id': e.event_id,
                'thread_id': e.thread_id,
                'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                'event_type': e.event_type,
                'sender': e.sender,
                'payload': e.payload,
                'metadata': e.metadata
            }
            for e in events
        ]
    
    # Agent activity operations
    async def add_agent_activity(self, thread_id: str, agent_name: str, 
                                activity_type: str, description: str = None,
                                tool_name: str = None, tool_input: str = None,
                                tool_output: str = None, metadata: Dict[str, Any] = None):
        """Record an agent activity"""
        # Ensure thread exists
        await self.create_thread(thread_id)
        
        await AgentActivity.create(
            thread_id=thread_id,
            agent_name=agent_name,
            activity_type=activity_type,
            description=description,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            metadata=metadata
        )
    
    async def get_agent_activities(self, thread_id: str, agent_name: str = None,
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get agent activities for a thread"""
        query = AgentActivity.filter(thread_id=thread_id)
        if agent_name:
            query = query.filter(agent_name=agent_name)
        
        activities = await query.order_by('timestamp').limit(limit)
        
        return [
            {
                'thread_id': a.thread_id,
                'agent_name': a.agent_name,
                'timestamp': a.timestamp.isoformat() if a.timestamp else None,
                'activity_type': a.activity_type,
                'description': a.description,
                'tool_name': a.tool_name,
                'tool_input': a.tool_input,
                'tool_output': a.tool_output,
                'metadata': a.metadata
            }
            for a in activities
        ]
    
    # Eval suite operations
    async def save_eval_suite(self, suite_id: str, spec_name: str, spec_version: str,
                             test_cases: List[Dict[str, Any]], thread_id: str = None,
                             target_agent_url: str = None, target_agent_id: str = None,
                             metadata: Dict[str, Any] = None):
        """Save an eval suite"""
        if thread_id:
            await self.create_thread(thread_id)
        
        await EvalSuite.update_or_create(
            suite_id=suite_id,
            defaults={
                'thread_id': thread_id,
                'spec_name': spec_name,
                'spec_version': spec_version,
                'target_agent_url': target_agent_url,
                'target_agent_id': target_agent_id,
                'test_cases': test_cases,
                'metadata': metadata
            }
        )
    
    async def get_eval_suite(self, suite_id: str) -> Optional[Dict[str, Any]]:
        """Get eval suite by ID"""
        suite = await EvalSuite.filter(suite_id=suite_id).first()
        if suite:
            return {
                'suite_id': suite.suite_id,
                'thread_id': suite.thread_id,
                'spec_name': suite.spec_name,
                'spec_version': suite.spec_version,
                'target_agent_url': suite.target_agent_url,
                'target_agent_id': suite.target_agent_id,
                'test_cases': suite.test_cases,
                'created_at': suite.created_at.isoformat() if suite.created_at else None,
                'metadata': suite.metadata
            }
        return None
    
    async def get_eval_suites(self, target_agent_url: str = None, target_agent_id: str = None,
                             thread_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get eval suites with optional filters"""
        query = EvalSuite.all()
        
        if target_agent_url:
            query = query.filter(target_agent_url=target_agent_url)
        
        if target_agent_id:
            query = query.filter(target_agent_id=target_agent_id)
        
        if thread_id:
            query = query.filter(thread_id=thread_id)
        
        suites = await query.order_by('-created_at').limit(limit)
        
        return [
            {
                'suite_id': s.suite_id,
                'thread_id': s.thread_id,
                'spec_name': s.spec_name,
                'spec_version': s.spec_version,
                'target_agent_url': s.target_agent_url,
                'target_agent_id': s.target_agent_id,
                'test_cases': s.test_cases,
                'created_at': s.created_at.isoformat() if s.created_at else None,
                'metadata': s.metadata
            }
            for s in suites
        ]
    
    # Test results operations
    async def save_test_result(self, result_id: str, suite_id: str, thread_id: str,
                              test_case_id: str, input_sent: str, actual_output: str,
                              expected_behavior: str, passed: bool, score: float,
                              judge_reasoning: str, metadata: Dict[str, Any] = None):
        """Save a test result"""
        await self.create_thread(thread_id)
        
        await TestResult.update_or_create(
            result_id=result_id,
            defaults={
                'suite_id': suite_id,
                'thread_id': thread_id,
                'test_case_id': test_case_id,
                'input_sent': input_sent,
                'actual_output': actual_output,
                'expected_behavior': expected_behavior,
                'passed': passed,
                'score': score,
                'judge_reasoning': judge_reasoning,
                'metadata': metadata
            }
        )
    
    async def get_test_results(self, suite_id: str = None, thread_id: str = None,
                              limit: int = 1000) -> List[Dict[str, Any]]:
        """Get test results with optional filters"""
        query = TestResult.all()
        
        if suite_id:
            query = query.filter(suite_id=suite_id)
        
        if thread_id:
            query = query.filter(thread_id=thread_id)
        
        results = await query.order_by('-created_at').limit(limit)
        
        return [
            {
                'result_id': r.result_id,
                'suite_id': r.suite_id,
                'thread_id': r.thread_id,
                'test_case_id': r.test_case_id,
                'input_sent': r.input_sent,
                'actual_output': r.actual_output,
                'expected_behavior': r.expected_behavior,
                'passed': r.passed,
                'score': r.score,
                'judge_reasoning': r.judge_reasoning,
                'created_at': r.created_at.isoformat() if r.created_at else None,
                'metadata': r.metadata
            }
            for r in results
        ]
    
    # Subscriber operations
    async def register_subscriber(self, agent_name: str, filters: Dict[str, Any] = None):
        """Register a subscriber"""
        await Subscriber.update_or_create(
            agent_name=agent_name,
            defaults={
                'filters': filters,
                'last_poll': datetime.now()
            }
        )
    
    async def update_subscriber_poll(self, agent_name: str):
        """Update subscriber's last poll time"""
        subscriber = await Subscriber.filter(agent_name=agent_name).first()
        if subscriber:
            subscriber.last_poll = datetime.now()
            subscriber.message_count += 1
            await subscriber.save()
    
    async def update_subscriber_publish(self, agent_name: str):
        """Increment subscriber's publish count"""
        subscriber = await Subscriber.filter(agent_name=agent_name).first()
        if subscriber:
            subscriber.publish_count += 1
            await subscriber.save()
    
    async def get_subscribers(self) -> List[Dict[str, Any]]:
        """Get all subscribers"""
        subscribers = await Subscriber.filter(status='active').all()
        
        return [
            {
                'agent_name': s.agent_name,
                'registered_at': s.registered_at.isoformat() if s.registered_at else None,
                'last_poll': s.last_poll.isoformat() if s.last_poll else None,
                'message_count': s.message_count,
                'publish_count': s.publish_count,
                'filters': s.filters,
                'status': s.status,
                'metadata': s.metadata
            }
            for s in subscribers
        ]
    
    async def get_thread_statistics(self, thread_id: str) -> Dict[str, Any]:
        """Get statistics for a thread"""
        # Message count
        message_count = await Message.filter(thread_id=thread_id).count()
        
        # Event count
        event_count = await Event.filter(thread_id=thread_id).count()
        
        # Agent activity count
        activity_count = await AgentActivity.filter(thread_id=thread_id).count()
        
        # Active agents
        activities = await AgentActivity.filter(thread_id=thread_id).distinct().values('agent_name')
        active_agents = [a['agent_name'] for a in activities]
        
        return {
            "thread_id": thread_id,
            "message_count": message_count,
            "event_count": event_count,
            "activity_count": activity_count,
            "active_agents": active_agents
        }


# Global database instance
_db_instance: Optional[Database] = None


def get_db() -> Database:
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


async def init_db(db_path: str = None) -> Database:
    """Initialize global database instance"""
    global _db_instance
    _db_instance = Database(db_path)
    await _db_instance.init()
    return _db_instance


async def close_db():
    """Close global database connection"""
    global _db_instance
    if _db_instance is not None:
        await _db_instance.close()
        _db_instance = None
