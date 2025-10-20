"""
Database module for Seer - PostgreSQL with Peewee ORM
Replaces the old SQLite implementation
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from .models import (
    db, init_db, close_db,
    Thread, Message, Event, AgentActivity,
    EvalSuite, TestResult, Subscriber
)


class Database:
    """PostgreSQL database manager using Peewee ORM"""
    
    def __init__(self):
        """Initialize database connection and create tables"""
        init_db()
    
    @contextmanager
    def atomic(self):
        """Context manager for atomic transactions"""
        with db.atomic():
            yield
    
    # Thread operations
    def create_thread(self, thread_id: str, user_id: str = None, metadata: Dict[str, Any] = None):
        """Create a new thread"""
        Thread.get_or_create(
            thread_id=thread_id,
            defaults={
                'user_id': user_id,
                'metadata': metadata
            }
        )
    
    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get thread by ID"""
        try:
            thread = Thread.get(Thread.thread_id == thread_id)
            return {
                'thread_id': thread.thread_id,
                'created_at': thread.created_at.isoformat() if thread.created_at else None,
                'updated_at': thread.updated_at.isoformat() if thread.updated_at else None,
                'user_id': thread.user_id,
                'status': thread.status,
                'metadata': thread.metadata
            }
        except Thread.DoesNotExist:
            return None
    
    def list_threads(self, limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
        """List all threads"""
        query = Thread.select().order_by(Thread.updated_at.desc()).limit(limit)
        
        if status:
            query = query.where(Thread.status == status)
        
        return [
            {
                'thread_id': t.thread_id,
                'created_at': t.created_at.isoformat() if t.created_at else None,
                'updated_at': t.updated_at.isoformat() if t.updated_at else None,
                'user_id': t.user_id,
                'status': t.status,
                'metadata': t.metadata
            }
            for t in query
        ]
    
    def update_thread(self, thread_id: str, **kwargs):
        """Update thread attributes"""
        kwargs['updated_at'] = datetime.now()
        Thread.update(**kwargs).where(Thread.thread_id == thread_id).execute()
    
    # Message operations
    def add_message(self, thread_id: str, message_id: str, timestamp: str, 
                   role: str, sender: str, content: str, 
                   message_type: str = None, metadata: Dict[str, Any] = None):
        """Add a message to a thread"""
        # Ensure thread exists
        Thread.get_or_create(thread_id=thread_id)
        
        # Parse timestamp if string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        
        Message.create(
            message_id=message_id,
            thread=thread_id,
            timestamp=timestamp,
            role=role,
            sender=sender,
            content=content,
            message_type=message_type,
            metadata=metadata
        )
        
        # Update thread's updated_at
        Thread.update(updated_at=datetime.now()).where(Thread.thread_id == thread_id).execute()
    
    def get_thread_messages(self, thread_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get all messages in a thread"""
        query = Message.select().where(Message.thread == thread_id).order_by(Message.timestamp.asc())
        
        if limit:
            query = query.limit(limit)
        
        return [
            {
                'message_id': m.message_id,
                'thread_id': m.thread.thread_id,
                'timestamp': m.timestamp.isoformat() if m.timestamp else None,
                'role': m.role,
                'sender': m.sender,
                'content': m.content,
                'message_type': m.message_type,
                'metadata': m.metadata
            }
            for m in query
        ]
    
    # Event operations
    def publish_event(self, event_id: str, event_type: str, sender: str, 
                     payload: Dict[str, Any], thread_id: str = None, 
                     metadata: Dict[str, Any] = None):
        """Publish an event"""
        Event.create(
            event_id=event_id,
            thread=thread_id,
            timestamp=datetime.now(),
            event_type=event_type,
            sender=sender,
            payload=json.dumps(payload),
            metadata=metadata
        )
    
    def get_events(self, thread_id: str = None, event_type: str = None, 
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get events with optional filters"""
        query = Event.select().order_by(Event.timestamp.desc()).limit(limit)
        
        if thread_id:
            query = query.where(Event.thread == thread_id)
        if event_type:
            query = query.where(Event.event_type == event_type)
        
        return [
            {
                'event_id': e.event_id,
                'thread_id': e.thread.thread_id if e.thread else None,
                'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                'event_type': e.event_type,
                'sender': e.sender,
                'payload': json.loads(e.payload) if e.payload else {},
                'metadata': e.metadata
            }
            for e in query
        ]
    
    # Agent activity operations
    def log_agent_activity(self, thread_id: str, agent_name: str, activity_type: str,
                          description: str = None, tool_name: str = None,
                          tool_input: str = None, tool_output: str = None,
                          metadata: Dict[str, Any] = None):
        """Log an agent activity"""
        # Ensure thread exists
        Thread.get_or_create(thread_id=thread_id)
        
        AgentActivity.create(
            thread=thread_id,
            agent_name=agent_name,
            timestamp=datetime.now(),
            activity_type=activity_type,
            description=description,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            metadata=metadata
        )
    
    def get_agent_activities(self, thread_id: str = None, agent_name: str = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get agent activities with optional filters"""
        query = AgentActivity.select().order_by(AgentActivity.timestamp.desc()).limit(limit)
        
        if thread_id:
            query = query.where(AgentActivity.thread == thread_id)
        if agent_name:
            query = query.where(AgentActivity.agent_name == agent_name)
        
        return [
            {
                'thread_id': a.thread.thread_id,
                'agent_name': a.agent_name,
                'timestamp': a.timestamp.isoformat() if a.timestamp else None,
                'activity_type': a.activity_type,
                'description': a.description,
                'tool_name': a.tool_name,
                'tool_input': a.tool_input,
                'tool_output': a.tool_output,
                'metadata': a.metadata
            }
            for a in query
        ]
    
    # Eval suite operations
    def save_eval_suite(self, suite_id: str, spec_name: str, spec_version: str,
                       test_cases: List[Dict[str, Any]], thread_id: str = None,
                       target_agent_url: str = None, target_agent_id: str = None,
                       langgraph_thread_id: str = None, metadata: Dict[str, Any] = None):
        """Save an evaluation suite"""
        # Ensure thread exists if provided
        if thread_id:
            Thread.get_or_create(thread_id=thread_id)
        
        EvalSuite.create(
            suite_id=suite_id,
            thread=thread_id,
            spec_name=spec_name,
            spec_version=spec_version,
            target_agent_url=target_agent_url,
            target_agent_id=target_agent_id,
            langgraph_thread_id=langgraph_thread_id,
            test_cases=test_cases,  # JSONField handles serialization
            metadata=metadata
        )
    
    def get_eval_suite(self, suite_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific eval suite"""
        try:
            suite = EvalSuite.get(EvalSuite.suite_id == suite_id)
            return {
                'suite_id': suite.suite_id,
                'thread_id': suite.thread.thread_id if suite.thread else None,
                'spec_name': suite.spec_name,
                'spec_version': suite.spec_version,
                'target_agent_url': suite.target_agent_url,
                'target_agent_id': suite.target_agent_id,
                'langgraph_thread_id': suite.langgraph_thread_id,
                'test_cases': suite.test_cases,
                'created_at': suite.created_at.isoformat() if suite.created_at else None,
                'metadata': suite.metadata
            }
        except EvalSuite.DoesNotExist:
            return None
    
    def get_eval_suites(self, target_agent_url: str = None, target_agent_id: str = None,
                       thread_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get eval suites with optional filters"""
        query = EvalSuite.select().order_by(EvalSuite.created_at.desc()).limit(limit)
        
        if target_agent_url:
            query = query.where(EvalSuite.target_agent_url == target_agent_url)
        if target_agent_id:
            query = query.where(EvalSuite.target_agent_id == target_agent_id)
        if thread_id:
            query = query.where(EvalSuite.thread == thread_id)
        
        return [
            {
                'suite_id': s.suite_id,
                'thread_id': s.thread.thread_id if s.thread else None,
                'spec_name': s.spec_name,
                'spec_version': s.spec_version,
                'target_agent_url': s.target_agent_url,
                'target_agent_id': s.target_agent_id,
                'langgraph_thread_id': s.langgraph_thread_id,
                'test_cases': s.test_cases,
                'created_at': s.created_at.isoformat() if s.created_at else None,
                'metadata': s.metadata
            }
            for s in query
        ]
    
    # Test result operations
    def save_test_result(self, result_id: str, suite_id: str, thread_id: str,
                        test_case_id: str, input_sent: str, actual_output: str,
                        expected_behavior: str, passed: bool, score: float,
                        judge_reasoning: str, metadata: Dict[str, Any] = None):
        """Save a test result"""
        # Ensure thread exists
        Thread.get_or_create(thread_id=thread_id)
        
        TestResult.create(
            result_id=result_id,
            suite=suite_id,
            thread=thread_id,
            test_case_id=test_case_id,
            input_sent=input_sent,
            actual_output=actual_output,
            expected_behavior=expected_behavior,
            passed=passed,
            score=score,
            judge_reasoning=judge_reasoning,
            metadata=metadata
        )
    
    def get_test_results(self, suite_id: str = None, thread_id: str = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """Get test results with optional filters"""
        query = TestResult.select().order_by(TestResult.created_at.desc()).limit(limit)
        
        if suite_id:
            query = query.where(TestResult.suite == suite_id)
        if thread_id:
            query = query.where(TestResult.thread == thread_id)
        
        return [
            {
                'result_id': r.result_id,
                'suite_id': r.suite.suite_id,
                'thread_id': r.thread.thread_id,
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
            for r in query
        ]
    
    # Subscriber (agent registry) operations
    def register_subscriber(self, agent_name: str, filters: Dict[str, Any] = None):
        """Register a new subscriber (agent)"""
        Subscriber.get_or_create(
            agent_name=agent_name,
            defaults={
                'filters': filters,
                'status': 'active'
            }
        )
    
    def update_subscriber(self, agent_name: str, **kwargs):
        """Update subscriber attributes"""
        Subscriber.update(**kwargs).where(Subscriber.agent_name == agent_name).execute()
    
    def get_subscriber(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get subscriber by name"""
        try:
            sub = Subscriber.get(Subscriber.agent_name == agent_name)
            return {
                'agent_name': sub.agent_name,
                'registered_at': sub.registered_at.isoformat() if sub.registered_at else None,
                'last_poll': sub.last_poll.isoformat() if sub.last_poll else None,
                'message_count': sub.message_count,
                'publish_count': sub.publish_count,
                'filters': sub.filters,
                'status': sub.status,
                'metadata': sub.metadata
            }
        except Subscriber.DoesNotExist:
            return None
    
    def get_subscribers(self, status: str = None) -> List[Dict[str, Any]]:
        """Get all subscribers"""
        query = Subscriber.select()
        
        if status:
            query = query.where(Subscriber.status == status)
        
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
            for s in query
        ]
    
    def delete_subscriber(self, agent_name: str):
        """Delete a subscriber"""
        Subscriber.delete().where(Subscriber.agent_name == agent_name).execute()
    
    # Cleanup and utility methods
    def close(self):
        """Close database connection"""
        close_db()
    
    def __del__(self):
        """Ensure connection is closed on deletion"""
        try:
            self.close()
        except:
            pass


# Global database instance
_db_instance: Optional[Database] = None


def get_db() -> Database:
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance

