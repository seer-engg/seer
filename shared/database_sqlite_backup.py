"""
Database module for Seer - SQLite persistence layer

Stores:
- Chat threads and messages
- Event bus events  
- Agent activities
- Eval suites and test results
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import threading


# Thread-local storage for database connections
_thread_local = threading.local()


class Database:
    """SQLite database manager for Seer"""
    
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
        self._init_db()
    
    @contextmanager
    def get_connection(self):
        """Get thread-safe database connection"""
        # Each thread gets its own connection
        if not hasattr(_thread_local, 'conn') or _thread_local.conn is None:
            _thread_local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            _thread_local.conn.row_factory = sqlite3.Row
        
        try:
            yield _thread_local.conn
        except Exception as e:
            _thread_local.conn.rollback()
            raise e
    
    def _init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Threads table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)
            
            # Messages table - stores all messages in threads
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message_id TEXT UNIQUE,
                    timestamp TIMESTAMP NOT NULL,
                    role TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_type TEXT,
                    metadata TEXT,
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                )
            """)
            
            # Events table - stores all event bus events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    thread_id TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                )
            """)
            
            # Agent activities table - tracks what each agent did
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    activity_type TEXT NOT NULL,
                    description TEXT,
                    tool_name TEXT,
                    tool_input TEXT,
                    tool_output TEXT,
                    metadata TEXT,
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                )
            """)
            
            # Eval suites table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eval_suites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_id TEXT UNIQUE NOT NULL,
                    thread_id TEXT,
                    spec_name TEXT NOT NULL,
                    spec_version TEXT NOT NULL,
                    target_agent_url TEXT,
                    target_agent_id TEXT,
                    langgraph_thread_id TEXT,
                    test_cases TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                )
            """)
            
            # Add langgraph_thread_id column if it doesn't exist (for existing databases)
            try:
                cursor.execute("ALTER TABLE eval_suites ADD COLUMN langgraph_thread_id TEXT")
            except Exception:
                # Column already exists, ignore
                pass
            
            # Test results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id TEXT UNIQUE NOT NULL,
                    suite_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    test_case_id TEXT NOT NULL,
                    input_sent TEXT NOT NULL,
                    actual_output TEXT NOT NULL,
                    expected_behavior TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    score REAL NOT NULL,
                    judge_reasoning TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (suite_id) REFERENCES eval_suites(suite_id),
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                )
            """)
            
            # Subscribers table - track active subscribers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subscribers (
                    agent_name TEXT PRIMARY KEY,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_poll TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    publish_count INTEGER DEFAULT 0,
                    filters TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_thread_id 
                ON messages(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                ON messages(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_thread_id 
                ON events(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_event_type 
                ON events(event_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_activities_thread_id 
                ON agent_activities(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_activities_agent_name 
                ON agent_activities(agent_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_results_suite_id 
                ON test_results(suite_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_results_thread_id 
                ON test_results(thread_id)
            """)
            
            conn.commit()
    
    # Thread operations
    def create_thread(self, thread_id: str, user_id: str = None, metadata: Dict[str, Any] = None):
        """Create a new thread"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO threads (thread_id, user_id, metadata)
                VALUES (?, ?, ?)
            """, (thread_id, user_id, json.dumps(metadata) if metadata else None))
            conn.commit()
    
    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get thread by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_threads(self, limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
        """List all threads"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("""
                    SELECT * FROM threads 
                    WHERE status = ?
                    ORDER BY updated_at DESC 
                    LIMIT ?
                """, (status, limit))
            else:
                cursor.execute("""
                    SELECT * FROM threads 
                    ORDER BY updated_at DESC 
                    LIMIT ?
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def update_thread_timestamp(self, thread_id: str):
        """Update thread's updated_at timestamp"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE threads 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE thread_id = ?
            """, (thread_id,))
            conn.commit()
    
    # Message operations
    def add_message(self, thread_id: str, message_id: str, timestamp: str, 
                   role: str, sender: str, content: str, 
                   message_type: str = None, metadata: Dict[str, Any] = None):
        """Add a message to a thread"""
        # Ensure thread exists
        self.create_thread(thread_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO messages 
                (thread_id, message_id, timestamp, role, sender, content, message_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (thread_id, message_id, timestamp, role, sender, content, 
                  message_type, json.dumps(metadata) if metadata else None))
            conn.commit()
        
        # Update thread timestamp
        self.update_thread_timestamp(thread_id)
    
    def get_thread_messages(self, thread_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all messages in a thread"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM messages 
                WHERE thread_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (thread_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    # Event operations
    def add_event(self, event_id: str, thread_id: str, timestamp: str,
                 event_type: str, sender: str, payload: Dict[str, Any],
                 metadata: Dict[str, Any] = None):
        """Add an event to the database"""
        # Ensure thread exists if thread_id is provided
        if thread_id:
            self.create_thread(thread_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO events
                (event_id, thread_id, timestamp, event_type, sender, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (event_id, thread_id, timestamp, event_type, sender,
                  json.dumps(payload), json.dumps(metadata) if metadata else None))
            conn.commit()
        
        # Update thread timestamp if applicable
        if thread_id:
            self.update_thread_timestamp(thread_id)
    
    def get_thread_events(self, thread_id: str, event_type: str = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get all events for a thread"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if event_type:
                cursor.execute("""
                    SELECT * FROM events 
                    WHERE thread_id = ? AND event_type = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (thread_id, event_type, limit))
            else:
                cursor.execute("""
                    SELECT * FROM events 
                    WHERE thread_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (thread_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_events(self, limit: int = 100, event_type: str = None,
                         since: str = None) -> List[Dict[str, Any]]:
        """Get recent events across all threads"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            if since:
                query += " AND timestamp > ?"
                params.append(since)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # Agent activity operations
    def add_agent_activity(self, thread_id: str, agent_name: str, 
                          activity_type: str, description: str = None,
                          tool_name: str = None, tool_input: str = None,
                          tool_output: str = None, metadata: Dict[str, Any] = None):
        """Record an agent activity"""
        # Ensure thread exists
        self.create_thread(thread_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_activities
                (thread_id, agent_name, timestamp, activity_type, description,
                 tool_name, tool_input, tool_output, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (thread_id, agent_name, datetime.now().isoformat(), 
                  activity_type, description, tool_name, tool_input, tool_output,
                  json.dumps(metadata) if metadata else None))
            conn.commit()
    
    def get_agent_activities(self, thread_id: str, agent_name: str = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get agent activities for a thread"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if agent_name:
                cursor.execute("""
                    SELECT * FROM agent_activities
                    WHERE thread_id = ? AND agent_name = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (thread_id, agent_name, limit))
            else:
                cursor.execute("""
                    SELECT * FROM agent_activities
                    WHERE thread_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (thread_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    # Eval suite operations
    def save_eval_suite(self, suite_id: str, spec_name: str, spec_version: str,
                       test_cases: List[Dict[str, Any]], thread_id: str = None,
                       target_agent_url: str = None, target_agent_id: str = None,
                       langgraph_thread_id: str = None, metadata: Dict[str, Any] = None):
        """Save an eval suite"""
        if thread_id:
            self.create_thread(thread_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO eval_suites
                (suite_id, thread_id, spec_name, spec_version, target_agent_url,
                 target_agent_id, langgraph_thread_id, test_cases, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (suite_id, thread_id, spec_name, spec_version, target_agent_url,
                  target_agent_id, langgraph_thread_id, json.dumps(test_cases), 
                  json.dumps(metadata) if metadata else None))
            conn.commit()
    
    def get_eval_suite(self, suite_id: str) -> Optional[Dict[str, Any]]:
        """Get eval suite by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM eval_suites WHERE suite_id = ?", (suite_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['test_cases'] = json.loads(result['test_cases'])
                return result
            return None
    
    def get_eval_suites(self, target_agent_url: str = None, target_agent_id: str = None,
                       thread_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get eval suites with optional filters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM eval_suites WHERE 1=1"
            params = []
            
            if target_agent_url:
                query += " AND target_agent_url = ?"
                params.append(target_agent_url)
            
            if target_agent_id:
                query += " AND target_agent_id = ?"
                params.append(target_agent_id)
            
            if thread_id:
                query += " AND thread_id = ?"
                params.append(thread_id)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['test_cases'] = json.loads(result['test_cases'])
                results.append(result)
            return results
    
    # Test results operations
    def save_test_result(self, result_id: str, suite_id: str, thread_id: str,
                        test_case_id: str, input_sent: str, actual_output: str,
                        expected_behavior: str, passed: bool, score: float,
                        judge_reasoning: str, metadata: Dict[str, Any] = None):
        """Save a test result"""
        self.create_thread(thread_id)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO test_results
                (result_id, suite_id, thread_id, test_case_id, input_sent,
                 actual_output, expected_behavior, passed, score, judge_reasoning, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (result_id, suite_id, thread_id, test_case_id, input_sent,
                  actual_output, expected_behavior, passed, score, judge_reasoning,
                  json.dumps(metadata) if metadata else None))
            conn.commit()
    
    def get_test_results(self, suite_id: str = None, thread_id: str = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """Get test results with optional filters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM test_results WHERE 1=1"
            params = []
            
            if suite_id:
                query += " AND suite_id = ?"
                params.append(suite_id)
            
            if thread_id:
                query += " AND thread_id = ?"
                params.append(thread_id)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # Subscriber operations
    def register_subscriber(self, agent_name: str, filters: Dict[str, Any] = None):
        """Register a subscriber"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO subscribers
                (agent_name, filters, registered_at, last_poll)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (agent_name, json.dumps(filters) if filters else None))
            conn.commit()
    
    def update_subscriber_poll(self, agent_name: str):
        """Update subscriber's last poll time"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE subscribers 
                SET last_poll = CURRENT_TIMESTAMP,
                    message_count = message_count + 1
                WHERE agent_name = ?
            """, (agent_name,))
            conn.commit()
    
    def update_subscriber_publish(self, agent_name: str):
        """Increment subscriber's publish count"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE subscribers 
                SET publish_count = publish_count + 1
                WHERE agent_name = ?
            """, (agent_name,))
            conn.commit()
    
    def get_subscribers(self) -> List[Dict[str, Any]]:
        """Get all subscribers"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM subscribers WHERE status = 'active'")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_thread_statistics(self, thread_id: str) -> Dict[str, Any]:
        """Get statistics for a thread"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Message count
            cursor.execute("""
                SELECT COUNT(*) as count FROM messages WHERE thread_id = ?
            """, (thread_id,))
            message_count = cursor.fetchone()['count']
            
            # Event count
            cursor.execute("""
                SELECT COUNT(*) as count FROM events WHERE thread_id = ?
            """, (thread_id,))
            event_count = cursor.fetchone()['count']
            
            # Agent activity count
            cursor.execute("""
                SELECT COUNT(*) as count FROM agent_activities WHERE thread_id = ?
            """, (thread_id,))
            activity_count = cursor.fetchone()['count']
            
            # Active agents
            cursor.execute("""
                SELECT DISTINCT agent_name FROM agent_activities WHERE thread_id = ?
            """, (thread_id,))
            active_agents = [row['agent_name'] for row in cursor.fetchall()]
            
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


def init_db(db_path: str = None) -> Database:
    """Initialize global database instance"""
    global _db_instance
    _db_instance = Database(db_path)
    return _db_instance

