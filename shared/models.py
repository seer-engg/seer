"""
Peewee ORM Models for Seer
Defines database schema using Peewee for PostgreSQL
"""

import os
from datetime import datetime
from peewee import (
    Model, PostgresqlDatabase, CharField, TextField, 
    DateTimeField, IntegerField, BooleanField, FloatField,
    ForeignKeyField
)
from playhouse.postgres_ext import JSONField
from dotenv import load_dotenv
from seer.shared.logger import get_logger

load_dotenv()

# Get logger for database operations
logger = get_logger('database')

# Database connection
DATABASE_URL = os.getenv("POSTGRESQL_CONNECTION_STRING")

if not DATABASE_URL:
    raise ValueError("POSTGRESQL_CONNECTION_STRING not found in environment variables. Please add it to .env file.")

# Parse connection string
# Format: postgresql://user:password@host:port/database
db = PostgresqlDatabase(None, autoconnect=False)

# Parse and connect
try:
    from urllib.parse import urlparse
    parsed = urlparse(DATABASE_URL)
    
    db.init(
        parsed.path[1:],  # Database name (remove leading /)
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port or 5432
    )
except Exception as e:
    logger.error(f"Error parsing DATABASE_URL: {e}")
    logger.error(f"Expected format: postgresql://username:password@host:port/database")
    raise


class BaseModel(Model):
    """Base model with common configuration"""
    
    class Meta:
        database = db
        legacy_table_names = False


class Thread(BaseModel):
    """Conversation threads"""
    thread_id = CharField(primary_key=True, max_length=255)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)
    user_id = CharField(null=True, max_length=255)
    status = CharField(default='active', max_length=50)
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'threads'


class Message(BaseModel):
    """Messages in conversation threads"""
    message_id = CharField(unique=True, max_length=255)
    thread = ForeignKeyField(Thread, backref='messages', column_name='thread_id', field='thread_id')
    timestamp = DateTimeField(default=datetime.now, index=True)
    role = CharField(max_length=50)
    sender = CharField(max_length=255)
    content = TextField()
    message_type = CharField(null=True, max_length=50)
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'messages'
        indexes = (
            (('thread', 'timestamp'), False),
        )


class Event(BaseModel):
    """Event bus events"""
    event_id = CharField(unique=True, max_length=255)
    thread = ForeignKeyField(Thread, backref='events', null=True, column_name='thread_id', field='thread_id')
    timestamp = DateTimeField(default=datetime.now, index=True)
    event_type = CharField(max_length=100, index=True)
    sender = CharField(max_length=255)
    payload = TextField()
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'events'
        indexes = (
            (('thread', 'event_type'), False),
        )


class AgentActivity(BaseModel):
    """Agent activities for debugging and tracing"""
    thread = ForeignKeyField(Thread, backref='activities', column_name='thread_id', field='thread_id')
    agent_name = CharField(max_length=255, index=True)
    timestamp = DateTimeField(default=datetime.now)
    activity_type = CharField(max_length=100)
    description = TextField(null=True)
    tool_name = CharField(null=True, max_length=255)
    tool_input = TextField(null=True)
    tool_output = TextField(null=True)
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'agent_activities'
        indexes = (
            (('thread', 'agent_name'), False),
        )


class EvalSuite(BaseModel):
    """Evaluation test suites"""
    suite_id = CharField(unique=True, max_length=255)
    thread = ForeignKeyField(Thread, backref='eval_suites', null=True, column_name='thread_id', field='thread_id')
    spec_name = CharField(max_length=255)
    spec_version = CharField(max_length=50)
    target_agent_url = CharField(null=True, max_length=500)
    target_agent_id = CharField(null=True, max_length=255)
    langgraph_thread_id = CharField(null=True, max_length=255)
    test_cases = JSONField()  # Store as JSON
    created_at = DateTimeField(default=datetime.now)
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'eval_suites'
        indexes = (
            (('target_agent_url', 'target_agent_id'), False),
        )


class TestResult(BaseModel):
    """Test execution results"""
    result_id = CharField(unique=True, max_length=255)
    suite = ForeignKeyField(EvalSuite, backref='results', column_name='suite_id', field='suite_id')
    thread = ForeignKeyField(Thread, backref='test_results', column_name='thread_id', field='thread_id')
    test_case_id = CharField(max_length=255)
    input_sent = TextField()
    actual_output = TextField()
    expected_behavior = TextField()
    passed = BooleanField()
    score = FloatField()
    judge_reasoning = TextField()
    created_at = DateTimeField(default=datetime.now)
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'test_results'
        indexes = (
            (('suite', 'thread'), False),
        )


class Subscriber(BaseModel):
    """Event bus subscribers (agents)"""
    agent_name = CharField(primary_key=True, max_length=255)
    registered_at = DateTimeField(default=datetime.now)
    last_poll = DateTimeField(null=True)
    message_count = IntegerField(default=0)
    publish_count = IntegerField(default=0)
    filters = JSONField(null=True)
    status = CharField(default='active', max_length=50)
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'subscribers'


class TargetAgentExpectation(BaseModel):
    """Target agent expectations collected by orchestrator (one-to-one with Thread)"""
    thread = ForeignKeyField(Thread, backref='target_expectation', unique=True, column_name='thread_id', field='thread_id')
    expectations = JSONField()  # List of strings
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'target_agent_expectations'


class TargetAgentConfig(BaseModel):
    """Target agent configuration collected by orchestrator (one-to-one with Thread)"""
    thread = ForeignKeyField(Thread, backref='target_config', unique=True, column_name='thread_id', field='thread_id')
    target_agent_port = IntegerField(null=True)
    target_agent_url = CharField(null=True, max_length=500)
    target_agent_github_url = CharField(null=True, max_length=500)
    target_agent_assistant_id = CharField(null=True, max_length=255)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)
    metadata = JSONField(null=True)
    
    class Meta:
        table_name = 'target_agent_configs'


class RemoteThreadLink(BaseModel):
    """Persistent mapping from a local user thread to a remote agent thread per agent pair"""
    user_thread = ForeignKeyField(Thread, backref='remote_threads', column_name='thread_id', field='thread_id')
    src_agent = CharField(max_length=255)
    dst_agent = CharField(max_length=255)
    remote_base_url = CharField(max_length=500)
    remote_thread_id = CharField(max_length=255)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'remote_thread_links'
        indexes = (
            (("user_thread", "src_agent", "dst_agent"), True),
        )

# List of all models for easy iteration
ALL_MODELS = [
    Thread,
    Message,
    Event,
    AgentActivity,
    EvalSuite,
    TestResult,
    Subscriber,
    TargetAgentExpectation,
    TargetAgentConfig,
    RemoteThreadLink
]


def init_db():
    """Initialize database connection and create tables"""
    db.connect(reuse_if_open=True)
    db.create_tables(ALL_MODELS, safe=True)
    logger.info("✅ Database initialized with Peewee ORM")


def close_db():
    """Close database connection"""
    if not db.is_closed():
        db.close()

