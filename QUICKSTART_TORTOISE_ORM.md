# Tortoise ORM Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install tortoise-orm>=0.21.0 aerich>=0.7.2
```

## Initialize Database

```bash
python init_db.py
```

This will:
- Create the SQLite database at `data/seer.db`
- Initialize all tables with Tortoise ORM
- Show table statistics

### Reset Database

To start fresh (deletes all data):
```bash
python init_db.py --reset
```

### Migrate Old Data

To migrate existing eval JSON files:
```bash
python init_db.py --migrate
```

## Using the Database

### In Async Context (Recommended)

```python
from shared.database import init_db, get_db

# Initialize (do once at startup)
await init_db()

# Get database instance
db = get_db()

# Use database methods (all are async)
threads = await db.list_threads(limit=10)
thread = await db.get_thread("thread_123")
await db.create_thread("new_thread", user_id="user_1")
```

### Using ORM Models Directly

```python
from shared.models import Thread, Message, Event

# Query threads
active_threads = await Thread.filter(status='active').all()

# Get specific thread
thread = await Thread.get(thread_id="thread_123")

# Create new thread
new_thread = await Thread.create(
    thread_id="thread_456",
    user_id="user_1",
    status="active"
)

# Update thread
thread.status = "completed"
await thread.save()

# Delete thread
await thread.delete()
```

### Common Queries

#### Get Recent Events
```python
from shared.models import Event
from datetime import datetime, timedelta

yesterday = datetime.now() - timedelta(days=1)
recent = await Event.filter(
    timestamp__gte=yesterday,
    event_type="MessageFromUser"
).order_by('-timestamp').limit(10)
```

#### Get Thread with Messages
```python
from shared.models import Thread, Message

thread = await Thread.get(thread_id="thread_123")
messages = await Message.filter(thread_id=thread.thread_id).all()
```

#### Count Records
```python
from shared.models import Message

count = await Message.filter(thread_id="thread_123").count()
```

## Event Bus Integration

The event bus automatically initializes Tortoise ORM on startup:

```python
# In event_bus/server.py

@app.on_event("startup")
async def startup_event():
    await init_db()
    print("🗄️  Database initialized")

@app.on_event("shutdown")
async def shutdown_event():
    await close_db()
    print("🗄️  Database closed")
```

No additional setup needed when using the event bus!

## Database Migrations with Aerich

### Initialize Aerich

```bash
aerich init -t shared.models.TORTOISE_ORM_CONFIG
aerich init-db
```

### Create Migration

After changing models:
```bash
aerich migrate --name "description_of_changes"
```

### Apply Migration

```bash
aerich upgrade
```

### Rollback Migration

```bash
aerich downgrade
```

## Model Reference

### Thread
```python
thread = await Thread.create(
    thread_id="thread_123",
    user_id="user_1",
    status="active",
    metadata={"key": "value"}
)
```

### Message
```python
message = await Message.create(
    thread_id="thread_123",
    message_id="msg_456",
    timestamp=datetime.now(),
    role="user",
    sender="user_1",
    content="Hello!",
    message_type="text",
    metadata={"key": "value"}
)
```

### Event
```python
event = await Event.create(
    event_id="evt_789",
    thread_id="thread_123",
    timestamp=datetime.now(),
    event_type="MessageFromUser",
    sender="user_1",
    payload={"content": "Hello!"},
    metadata={"key": "value"}
)
```

### AgentActivity
```python
activity = await AgentActivity.create(
    thread_id="thread_123",
    agent_name="cs_agent",
    activity_type="tool_call",
    description="Called search tool",
    tool_name="search",
    tool_input='{"query": "docs"}',
    tool_output='{"results": [...]}'
)
```

### EvalSuite
```python
suite = await EvalSuite.create(
    suite_id="suite_123",
    thread_id="thread_123",
    spec_name="CustomerSuccess",
    spec_version="1.0",
    target_agent_url="http://localhost:2024",
    target_agent_id="cs_agent",
    test_cases=[{"id": "test_1", "input": "..."}]
)
```

### TestResult
```python
result = await TestResult.create(
    result_id="result_123",
    suite_id="suite_123",
    thread_id="thread_123",
    test_case_id="test_1",
    input_sent="Hello",
    actual_output="Hi there!",
    expected_behavior="Should greet",
    passed=True,
    score=1.0,
    judge_reasoning="Correct greeting"
)
```

### Subscriber
```python
subscriber = await Subscriber.create(
    agent_name="cs_agent",
    filters={"event_types": ["MessageFromUser"]},
    status="active"
)
```

## Tips

### 1. Always Use Async
All database operations are async. Use `await`:
```python
threads = await Thread.all()  # ✅ Correct
threads = Thread.all()         # ❌ Wrong
```

### 2. Initialize Once
Initialize Tortoise ORM once at startup:
```python
await init_db()  # Call once at app startup
```

### 3. Close on Shutdown
Close connections gracefully:
```python
await close_db()  # Call once at app shutdown
```

### 4. Use get_or_create
Avoid duplicates:
```python
thread, created = await Thread.get_or_create(
    thread_id="thread_123",
    defaults={"user_id": "user_1"}
)
```

### 5. Bulk Operations
For performance:
```python
# Bulk create
await Message.bulk_create([msg1, msg2, msg3])

# Bulk update
await Message.filter(thread_id="thread_123").update(status="read")

# Bulk delete
await Message.filter(thread_id="thread_123").delete()
```

## Troubleshooting

### Database Already Initialized
```python
# Close and reinitialize
await close_db()
await init_db()
```

### Connection Errors
```python
# Check database path
from shared.database import get_db
db = get_db()
print(f"Database path: {db.db_path}")
```

### Query Errors
```python
# Use .values() for debugging
events = await Event.filter(thread_id="thread_123").values()
print(events)
```

## Resources

- [Tortoise ORM Documentation](https://tortoise.github.io/)
- [Aerich Documentation](https://github.com/tortoise/aerich)
- [Migration Guide](TORTOISE_ORM_MIGRATION.md)

---

**Version**: 0.2.0  
**Last Updated**: October 19, 2025

