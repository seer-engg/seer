# Seer Database Documentation

## Overview

Seer now uses SQLite for persistent storage of all chat threads, messages, events, and evaluation data. This replaces the previous in-memory and file-based storage systems.

## Database Location

By default, the database is stored at:
```
/home/ubuntu/lokesh/learning_agent/data/seer.db
```

## Schema

### Tables

#### 1. `threads`
Stores conversation threads.

| Column | Type | Description |
|--------|------|-------------|
| thread_id | TEXT (PK) | Unique thread identifier |
| created_at | TIMESTAMP | Thread creation time |
| updated_at | TIMESTAMP | Last update time |
| user_id | TEXT | User identifier |
| status | TEXT | Thread status (active, archived) |
| metadata | TEXT (JSON) | Additional metadata |

#### 2. `messages`
Stores all messages in threads.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Auto-increment ID |
| thread_id | TEXT | Reference to thread |
| message_id | TEXT (UNIQUE) | Unique message identifier |
| timestamp | TIMESTAMP | Message timestamp |
| role | TEXT | Message role (user, assistant) |
| sender | TEXT | Sender identifier |
| content | TEXT | Message content |
| message_type | TEXT | Message type |
| metadata | TEXT (JSON) | Additional metadata |

#### 3. `events`
Stores all event bus events.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Auto-increment ID |
| event_id | TEXT (UNIQUE) | Unique event identifier |
| thread_id | TEXT | Reference to thread |
| timestamp | TIMESTAMP | Event timestamp |
| event_type | TEXT | Event type |
| sender | TEXT | Sender identifier |
| payload | TEXT (JSON) | Event payload |
| metadata | TEXT (JSON) | Additional metadata |

#### 4. `agent_activities`
Tracks what each agent did in threads.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Auto-increment ID |
| thread_id | TEXT | Reference to thread |
| agent_name | TEXT | Agent identifier |
| timestamp | TIMESTAMP | Activity timestamp |
| activity_type | TEXT | Activity type |
| description | TEXT | Activity description |
| tool_name | TEXT | Tool used (if applicable) |
| tool_input | TEXT | Tool input |
| tool_output | TEXT | Tool output |
| metadata | TEXT (JSON) | Additional metadata |

#### 5. `eval_suites`
Stores generated evaluation suites.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Auto-increment ID |
| suite_id | TEXT (UNIQUE) | Unique suite identifier |
| thread_id | TEXT | Reference to thread |
| spec_name | TEXT | Spec name |
| spec_version | TEXT | Spec version |
| target_agent_url | TEXT | Target agent URL |
| target_agent_id | TEXT | Target agent ID |
| test_cases | TEXT (JSON) | Test cases array |
| created_at | TIMESTAMP | Creation time |
| metadata | TEXT (JSON) | Additional metadata |

#### 6. `test_results`
Stores test execution results.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Auto-increment ID |
| result_id | TEXT (UNIQUE) | Unique result identifier |
| suite_id | TEXT | Reference to eval suite |
| thread_id | TEXT | Reference to thread |
| test_case_id | TEXT | Test case identifier |
| input_sent | TEXT | Input message |
| actual_output | TEXT | Agent's actual output |
| expected_behavior | TEXT | Expected behavior |
| passed | BOOLEAN | Pass/fail status |
| score | REAL | Score (0.0-1.0) |
| judge_reasoning | TEXT | Judge's reasoning |
| created_at | TIMESTAMP | Creation time |
| metadata | TEXT (JSON) | Additional metadata |

#### 7. `subscribers`
Tracks active event bus subscribers.

| Column | Type | Description |
|--------|------|-------------|
| agent_name | TEXT (PK) | Agent identifier |
| registered_at | TIMESTAMP | Registration time |
| last_poll | TIMESTAMP | Last poll time |
| message_count | INTEGER | Message count |
| publish_count | INTEGER | Publish count |
| filters | TEXT (JSON) | Subscription filters |
| status | TEXT | Subscriber status |
| metadata | TEXT (JSON) | Additional metadata |

## Initialization

### Fresh Installation

```bash
python init_db.py
```

This will create the database with all tables at `data/seer.db`.

### Reset Database

```bash
python init_db.py --reset
```

This will delete the existing database and create a fresh one.

### Migration

To migrate existing eval JSON files to the database:

```bash
python init_db.py --migrate
```

This will:
1. Find all eval suite files in `agents/eval_agent/generated_evals/`
2. Import them into the `eval_suites` table
3. Import all test results into the `test_results` table

## Usage

### In Python Code

```python
from shared.database import get_db

# Get database instance
db = get_db()

# Create a thread
db.create_thread("thread-123", user_id="user-1")

# Add a message
db.add_message(
    thread_id="thread-123",
    message_id="msg-1",
    timestamp="2025-10-18T12:00:00",
    role="user",
    sender="user-1",
    content="Hello!"
)

# Get thread messages
messages = db.get_thread_messages("thread-123")

# Save eval suite
db.save_eval_suite(
    suite_id="eval-1",
    spec_name="my_agent",
    spec_version="1.0",
    test_cases=[...],
    target_agent_url="http://localhost:2024",
    target_agent_id="my_agent"
)

# Get eval suites
suites = db.get_eval_suites(
    target_agent_url="http://localhost:2024",
    target_agent_id="my_agent"
)
```

### Via Event Bus API

The Event Bus automatically persists all data to the database:

```bash
# Get threads
curl http://localhost:8000/threads

# Get events history
curl "http://localhost:8000/history?limit=50"

# Get eval suites
curl "http://localhost:8000/evals?agent_url=http://localhost:2024&agent_id=my_agent"

# Get test results
curl http://localhost:8000/test_results/eval-suite-id
```

## Benefits

### 1. **Persistence**
- All data survives restarts
- No data loss on crashes
- Complete history available

### 2. **Queryability**
- Fast queries with indexes
- Filter by thread, agent, time, etc.
- Complex analytics possible

### 3. **Scalability**
- Handles large volumes of data
- Efficient storage
- Good performance

### 4. **Debugging**
- Complete audit trail
- See what each agent did
- Track down issues easily

### 5. **Integration**
- Easy to integrate with other tools
- Standard SQL interface
- Export data as needed

## Maintenance

### Backup

```bash
# Create backup
cp data/seer.db data/seer_backup_$(date +%Y%m%d).db

# Or use SQLite backup
sqlite3 data/seer.db ".backup 'data/seer_backup.db'"
```

### Vacuum (Optimize)

```bash
sqlite3 data/seer.db "VACUUM;"
```

### Check Size

```bash
ls -lh data/seer.db
```

### Inspect Data

```bash
# Open SQLite shell
sqlite3 data/seer.db

# Show tables
.tables

# Show schema
.schema threads

# Query data
SELECT * FROM threads LIMIT 10;

# Count records
SELECT COUNT(*) FROM events;
```

## Migration from File-Based Storage

Old eval files (`agents/eval_agent/generated_evals/*.json`) are kept for backward compatibility but new data goes to the database. To migrate:

```bash
python init_db.py --migrate
```

This is safe to run multiple times - it skips already-migrated data.

## Troubleshooting

### Database Locked Error

If you see "database is locked" errors:
1. Check if multiple processes are accessing the database
2. Ensure only one event bus instance is running
3. Database uses thread-safe connections automatically

### Corrupted Database

If database is corrupted:
```bash
# Backup first
cp data/seer.db data/seer_corrupted.db

# Try to recover
sqlite3 data/seer.db ".recover" | sqlite3 data/seer_recovered.db

# Or reset and re-migrate
python init_db.py --reset --migrate
```

### Performance Issues

If queries are slow:
1. Database has indexes on common query fields
2. Use VACUUM to optimize: `sqlite3 data/seer.db "VACUUM;"`
3. Consider limiting query results with LIMIT
4. Archive old threads if database is very large

## Future Enhancements

Potential improvements:
- Archive old threads to separate database
- Add database replication for high availability
- Export to other databases (PostgreSQL, MySQL)
- Add database analytics dashboard
- Automated backup system

