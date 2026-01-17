# Triggers API Module

**Purpose**: Workflow trigger system - catalog of trigger types, subscription management, and background polling engine.

## Architecture

```
Trigger Subscription (workflow + trigger config)
    ↓
Polling Engine (background scheduler)
    ↓
Poll Adapter (trigger-specific polling logic)
    ↓
Detect Events
    ↓
Event Deduplication (prevent duplicate execution)
    ↓
Workflow Execution (api/workflows/services/execution.py)
```

## Core Components

### 1. Trigger Catalog (`services.py`)

**Purpose**: Registry of available trigger types

```python
from seer.api.triggers.services import list_triggers

triggers = await list_triggers()
# Returns list of TriggerDefinition with metadata
```

**Trigger Types**:
- `cron_schedule` - Time-based scheduling
- `gmail_email_received` - New Gmail emails
- `github_pull_request` - GitHub PR events
- `supabase_webhook` - Supabase database events
- `form_submission` - Form trigger (separate module)

### 2. Trigger Subscriptions

**Model**: `TriggerSubscription` (shared/database/models_triggers.py)

```python
class TriggerSubscription(Model):
    user: ForeignKey[User]
    workflow: ForeignKey[Workflow]
    trigger_type: str
    config: dict                # Trigger-specific config
    enabled: bool
```

**Subscription Lifecycle**:
1. User creates subscription via `POST /v1/trigger-subscriptions`
2. Subscription stored in database
3. Polling engine picks up enabled subscriptions
4. On event detection → execute workflow

### 3. Polling Engine (`polling/engine.py` + `polling/scheduler.py`)

**Purpose**: Background process polling for trigger events

**Architecture**:
```
Scheduler (APScheduler)
    ↓ Every 30s-5min (per trigger type)
PollEngine.poll_subscription(subscription)
    ↓
PollAdapter.poll(subscription.config)
    ↓
Return events[]
    ↓
Deduplicate events
    ↓
Execute workflow for each event
```

**Engine Flow**:
```python
async def poll_subscription(subscription: TriggerSubscription):
    adapter = get_poll_adapter(subscription.trigger_type)
    events = await adapter.poll(subscription.config, subscription.user)

    for event in events:
        # Deduplicate
        if await is_duplicate_event(event):
            continue

        # Store event
        trigger_event = await TriggerEvent.create(
            subscription=subscription,
            event_data=event,
            event_hash=hash_event(event)
        )

        # Execute workflow
        await run_saved_workflow(
            user=subscription.user,
            workflow_id=subscription.workflow.workflow_id,
            payload={"trigger_event": event},
            source=WorkflowRunSource.TRIGGER
        )
```

### 4. Poll Adapters (`polling/adapters/`)

**Purpose**: Trigger-specific polling logic

```python
# polling/adapters/base.py
class PollAdapter(Protocol):
    trigger_type: str

    async def poll(
        config: dict,
        user: User
    ) -> list[dict]:
        """Poll for events, return list of event dicts"""
```

**Implementations**:
- `cron_schedule.py` - Scheduled execution (evaluates cron expression)
- `gmail_email_received.py` - Gmail API polling for new emails

**Example: Gmail Adapter**

```python
class GmailEmailReceivedAdapter(PollAdapter):
    trigger_type = "gmail_email_received"

    async def poll(self, config, user):
        connection_id = config["connection_id"]
        query = config.get("query", "is:unread")

        # Get OAuth token
        connection, token = await get_oauth_token(user, connection_id=connection_id)

        # Poll Gmail API
        messages = await gmail_api.list_messages(token, query=query, max_results=10)

        # Return events
        return [
            {
                "message_id": msg["id"],
                "subject": msg["subject"],
                "from": msg["from"],
                "timestamp": msg["timestamp"]
            }
            for msg in messages
        ]
```

### 5. Event Deduplication (`polling/dedupe.py`)

**Purpose**: Prevent same event triggering multiple executions

**Strategy**: SHA256 hash of event data + 24h TTL

```python
async def is_duplicate_event(subscription, event_data):
    event_hash = _hash_event(event_data)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    existing = await TriggerEvent.filter(
        subscription=subscription,
        event_hash=event_hash,
        created_at__gte=cutoff
    ).first()

    return existing is not None
```

**Hash calculation**:
```python
def _hash_event(event_data: dict) -> str:
    serialized = json.dumps(event_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()
```

## Subscription Management

### Create Subscription

```
POST /v1/trigger-subscriptions
{
  "workflow_id": "wf_abc123",
  "trigger_type": "gmail_email_received",
  "config": {
    "connection_id": "conn_xyz",
    "query": "from:boss@company.com is:unread"
  },
  "enabled": true
}
```

### Update Subscription

```
PATCH /v1/trigger-subscriptions/{subscription_id}
{
  "config": {
    "query": "from:anyone@company.com is:unread"
  },
  "enabled": false
}
```

### Test Subscription

```
POST /v1/trigger-subscriptions/{subscription_id}/test
{
  "sample_event": {...}
}
```

**Response**: Executes workflow once with sample event, returns run details.

## Polling Schedule

**Managed by APScheduler** (`polling/scheduler.py`):

```python
# Default polling intervals per trigger type
POLL_INTERVALS = {
    "cron_schedule": 60,          # Check every 60s
    "gmail_email_received": 300,  # Poll every 5 min
    "github_pull_request": 180    # Poll every 3 min
}
```

**Scheduler**:
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

for trigger_type, interval in POLL_INTERVALS.items():
    scheduler.add_job(
        poll_trigger_type,
        'interval',
        seconds=interval,
        args=[trigger_type]
    )

scheduler.start()
```

## API Endpoints

### Trigger Catalog

- `GET /v1/triggers` - List available trigger types

### Subscription Management

- `POST /v1/trigger-subscriptions` - Create subscription
- `GET /v1/trigger-subscriptions` - List subscriptions (optionally filter by workflow_id)
- `GET /v1/trigger-subscriptions/{id}` - Get subscription
- `PATCH /v1/trigger-subscriptions/{id}` - Update subscription
- `DELETE /v1/trigger-subscriptions/{id}` - Delete subscription
- `POST /v1/trigger-subscriptions/{id}/test` - Test subscription with sample event

## Adding New Trigger Type

### Step 1: Create Poll Adapter

```python
# polling/adapters/my_trigger.py
from seer.api.triggers.polling.adapters.base import PollAdapter

class MyTriggerAdapter(PollAdapter):
    trigger_type = "my_trigger"

    async def poll(self, config: dict, user: User) -> list[dict]:
        # 1. Extract config
        api_key = config["api_key"]

        # 2. Poll external API
        events = await my_api.get_events(api_key)

        # 3. Return event list
        return [
            {
                "event_id": event["id"],
                "data": event["data"],
                "timestamp": event["timestamp"]
            }
            for event in events
        ]
```

### Step 2: Register Adapter

```python
# polling/adapters/__init__.py
from seer.api.triggers.polling.adapters.registry import AdapterRegistry
from .my_trigger import MyTriggerAdapter

AdapterRegistry.register(MyTriggerAdapter())
```

### Step 3: Add Trigger Definition

```python
# services.py
TRIGGER_DEFINITIONS = [
    {
        "type": "my_trigger",
        "name": "My Trigger",
        "description": "Triggers on my event",
        "config_schema": {
            "type": "object",
            "properties": {
                "api_key": {
                    "type": "string",
                    "description": "API key for my service"
                }
            },
            "required": ["api_key"]
        }
    },
    # ...
]
```

### Step 4: Configure Polling Interval

```python
# polling/scheduler.py
POLL_INTERVALS["my_trigger"] = 120  # Poll every 2 minutes
```

## Workflow Trigger Inputs

Workflows receive trigger event data as input:

```python
# In workflow spec
{
  "nodes": {
    "trigger_node": {
      "type": "trigger",
      "config": {}
    },
    "process_event": {
      "type": "code",
      "config": {
        "code": "email = inputs['trigger_event']['subject']"
      }
    }
  }
}
```

**Runtime state**:
```python
{
    "workflow_inputs": {
        "trigger_event": {
            "message_id": "msg_123",
            "subject": "Important Email",
            "from": "boss@company.com"
        }
    }
}
```

## Webhook vs Polling

### Polling Triggers (Current)
- ✅ No webhook setup required
- ✅ Works with APIs that don't support webhooks
- ❌ Latency (polls every N minutes)
- ❌ API rate limit concerns

### Webhook Triggers (Planned)
- ✅ Real-time (instant execution)
- ✅ No polling overhead
- ❌ Requires webhook endpoint exposure
- ❌ Provider must support webhooks

**Current webhook support**: Supabase (see `/api/webhooks/`)

## Known Issues & Improvements Planned

- [ ] **Webhook support**: Add webhook receiver for real-time triggers
- [ ] **Polling efficiency**: Batch subscriptions of same type
- [ ] **Error handling**: Retry failed polls with exponential backoff
- [ ] **Monitoring**: Track polling health, event throughput
- [ ] **Schema inference**: Auto-infer event schema from sample events (see `schema_inference.py`)

## Related Documentation

- [Workflows API](../workflows/README.md) - Workflow execution triggered by events
- [Database Models](../../shared/database/README.md) - TriggerSubscription, TriggerEvent schemas
- [Tools System](../../shared/tools/README.md) - OAuth token management for API polling
