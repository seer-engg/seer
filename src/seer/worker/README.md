> **[← Back to Main README](../README.md)** | **[Full Documentation](../docs/)**

# Seer Worker

Taskiq powers our background worker so that long-running operations do not block the FastAPI app. The worker listens on Redis Streams, fans out jobs, coordinates trigger polling, and keeps a database connection pool ready for workflow executions.

## Architecture

- `worker.broker` configures a `RedisStreamBroker` plus `RedisAsyncResultBackend`. The broker url comes from, in priority order, `config.redis_url`, `REDIS_URL`, then `redis://localhost:6379/0`.
- On `TaskiqEvents.WORKER_STARTUP` the worker:
  - Initializes shared DB connections (`init_db`).
  - Boots a `TriggerPollScheduler` if `config.trigger_poller_enabled` is true. Scheduler cadence, batch size, and lock timeout all come from `config.trigger_poller_*` settings.
- On shutdown it stops the poll scheduler (if running) and calls `close_db` to free resources.

## Available tasks

| Task | Module | Purpose |
| ---- | ------ | ------- |
| `poll_triggers_once` | `worker.tasks.polling` | Runs a single `TriggerPollEngine.tick()` for ad-hoc debugging without starting the full scheduler. |
| `process_trigger_event(subscription_id, event_id)` | `worker.tasks.triggers` | Executes trigger bindings + workflow dispatch for a leased trigger event. API services enqueue this via `process_trigger_event.kiq(...)`. |
| `execute_saved_workflow(run_id, user_id)` | `worker.tasks.workflows` | Replays a persisted workflow run asynchronously so API requests can return immediately. |

All tasks are async functions decorated with `@broker.task`, which gives the API layer a `.kiq()` helper for enqueueing.

## Configuration

| Setting | Source | Description |
| ------- | ------ | ----------- |
| `redis_url` | `shared.config.SeerConfig` (env `REDIS_URL`) | Redis connection string for both broker and result backend. |
| `DATABASE_URL` | env / `.env` | Used by Tortoise ORM when the worker initializes the DB. |
| `trigger_poller_enabled` | config/env | Toggles background polling entirely. Set `False` to disable in dev. |
| `trigger_poller_interval_seconds` | config/env | Sleep between scheduler ticks. |
| `trigger_poller_max_batch_size` | config/env | Maximum trigger subscriptions leased per tick. |
| `trigger_poller_lock_timeout_seconds` | config/env | Lease/lock duration in seconds for a polled trigger subscription. |

Populate these via `.env`, Docker compose overrides, or direct environment variables before starting the worker.

## Running the worker

Local development:

```bash
uv run taskiq worker worker.broker:broker
```

Docker compose already defines a `taskiq-worker` service that runs the same command after Redis/Postgres are healthy. The worker only needs network access to Redis and the database; the FastAPI app communicates via Redis tasks.

## Development tips

- The poll scheduler only runs inside the worker process. If you want to manually advance polling logic (for tests or prod debugging), enqueue `poll_triggers_once.kiq()` or run the task directly via `uv run taskiq run worker.tasks.polling:poll_triggers_once`.
- Because Taskiq registers shutdown hooks, always let the worker exit cleanly (Ctrl+C) so the scheduler stops and DB connections close.
- Enable debug logs by setting `LOG_LEVEL=DEBUG` to trace polling leases and workflow dispatches.

## Related Documentation

- [Workflow Triggers](../docs/advanced/TRIGGERS.md) - Trigger setup and configuration
- [Configuration Reference](../docs/advanced/CONFIGURATION.md) - Complete configuration options
- [Main README](../README.md) - Quick start and installation
