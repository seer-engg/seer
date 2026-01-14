---
sidebar_position: 2
---

# Workflow Triggers

## Overview
Workflow triggers allow workflows to be executed automatically in response to events.

## Trigger Catalog
- `GET /api/v1/triggers` exposes the trigger catalog (currently `webhook.generic`) including normalized event schemas.

## Setting Up Triggers
- Attach triggers to saved workflows via `/api/v1/trigger-subscriptions` to configure filters, `${event...}` bindings, and per-subscription webhook secrets.
- Generic webhooks POST to `/api/v1/webhooks/generic/{subscription_id}` with the `X-Seer-Webhook-Secret` header; events are deduped, stored, and dispatched asynchronously.

## Observability
- Triggered runs are persisted in `workflow_runs` with `source="trigger"` plus links back to the originating subscription and event for observability.

## Worker Setup
- A dedicated Taskiq worker handles trigger polling, webhook dispatch, and saved-workflow execution so the FastAPI app stays responsive.
- Start it with `taskiq worker worker.broker:broker` (remember to point `REDIS_URL` at your Redis instance).

## Related Documentation
- [Configuration Reference](./CONFIGURATION)
