# Browserless Scaling Analysis for AWS Production

## Table of Contents

### Understanding the Problem
1. [Current Architecture](#current-architecture)
2. [Why the Current Design Doesn't Scale](#why-the-current-design-doesnt-scale)
3. [Can We Ditch WebSockets?](#can-we-ditch-websockets)
4. [The Real Problem: Stateful Application Layer](#the-real-problem)
5. [Refactored Architecture: Stateless App Layer](#refactored-architecture) — **the key section**
6. [Browserless Built-in Features We're Not Using](#browserless-builtins) — **quick wins**
7. [Core Challenges with Headless Browsers at Scale](#core-challenges)

### Infrastructure Options
8. [Scaling Options Compared](#scaling-options-compared)
9. [Recommended Architecture: ECS Fargate](#recommended-architecture)
10. [Alternative: ECS EC2 (Graduation Path)](#alternative-ecs-ec2)
11. [Alternative: EKS with Custom Metrics](#alternative-eks)

### Implementation Details
12. [Pool Manager Refactoring for Multi-Instance](#pool-manager-refactoring)
13. [HITL Implications](#hitl-implications)
14. [Configuration & Environment Variables](#configuration)
15. [Cost Estimation](#cost-estimation)
16. [Migration Path](#migration-path)

---

## 1. Current Architecture <a name="current-architecture"></a>

### How it works today (docker-compose / local)

```
┌──────────────┐    WebSocket (CDP)    ┌──────────────────────┐
│  API / Worker │ ──────────────────► │  Single Browserless  │
│  (Python)     │    ws://browserless  │  Container           │
│               │    :3000             │  CONCURRENT=5        │
│  BrowserPool  │                      │  TIMEOUT=300s        │
│  Manager      │                      │  ghcr.io/browserless │
│  (singleton)  │                      │  /chromium           │
└──────────────┘                      └──────────────────────┘
```

### Key components

| Component | File | Role |
|-----------|------|------|
| `BrowserPoolManager` | `services/browser/pool_manager.py` | Singleton. Semaphore-limited concurrency. Tracks sessions in `_sessions` dict. Runs reaper task. |
| `BrowserService` | `services/browser/browser_service.py` | Singleton. Creates sessions via pool, runs browser-use Agent, handles HITL, manages recordings. |
| `hitl_bridge.py` | `services/browser/hitl_bridge.py` | In-process dict of `asyncio.Future` objects keyed by `(run_id, request_id)`. Enables human-in-the-loop blocking. |
| Config | `config.py:555-564` | `browserless_url`, `browserless_token`, `browser_pool_max_concurrent` (default 5) |
| Docker | `docker-compose.yml:45-58` | Single `ghcr.io/browserless/chromium` container, health check on `/active` |

### Session lifecycle

```
create_session()
  → semaphore.acquire()
  → BrowserSession(cdp_url=config.browserless_cdp_url)
  → session.start()   (opens WebSocket to Browserless)
  → apply cookies via CDP
  → store in _sessions dict

[agent runs browser automation over the live CDP WebSocket]

release_session()
  → export cookies via CDP
  → session.stop()     (closes WebSocket)
  → semaphore.release()
```

---

## 2. Why the Current Design Doesn't Scale <a name="why-the-current-design-doesnt-scale"></a>

### Problem 1: Single Browserless instance = single point of failure

The `browserless_url` config points to exactly one WebSocket endpoint. If that container dies, all active sessions are lost and no new sessions can be created.

### Problem 2: Stateful WebSocket sessions are pinned to one container

Each `BrowserSession` maintains a persistent CDP WebSocket connection to a specific Browserless container. You cannot:
- Move a session mid-flight to another container
- Resume a dropped WebSocket on a different container
- Load-balance individual CDP commands across containers

**This is the fundamental constraint.** A browser session is bound to the container that spawned it for its entire lifetime.

### Problem 3: In-process state cannot survive process restarts

- `BrowserPoolManager._sessions` is an in-memory dict — if the API/worker process restarts, all session handles are lost (the browsers keep running on Browserless but become orphaned).
- `hitl_bridge._pending_futures` is also in-memory — if the process hosting the HITL wait restarts, the human response has nowhere to go.

### Problem 4: Semaphore is local, not distributed

The `asyncio.Semaphore(max_concurrent)` only limits concurrency within a single Python process. If you scale to multiple API replicas, each replica will independently allow `max_concurrent` sessions, potentially exceeding what Browserless can handle.

### Problem 5: No backpressure from Browserless to the application

The pool manager does not check Browserless capacity before creating a session. It only uses its local semaphore. If Browserless is overloaded (all `CONCURRENT` slots filled), the `BrowserSession.start()` call will either queue (if `QUEUED > 0` on Browserless) or fail with a connection error.

---

## 3. Can We Ditch WebSockets? <a name="can-we-ditch-websockets"></a>

**Short answer: No — but you don't need to.**

### Why CDP WebSockets are unavoidable

Chrome DevTools Protocol (CDP) is a **bidirectional, stateful protocol** over WebSocket. It's how *all* browser automation works:

```
Your App  ←──CDP WebSocket──→  Chrome Process (inside Browserless)
              │
              ├─ Navigate to URL
              ├─ Click element
              ├─ Get DOM snapshot
              ├─ Export cookies
              └─ Stream screencast frames
```

There is no REST/HTTP alternative for browser automation. Playwright, Puppeteer, browser-use — they all use CDP under the hood. The WebSocket is the wire between your code and a running Chromium process. You cannot serialize a running browser and resume it elsewhere, just like you can't serialize a running process.

### What about alternative protocols?

| Protocol | Can it replace CDP? | Why / Why not |
|----------|-------------------|---------------|
| **BiDi (WebDriver BiDi)** | Not yet | W3C successor to CDP. Also WebSocket-based. Still in early spec — Browserless doesn't support it yet. Same statefulness constraints. |
| **HTTP REST APIs** (ScrapingBee, etc.) | No | These are stateless scraping APIs — send URL, get HTML back. No interactive sessions, no cookie persistence, no HITL, no screencast. Completely different paradigm. |
| **Browserless REST endpoints** (`/content`, `/pdf`, `/screenshot`) | Partially | Browserless exposes stateless HTTP endpoints for simple tasks. But you use `BrowserSession` for multi-step, agent-driven automation with HITL — REST can't do that. |

### The insight

**The WebSocket between Browserless and Chrome is fine — it scales horizontally behind an ALB.** The problem isn't Browserless or CDP. The problem is that your **application layer** (`pool_manager.py`, `hitl_bridge.py`) keeps critical state in Python process memory, making *your app* the bottleneck.

```
The scaling bottleneck is NOT here:
   Browserless ←──CDP──→ Chrome     ← This scales fine (just add containers)

The bottleneck IS here:
   Your App (Python process)         ← In-memory dicts, local semaphore, local Futures
     ├─ _sessions: Dict              ← Lost on restart
     ├─ _semaphore: asyncio          ← Only limits this process
     └─ _pending_futures: Dict       ← Lost on restart
```

---

## 4. The Real Problem: Stateful Application Layer <a name="the-real-problem"></a>

Here's every piece of in-memory state in your browser stack and what it prevents:

### State inventory

| State | Location | Type | Prevents |
|-------|----------|------|----------|
| `_sessions` dict | `pool_manager.py:68` | `Dict[str, ManagedSession]` | **Horizontal scaling of API/worker.** Each replica has its own session dict. Replica B can't release a session created by Replica A. |
| `asyncio.Semaphore` | `pool_manager.py:67` | Local counter | **Correct concurrency limiting.** 3 replicas × 5 semaphore = 15 sessions attempted, but Browserless may only handle 10. |
| `_pending_futures` dict | `hitl_bridge.py:20` | `Dict[(run_id, req_id), Future]` | **HITL across replicas.** The API replica receiving the human response must be the same one holding the Future. |
| `ManagedSession` object | `pool_manager.py:32` | dataclass with `BrowserSession` handle | **Session recovery.** If the process dies, the CDP WebSocket handle is lost. The browser keeps running on Browserless as an orphan. |
| `_reaper_task` | `pool_manager.py:69` | `asyncio.Task` | **Orphan cleanup.** Only one reaper runs per process. With multiple replicas, each has its own reaper — but none can clean sessions from other replicas. |
| `BrowserService._instance` | `browser_service.py:236` | Singleton | **Not a problem** — this is stateless logic, the singleton pattern is fine. |
| `SessionContextManager` | `session_context_manager.py` | DB-backed | **Not a problem** — this correctly uses the database for persistence. |

### What needs to change

The goal is: **make the Python process disposable.** Any replica can handle any request. If a replica dies, its sessions are cleaned up, and HITL responses find their way to the right place.

---

## 5. Refactored Architecture: Stateless App Layer <a name="refactored-architecture"></a>

### Architecture overview

```
BEFORE (current):
┌─────────────────────────────┐         ┌──────────────┐
│  Python Process (API/Worker) │         │  Browserless  │
│                              │  CDP WS │  (single)     │
│  BrowserPoolManager          │────────►│               │
│    ._sessions  (in-memory)   │         └──────────────┘
│    ._semaphore (local)       │
│                              │
│  hitl_bridge                 │
│    ._pending_futures (local) │
└─────────────────────────────┘

AFTER (refactored):
┌──────────────────────┐
│  Python Replica A     │         ┌───────────────┐
│                       │  CDP WS │               │
│  BrowserPoolManager   │────────►│  ALB          │
│    (thin wrapper)     │         │    │          │
└──────┬───────────────┘         │  ┌▼────────┐  │
       │                          │  │BL Task 1│  │
       │  Redis/Valkey            │  │BL Task 2│  │
       │  (shared state)          │  │BL Task N│  │
       ▼                          │  └─────────┘  │
┌──────────────────────┐         └───────────────┘
│  Valkey               │
│                       │
│  sessions:{sid}  ────►│  Session registry (who owns what)
│  semaphore:browser ──►│  Distributed concurrency limit
│  hitl:{run}:{req} ───►│  HITL wait state + pub/sub
│  orphan:cleanup  ────►│  Orphan detection
└──────────────────────┘

┌──────────────────────┐
│  Python Replica B     │
│                       │  CDP WS
│  BrowserPoolManager   │────────► (same ALB)
│    (thin wrapper)     │
└──────┬───────────────┘
       │
       └──── same Valkey ──┘
```

### Change 1: Session registry in Redis

**Current:** `self._sessions: Dict[str, ManagedSession]` — local dict, lost on restart.

**Refactored:** Two-tier tracking — Redis for cross-replica visibility, local dict for the CDP handle.

```python
# pool_manager.py — refactored concept

class BrowserPoolManager:
    """
    Hybrid session manager:
    - Redis: session registry (metadata, ownership, expiry)
    - Local: CDP WebSocket handles (can't be serialized)
    """

    def __init__(self, ...):
        self._local_sessions: Dict[str, ManagedSession] = {}  # CDP handles (this replica only)
        self._redis: Redis = get_valkey_client()               # shared state
        self._replica_id: str = str(uuid4())                   # unique per process

    async def create_session(self, ...) -> ManagedSession:
        # 1. Check distributed capacity BEFORE creating session
        current_count = await self._redis.get("browser:active_count") or 0
        if int(current_count) >= self._global_max_concurrent:
            raise BrowserPoolExhausted("Global browser pool is full")

        # 2. Atomically increment the distributed counter
        await self._redis.incr("browser:active_count")

        try:
            # 3. Create the CDP session (unchanged — still talks to Browserless via ALB)
            browser_session = BrowserSession(cdp_url=cdp_url, ...)
            await browser_session.start()

            managed = ManagedSession(id=session_id, session=browser_session, ...)

            # 4. Register in Redis (metadata only — not the WebSocket handle)
            await self._redis.hset(f"browser:session:{session_id}", mapping={
                "replica_id": self._replica_id,
                "user_id": user_id,
                "profile_id": profile_id or "",
                "session_type": session_type,
                "created_at": str(time.time()),
                "timeout": str(timeout),
                "hitl_paused": "0",
            })
            await self._redis.expire(f"browser:session:{session_id}", timeout + 60)  # auto-cleanup

            # 5. Keep local handle for CDP operations
            self._local_sessions[session_id] = managed
            return managed

        except Exception:
            await self._redis.decr("browser:active_count")
            raise

    async def release_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        managed = self._local_sessions.pop(session_id, None)
        if managed is None:
            return None

        # Export cookies, stop session (unchanged)
        storage_state = await self._export_and_stop(managed)

        # Clean up Redis
        await self._redis.delete(f"browser:session:{session_id}")
        await self._redis.decr("browser:active_count")

        return storage_state
```

**Why this works:**
- Redis tracks *existence* and *metadata* of all sessions across all replicas
- The local dict tracks *CDP handles* that can only live in the process that created them
- If a replica dies, its local handles are gone, but Redis still knows about those sessions → orphan cleanup can handle them
- Any replica can check `browser:active_count` for global capacity

### Change 2: Distributed concurrency limiting

**Current:** `asyncio.Semaphore(5)` — local to one process.

**Refactored:** Redis atomic counter (shown above in `browser:active_count`).

For a more robust approach, use Redis + local semaphore together:

```python
class BrowserPoolManager:
    def __init__(self, ...):
        # Local semaphore: limits THIS replica's sessions (still useful for process-level backpressure)
        self._local_semaphore = asyncio.Semaphore(config.browser_pool_max_concurrent_per_replica)

        # Global limit: enforced via Redis counter
        self._global_max = config.browser_pool_max_concurrent_global  # total across all replicas

    async def create_session(self, ...):
        # Local gate — prevents this replica from being greedy
        await self._local_semaphore.acquire()

        try:
            # Global gate — prevents all replicas combined from exceeding Browserless capacity
            count = await self._redis.incr("browser:active_count")
            if count > self._global_max:
                await self._redis.decr("browser:active_count")
                self._local_semaphore.release()
                raise BrowserPoolExhausted(
                    f"Global limit reached: {count-1}/{self._global_max}"
                )

            # ... create session ...
        except Exception:
            self._local_semaphore.release()
            raise
```

**New config values needed:**

```python
# config.py additions
browser_pool_max_concurrent_per_replica: int = Field(
    default=5,
    description="Max concurrent sessions per API/worker replica"
)
browser_pool_max_concurrent_global: int = Field(
    default=20,
    description="Max concurrent sessions across ALL replicas (should match total Browserless capacity)"
)
```

### Change 3: HITL state in Redis with pub/sub

**Current:** `hitl_bridge._pending_futures` — in-process dict. The API replica receiving the human response MUST be the same one holding the Future.

**Refactored:** Redis stores the HITL wait state. Pub/sub notifies the correct replica.

```python
# hitl_bridge.py — refactored concept

import asyncio
from typing import Any, Dict, Optional
from redis.asyncio import Redis

# Local futures (still needed — you can't put an asyncio.Future in Redis)
_local_futures: Dict[str, asyncio.Future] = {}

class HITLBridge:
    """
    Distributed HITL wait/resume via Redis.

    Flow:
    1. Agent calls ask_human → register_wait() stores state in Redis + creates local Future
    2. Human responds via ANY API replica → resolve_wait() publishes to Redis channel
    3. The replica holding the local Future receives the pub/sub message → resolves Future
    4. Agent continues
    """

    def __init__(self, redis: Redis, replica_id: str):
        self._redis = redis
        self._replica_id = replica_id
        self._subscriber_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start listening for HITL responses on the pub/sub channel."""
        self._subscriber_task = asyncio.create_task(self._listen_for_responses())

    async def _listen_for_responses(self):
        """Subscribe to HITL response channel and resolve local Futures."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe("hitl:responses")

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            key = f"{data['run_id']}:{data['request_id']}"
            future = _local_futures.pop(key, None)
            if future and not future.done():
                future.set_result(data["response"])

    async def register_wait(self, run_id: str, request_id: str) -> asyncio.Future:
        """Register a HITL wait — stores in Redis + creates local Future."""
        key = f"{run_id}:{request_id}"

        # Store in Redis so any replica can see pending waits
        await self._redis.hset(f"hitl:wait:{key}", mapping={
            "replica_id": self._replica_id,
            "run_id": run_id,
            "request_id": request_id,
            "status": "waiting",
            "created_at": str(time.time()),
        })

        # Create local Future for the agent to await
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        _local_futures[key] = future
        return future

    async def resolve_wait(self, run_id: str, request_id: str, response: Any) -> bool:
        """
        Resolve a HITL wait. Called by ANY API replica when human responds.
        Publishes to Redis pub/sub — the replica holding the Future will pick it up.
        """
        key = f"{run_id}:{request_id}"
        wait_data = await self._redis.hgetall(f"hitl:wait:{key}")
        if not wait_data:
            return False

        # Publish response — the owning replica's subscriber will resolve the Future
        await self._redis.publish("hitl:responses", json.dumps({
            "run_id": run_id,
            "request_id": request_id,
            "response": response,
        }))

        # Clean up Redis state
        await self._redis.delete(f"hitl:wait:{key}")
        return True
```

**Why this is critical:** In production, your ALB routes API requests round-robin. The human's resume POST might hit Replica B, but the `asyncio.Future` lives on Replica A. Without Redis pub/sub, the response is lost.

### Change 4: Orphan detection and cleanup

**Current:** If a Python process dies, its browser sessions keep running on Browserless but nothing tracks them.

**Refactored:** Each replica heartbeats. A cleanup task detects dead replicas and kills their orphaned Browserless sessions.

```python
# pool_manager.py — add to BrowserPoolManager

async def _heartbeat_loop(self):
    """Periodically refresh this replica's TTL in Redis."""
    while not self._shutdown:
        await self._redis.set(
            f"browser:replica:{self._replica_id}",
            str(len(self._local_sessions)),
            ex=30,  # expires in 30s if heartbeat stops
        )
        await asyncio.sleep(10)

async def _orphan_cleanup_loop(self):
    """
    Periodically scan for sessions owned by dead replicas.
    Only one replica should run this (use Redis lock).
    """
    while not self._shutdown:
        await asyncio.sleep(60)

        # Try to acquire cleanup lock (only one cleaner at a time)
        acquired = await self._redis.set("browser:cleanup_lock", "1", ex=120, nx=True)
        if not acquired:
            continue

        # Find all registered sessions
        session_keys = await self._redis.keys("browser:session:*")
        for key in session_keys:
            session_data = await self._redis.hgetall(key)
            replica_id = session_data.get("replica_id")

            # Check if owning replica is still alive
            replica_alive = await self._redis.exists(f"browser:replica:{replica_id}")
            if not replica_alive:
                session_id = key.split(":")[-1]
                logger.warning(f"Orphaned session {session_id} from dead replica {replica_id}")

                # Decrement global counter and remove session
                await self._redis.delete(key)
                await self._redis.decr("browser:active_count")

                # The browser on Browserless will be cleaned up by Browserless's own
                # TIMEOUT setting — we just fix the bookkeeping here.
```

**Why you need this:** Without orphan detection, dead replicas leave "phantom" sessions in the Redis counter. Over time, `browser:active_count` drifts upward, and eventually the global limit blocks new sessions even though Browserless has capacity.

### Change 5: Connection retry with backpressure

**Current:** `BrowserSession.start()` either succeeds or throws. No retry, no backpressure awareness.

**Refactored:** Retry with exponential backoff, and check Browserless `/pressure` endpoint.

```python
# pool_manager.py — replace direct browser_session.start()

MAX_CONNECT_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

async def _start_session_with_retry(self, browser_session: BrowserSession) -> None:
    """Start browser session with retry and backpressure awareness."""

    for attempt in range(MAX_CONNECT_RETRIES):
        # Optional: check Browserless pressure before connecting
        if attempt > 0:
            pressure = await self._check_browserless_pressure()
            if pressure and not pressure.get("isAvailable", True):
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(f"Browserless under pressure, waiting {wait}s before retry")
                await asyncio.sleep(wait)
                continue

        try:
            await browser_session.start()
            return
        except Exception as e:
            if attempt == MAX_CONNECT_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF_BASE * (attempt + 1)
            logger.warning(
                f"Browserless connect attempt {attempt+1}/{MAX_CONNECT_RETRIES} "
                f"failed: {e}, retrying in {wait}s"
            )
            await asyncio.sleep(wait)

async def _check_browserless_pressure(self) -> Optional[Dict]:
    """Query Browserless /pressure endpoint for load status."""
    try:
        http_url = config.browserless_url.replace("ws://", "http://").replace("wss://", "https://")
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{http_url}/pressure", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                return await resp.json()
    except Exception:
        return None
```

### Summary: What changes in each file

| File | Change | Effort |
|------|--------|--------|
| `pool_manager.py` | Add Redis session registry, distributed counter, heartbeat, orphan cleanup, connection retry | **Large** — this is the core refactor |
| `hitl_bridge.py` | Replace in-memory dict with Redis + pub/sub | **Medium** — new class, same interface |
| `config.py` | Add `browser_pool_max_concurrent_per_replica`, `browser_pool_max_concurrent_global` | **Small** |
| `browser_service.py` | Update `execute_task()` to use new HITLBridge class instead of module-level functions | **Small** — interface stays similar |
| `browser_node.py` | **No changes needed** — it only calls `BrowserService.execute_task()` | **None** |
| `session_context_manager.py` | **No changes needed** — already DB-backed | **None** |

### What stays the same

- **CDP WebSocket connections** — still used, still stateful, still bound to one Browserless container. This is correct and unavoidable.
- **BrowserService singleton** — still fine, it's stateless logic.
- **SessionContextManager** — already correct, DB-backed cookie persistence.
- **Cookie export/import flow** — the existing checkpoint pattern is exactly right for recovery.
- **Recording service** — already flushes to DB, not process-local.

---

## 6. Browserless Built-in Features We're Not Using <a name="browserless-builtins"></a>

Browserless v2 ships with significant scaling, health, and operational infrastructure that we're currently leaving on the table. This section covers what's available, what we use today, and what we should enable.

### 6.1 What we use today (docker-compose.yml)

```yaml
environment:
  TOKEN: "${BROWSERLESS_TOKEN:-}"        # Auth
  CONCURRENT: "${BROWSERLESS_MAX_CONCURRENT:-5}"  # Max sessions
  TIMEOUT: "${BROWSERLESS_TIMEOUT:-300000}"        # Session timeout
```

That's it — 3 env vars. Browserless has ~15+ config options for production use.

### 6.2 Health-gated admission (NOT enabled — should be)

```yaml
# ADD THESE to Browserless container environment
HEALTH: "true"                  # Enable pre-request health checks
MAX_CPU_PERCENT: "80"           # Reject new sessions above 80% CPU
MAX_MEMORY_PERCENT: "85"        # Reject new sessions above 85% memory
```

**How it works:** Before accepting a new CDP connection, Browserless checks current CPU and memory. If either exceeds the threshold, the new session is **rejected with 429** — existing sessions continue running normally.

**Why it matters:** Without this, Browserless accepts sessions until the container OOMs or Chrome becomes unresponsive. With an ALB in front, rejected requests route to healthier containers automatically.

**Current behavior (dangerous):** Browserless runs with defaults `MAX_CPU_PERCENT=99`, `MAX_MEMORY_PERCENT=99` — essentially no protection.

### 6.3 Queue management (NOT configured — should be)

```yaml
# ADD THIS
QUEUED: "10"                    # Queue up to 10 requests when all CONCURRENT slots are full
```

**How CONCURRENT + QUEUED work together:**

```
Request arrives
  ├─ CONCURRENT slots available? → Start session immediately
  ├─ All slots busy, QUEUED has room? → Wait in FIFO queue until a slot frees
  └─ Queue full too? → Reject with HTTP 429
```

**Current behavior:** `QUEUED` defaults to 10 silently. This is fine, but we should:
1. Set it explicitly so it's documented
2. **Handle 429 in pool_manager.py** — currently `BrowserSession.start()` throws an opaque exception on rejection

**Recommended pool_manager.py addition:**

```python
# In create_session(), wrap browser_session.start() with retry
MAX_CONNECT_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

for attempt in range(MAX_CONNECT_RETRIES):
    try:
        await browser_session.start()
        break
    except Exception as e:
        if "429" in str(e) or "too many" in str(e).lower():
            if attempt < MAX_CONNECT_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE * (attempt + 1)
                logger.warning(f"Browserless queue full, retry {attempt+1} in {wait}s")
                await asyncio.sleep(wait)
                continue
        raise
```

### 6.4 Monitoring endpoints (NOT used — should be)

Browserless exposes these HTTP endpoints:

| Endpoint | Returns | How we should use it |
|----------|---------|---------------------|
| **`GET /pressure`** | `{ cpu: 23.1, memory: 45.2, queued: 0, rejected: 0, maxConcurrent: 10, ... }` | Poll for health checks, feed CloudWatch custom metrics, pressure-aware routing |
| **`GET /sessions`** | Array of active sessions: `[{ id, running, timeAliveMs, killURL, ... }]` | Orphan detection — compare against our Redis session registry |
| **`GET /config`** | Runtime config: `{ concurrent, queue, timeout, maxCPU, maxMemory, ... }` | Startup validation — confirm Browserless config matches expectations |
| **`GET /metrics`** | Historical session stats (up to 1 week) | Feed dashboards, capacity planning |
| **`GET /performance`** | Performance diagnostics | Debug slow sessions |

**The `/pressure` response (full schema):**

```json
{
  "date": 1680000000000,
  "cpu": 23.1,
  "memory": 45.2,
  "queued": 0,
  "rejected": 0,
  "successful": 142,
  "unhealthy": 0,
  "timedout": 3,
  "totalTime": 84200,
  "meanTime": 593,
  "maxTime": 2100,
  "minTime": 120,
  "maxConcurrent": 10,
  "sessionTimes": [450, 620, 380]
}
```

**Recommended: Add `/pressure` to pool manager health check:**

```python
# pool_manager.py — enhance health_status()

async def health_status(self) -> Dict[str, Any]:
    status = {
        "max_concurrent": self._max_concurrent,
        "active_sessions": len(self._sessions),
        "available_slots": self._max_concurrent - len(self._sessions),
        "sessions": [ ... ],  # existing
    }

    # Add Browserless-side health
    try:
        import aiohttp
        http_url = config.browserless_url.replace("ws://", "http://").replace("wss://", "https://")
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{http_url}/pressure", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                status["browserless_pressure"] = await resp.json()
    except Exception as e:
        status["browserless_pressure"] = {"error": str(e)}

    return status
```

**Recommended: Add `/sessions` for orphan detection:**

The `/sessions` endpoint returns every running browser on the Browserless container. Each session has a `killURL` you can call to terminate it. This means:
- You don't need to build your own orphan heartbeat system (Section 5, Change 4)
- Instead, periodically compare `/sessions` against your Redis session registry
- Any session on Browserless not in your registry = orphan → call its `killURL`

```python
async def _cleanup_orphaned_browsers(self):
    """Kill Browserless sessions not tracked in our registry."""
    try:
        http_url = config.browserless_url.replace("ws://", "http://")
        async with aiohttp.ClientSession() as http:
            # Get all sessions from Browserless
            async with http.get(f"{http_url}/sessions") as resp:
                bl_sessions = await resp.json()

            for bl_session in bl_sessions:
                bl_id = bl_session.get("browserId")
                # Check if we're tracking this session
                if bl_id and not await self._redis.exists(f"browser:session:*"):
                    # Orphan — kill it via Browserless API
                    kill_url = bl_session.get("killURL")
                    if kill_url:
                        await http.get(kill_url)
                        logger.warning(f"Killed orphaned Browserless session {bl_id}")
    except Exception as e:
        logger.error(f"Orphan cleanup failed: {e}")
```

### 6.5 Session reconnection (NOT used — should be for HITL)

```yaml
# ADD THIS for HITL resilience
MAX_RECONNECT_TIME: "60000"     # Keep browser alive for 60s after WebSocket disconnect
```

**How it works:** When a CDP WebSocket disconnects (network blip, ALB timeout, process restart), instead of immediately killing the browser, Browserless keeps it alive for `MAX_RECONNECT_TIME` milliseconds. Your app can reconnect to the same browser session.

**Why this matters for HITL:**
- HITL sessions can last minutes to hours
- ALB WebSocket idle timeout may disconnect the CDP connection
- Without reconnect: browser dies → session lost → human response goes nowhere
- With reconnect: browser stays alive → app reconnects → continues where it left off

**Current behavior:** No `MAX_RECONNECT_TIME` set. Default is 0 — browser dies immediately on disconnect.

### 6.6 Operational webhooks (NOT used — should be)

```yaml
# ADD THESE for alerting
QUEUE_ALERT_URL: "https://your-api/internal/webhooks/browserless/queued"
REJECT_ALERT_URL: "https://your-api/internal/webhooks/browserless/rejected"
TIMEOUT_ALERT_URL: "https://your-api/internal/webhooks/browserless/timeout"
FAILED_HEALTH_URL: "https://your-api/internal/webhooks/browserless/health-failure"
```

**How it works:** Browserless makes a `GET` request to these URLs when events fire. These are aggregate operational alerts (not per-session). Use them to:
- Trigger CloudWatch alarms or PagerDuty alerts
- Auto-scale Browserless instances (reject alert → add capacity)
- Track operational health without polling

### 6.7 OpenTelemetry (NOT used — optional)

```yaml
# ADD THESE if you use OTEL
OTEL_ENABLED: "true"
OTEL_EXPORTER_OTLP_ENDPOINT: "http://your-otel-collector:4317"
```

Browserless can export traces natively to your observability stack. Useful for correlating browser session performance with your application traces.

### 6.8 Recommended production docker-compose (updated)

```yaml
browserless:
  image: ghcr.io/browserless/chromium
  ports:
    - "${BROWSERLESS_PORT:-3000}:3000"
  environment:
    # Auth
    TOKEN: "${BROWSERLESS_TOKEN:-}"

    # Capacity
    CONCURRENT: "${BROWSERLESS_MAX_CONCURRENT:-10}"
    QUEUED: "${BROWSERLESS_QUEUED:-15}"
    TIMEOUT: "${BROWSERLESS_TIMEOUT:-300000}"

    # Health-gated admission (CRITICAL for production)
    HEALTH: "true"
    MAX_CPU_PERCENT: "80"
    MAX_MEMORY_PERCENT: "85"

    # Session reconnection (important for HITL)
    MAX_RECONNECT_TIME: "60000"

    # Fargate shm workaround (only needed on Fargate)
    # CHROME_FLAGS: "--disable-dev-shm-usage"

    # Webhooks (set to your alerting endpoints)
    # QUEUE_ALERT_URL: "https://your-api/internal/webhooks/browserless/queued"
    # REJECT_ALERT_URL: "https://your-api/internal/webhooks/browserless/rejected"
    # TIMEOUT_ALERT_URL: "https://your-api/internal/webhooks/browserless/timeout"
    # FAILED_HEALTH_URL: "https://your-api/internal/webhooks/browserless/health-failure"
  healthcheck:
    test: ["CMD-SHELL", "curl -sf http://localhost:3000/pressure | grep -q '\"cpu\"' || exit 1"]
    interval: 10s
    timeout: 5s
    retries: 5
  restart: unless-stopped
```

**Key changes from current config:**
1. `HEALTH=true` + `MAX_CPU_PERCENT=80` + `MAX_MEMORY_PERCENT=85` — health-gated admission
2. `QUEUED=15` — explicit queue depth
3. `MAX_RECONNECT_TIME=60000` — session reconnection for HITL
4. Healthcheck uses `/pressure` instead of `/active` — confirms Browserless is functional, not just running

---

## 7. Core Challenges with Headless Browsers at Scale <a name="core-challenges"></a>

### Memory

| State | RAM per session |
|-------|----------------|
| Cold Chromium (no page loaded) | 50–150 MB |
| Typical page | 200–500 MB |
| JS-heavy SPA (React dashboard, etc.) | 500 MB–1 GB+ |

**Critical:** Chromium's memory allocator does not return memory to the OS after tabs close. This means:
- Memory-based autoscaling will scale OUT but never scale IN
- Always scale on CPU, not memory

### CPU

- Active session (page loading, JS execution): 10–50% of one CPU core (bursty)
- Idle session (waiting for HITL, page loaded): near-zero CPU
- Rule of thumb: 1 CPU core per 2–3 concurrent **active** sessions

### `/dev/shm` (Shared Memory)

Chrome uses `/dev/shm` for inter-process communication. Docker defaults to 64 MB, Chrome needs ~2 GB.

| Environment | Solution |
|-------------|----------|
| Docker (EC2/bare metal) | `--shm-size=2g` |
| ECS EC2 | `sharedMemorySize: 2147483648` in task definition |
| ECS Fargate | **Not supported.** Must use `--disable-dev-shm-usage` Chrome flag (Browserless env: `CHROME_FLAGS=--disable-dev-shm-usage`) |
| Kubernetes | `emptyDir` volume with `medium: Memory` mounted at `/dev/shm` |

### WebSocket statefulness

CDP WebSocket connections are:
- Long-lived (minutes to hours for HITL workflows)
- Stateful (tied to a specific browser process on a specific container)
- Not resumable on a different host after disconnect

This means **sticky sessions / session affinity is mandatory** when running multiple Browserless instances behind a load balancer.

---

## 8. Scaling Options Compared <a name="scaling-options-compared"></a>

| Dimension | ECS Fargate | ECS EC2 | EKS | Browserless Cloud |
|-----------|-------------|---------|-----|-------------------|
| **Ops burden** | Low | Medium | High | None |
| **`--shm-size`** | No (workaround) | Yes | Yes | N/A |
| **Autoscaling** | CPU target tracking | CPU + custom ASG | HPA + custom metrics | Automatic |
| **Max per task/pod** | 4 vCPU / 30 GB | Instance-dependent | Pod-dependent | N/A |
| **Cost (baseline)** | ~$70–150/mo (2 tasks) | ~$50–120/mo (2 Spot) | ~$170+/mo (control plane + nodes) | Usage-based |
| **Cold start** | 30–60s | 2–5 min (new instance) | 30s (pod), 2–5 min (node) | None |
| **Session affinity** | ALB sticky sessions | ALB sticky sessions | Service + ingress | Built-in |
| **Data sovereignty** | Your VPC | Your VPC | Your VPC | Third-party |
| **Best for** | Getting started, moderate load | High load, need shm | Already on K8s | Zero-ops requirement |

---

## 9. Recommended Architecture: ECS Fargate <a name="recommended-architecture"></a>

### Why Fargate first

- Lowest operational overhead
- Native autoscaling without managing EC2 capacity
- Sufficient for most workloads (4 vCPU / 30 GB per task handles ~10–15 concurrent sessions)
- Clean isolation per task
- Graduation path to EC2 if needed

### Architecture diagram

```
                          ┌─────────────────────────┐
                          │    Application Load      │
   Seer API/Worker ──────►│    Balancer (ALB)        │
   (BROWSERLESS_URL=      │                          │
    ws://alb-dns:3000)    │  - WebSocket enabled     │
                          │  - Sticky sessions (AWSALB│
                          │    cookie, 5 min TTL)     │
                          │  - Health check: /active  │
                          └────────┬────────┬─────────┘
                                   │        │
                    ┌──────────────┘        └──────────────┐
                    ▼                                      ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │  ECS Task 1       │                  │  ECS Task 2       │
         │                   │                  │                   │
         │  Browserless      │                  │  Browserless      │
         │  CONCURRENT=10    │                  │  CONCURRENT=10    │
         │  QUEUED=15        │                  │  QUEUED=15        │
         │  TIMEOUT=300000   │                  │  TIMEOUT=300000   │
         │  HEALTH=true      │                  │  HEALTH=true      │
         │  MAX_CPU_PERCENT  │                  │  MAX_CPU_PERCENT  │
         │  =85              │                  │  =85              │
         │                   │                  │                   │
         │  2 vCPU / 4 GB    │                  │  2 vCPU / 4 GB    │
         └──────────────────┘                  └──────────────────┘
                              ... Task N ...

   ECS Service Autoscaling:
     - Target: ECSServiceAverageCPUUtilization = 65%
     - Min: 2 tasks (HA)
     - Max: 10 tasks (adjust to expected peak)
     - Scale-in cooldown: 300s (let sessions drain)
     - Scale-out cooldown: 60s (react quickly to load)
```

### ECS Task Definition (key settings)

```json
{
  "family": "browserless",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "browserless",
      "image": "ghcr.io/browserless/chromium",
      "essential": true,
      "portMappings": [
        { "containerPort": 3000, "protocol": "tcp" }
      ],
      "environment": [
        { "name": "TOKEN", "value": "your-secret-token" },
        { "name": "CONCURRENT", "value": "10" },
        { "name": "QUEUED", "value": "15" },
        { "name": "TIMEOUT", "value": "300000" },
        { "name": "HEALTH", "value": "true" },
        { "name": "MAX_CPU_PERCENT", "value": "85" },
        { "name": "MAX_MEMORY_PERCENT", "value": "90" },
        { "name": "CHROME_FLAGS", "value": "--disable-dev-shm-usage" }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -sf http://localhost:3000/active || exit 1"],
        "interval": 10,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 30
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/browserless",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### ALB Configuration

```
Listener: TCP 3000 (or 443 with TLS termination)
Target Group:
  - Protocol: HTTP
  - Port: 3000
  - Health check path: /active
  - Health check interval: 10s
  - Healthy threshold: 2
  - Unhealthy threshold: 3
  - Deregistration delay: 120s  (let active sessions finish)
  - Stickiness: enabled
    - Type: lb_cookie (AWSALB)
    - Duration: 300s (match session timeout)
```

### Autoscaling Policy

```
Target Tracking:
  Metric: ECSServiceAverageCPUUtilization
  Target: 65
  Scale-out cooldown: 60s
  Scale-in cooldown: 300s

Step Scaling (optional secondary):
  Metric: ALBRequestCountPerTarget
  Threshold: If > 8 requests/target for 2 min → add 2 tasks
  Threshold: If < 2 requests/target for 5 min → remove 1 task
```

### Fargate `/dev/shm` workaround

Fargate does not support `--shm-size`. The fix:

```
CHROME_FLAGS=--disable-dev-shm-usage
```

This tells Chrome to use `/tmp` instead of `/dev/shm`. Performance impact is minimal (single-digit percent). This is the official workaround recommended by both Browserless and AWS.

---

## 10. Alternative: ECS EC2 <a name="alternative-ecs-ec2"></a>

### When to graduate from Fargate

- You observe Chrome crashes or instability due to `/dev/shm` workaround
- You need > 4 vCPU or > 30 GB per task
- You want to use Spot instances for cost savings (60–70% cheaper)
- You need GPU instances for specific rendering workloads

### Architecture delta from Fargate

Everything stays the same except:
1. Replace `FARGATE` launch type with `EC2`
2. Add an EC2 Auto Scaling Group as the capacity provider
3. Add `sharedMemorySize: 2147483648` to the container definition (removes need for `--disable-dev-shm-usage`)
4. Instance type: `c6i.xlarge` (4 vCPU, 8 GB) — compute-optimized for browser workloads

### Spot instance strategy

```
Capacity Provider:
  - Primary: Spot (c6i.xlarge, c6a.xlarge, c5.xlarge)
    Weight: 80%
    Managed termination protection: enabled
  - Fallback: On-Demand (c6i.xlarge)
    Weight: 20%
```

Spot interruption handling: ECS will drain tasks before the instance is reclaimed (2-minute warning). Set `stopTimeout: 120` on the container to allow sessions to complete.

---

## 11. Alternative: EKS with Custom Metrics <a name="alternative-eks"></a>

### When to use EKS

- You're already running Kubernetes for other services
- You need fine-grained custom metric scaling (e.g., scale on active browser sessions, not just CPU)
- You want pod-level resource isolation with taints/tolerations

### Key configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: browserless
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: browserless
        image: ghcr.io/browserless/chromium
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: CONCURRENT
          value: "10"
        - name: QUEUED
          value: "15"
        - name: HEALTH
          value: "true"
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: browserless-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: browserless
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 60
```

### Custom metric scaling (advanced)

Browserless exposes `/pressure` endpoint returning real-time load. You can use a Prometheus adapter to expose this as a custom metric for HPA:

```
/pressure → { "date": "...", "reason": "ok", "isAvailable": true }
```

Scale on `isAvailable: false` ratio across pods.

---

## 12. Pool Manager Refactoring for Multi-Instance <a name="pool-manager-refactoring"></a>

The current `BrowserPoolManager` works fine against a single Browserless instance or an ALB (the ALB handles routing). However, several changes are needed for production resilience:

### Change 1: Distributed semaphore (if running multiple API/worker replicas)

**Problem:** Each API/worker process has its own `asyncio.Semaphore(5)`. With 3 replicas, you could send 15 concurrent sessions to Browserless instances that can only handle 10 × N.

**Options:**

| Approach | Complexity | Recommendation |
|----------|------------|----------------|
| **Do nothing** — rely on Browserless `QUEUED` to absorb overflow | Low | Good starting point. Browserless returns 429 when queue is full; catch and retry with backoff. |
| **Redis-based distributed semaphore** | Medium | Use Valkey (already in your stack) with `SETNX`-based lock or Redlock. Limit total sessions across all replicas. |
| **Pressure-aware routing** | High | Query `/pressure` on Browserless before creating session. Route to least-loaded instance. Most robust but most complex. |

**Recommendation for Phase 1:** Rely on Browserless queuing + add retry logic. Simplest, handles 90% of cases.

```python
# Suggested addition to pool_manager.py create_session():
# After BrowserSession.start() — catch connection refused / 429 and retry

MAX_CONNECT_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds

for attempt in range(MAX_CONNECT_RETRIES):
    try:
        await browser_session.start()
        break
    except Exception as e:
        if attempt == MAX_CONNECT_RETRIES - 1:
            raise
        wait = RETRY_BACKOFF_BASE * (attempt + 1)
        logger.warning(f"Browserless connect attempt {attempt+1} failed: {e}, retrying in {wait}s")
        await asyncio.sleep(wait)
```

### Change 2: Graceful handling of WebSocket disconnects

**Problem:** If a Browserless container is killed (scale-in, deployment, crash), active CDP WebSocket connections drop. The current code will raise an exception but won't gracefully recover.

**Recommendation:** Add a `try/except` around agent execution that catches WebSocket disconnects and marks the session as failed rather than crashing the entire workflow run.

### Change 3: Health check integration

**Problem:** `BrowserPoolManager.health_status()` only reports local pool state. It doesn't reflect Browserless-side health.

**Recommendation:** Add a Browserless health probe to `health_status()`:

```python
async def health_status(self) -> Dict[str, Any]:
    local_status = { ... }  # existing logic

    # Add Browserless-side health
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{config.browserless_url.replace('ws://', 'http://')}/pressure") as resp:
                browserless_pressure = await resp.json()
    except Exception:
        browserless_pressure = {"error": "unreachable"}

    local_status["browserless"] = browserless_pressure
    return local_status
```

### Change 4: Session migration is NOT feasible

It's important to document this clearly: **you cannot migrate a live CDP session between Browserless containers.** If a container goes down, sessions on it are lost. The mitigation strategies are:

1. **Cookie persistence** (you already do this) — session state (auth cookies) is exported before release and stored in DB. A new session can be created on a different container with the same cookies.
2. **Graceful drain** — use ALB deregistration delay (120s) + Browserless `QUEUED` so in-flight sessions complete before the container is removed.
3. **Workflow retry** — if a browser node fails due to WebSocket disconnect, the workflow engine can retry the node (creates a new session on a healthy container).

---

## 13. HITL Implications <a name="hitl-implications"></a>

Human-in-the-loop is the trickiest part to scale because of its dual statefulness:

### Current HITL flow

```
1. Browser agent calls ask_human()
2. hitl_bridge.register_hitl_wait(run_id, request_id) → creates asyncio.Future
3. managed_session.hitl_paused = True (prevents reaper from killing it)
4. Agent awaits the Future (could be minutes/hours)
5. Human responds via API → hitl_bridge.resolve_hitl_wait() → Future resolves
6. Agent continues on the same CDP session
```

### HITL scaling challenges

| Issue | Impact | Mitigation |
|-------|--------|------------|
| `hitl_bridge._pending_futures` is in-memory | If the API process that registered the Future dies, the HITL response is lost | Move HITL state to Redis/Valkey with pub/sub for resolution |
| CDP session is tied to one Browserless container | Container can't be killed during HITL wait (even for scale-in) | Protect HITL-paused sessions from scale-in via task protection / longer drain time |
| HITL sessions hold semaphore slots for long periods | Reduces throughput for workflow sessions | Consider separate pools: one for workflow (short-lived), one for interactive/HITL (long-lived) |

### Recommended HITL architecture for production

```
Phase 1 (Quick win):
  - Keep hitl_bridge as-is
  - Ensure the API process handling HITL is the same one that created the session
  - Set ALB stickiness to cover max HITL timeout (1800s if that's your max)
  - Set ECS deregistration delay to match

Phase 2 (Resilient):
  - Move HITL wait state to Redis:
    - Key: hitl:{run_id}:{request_id} → {status: "waiting", container_hint: "task-arn"}
    - When human responds: write response to Redis, publish on channel
    - Agent process subscribes to channel, resolves its local Future
  - This decouples the API instance receiving the response from the one awaiting it
```

---

## 14. Configuration & Environment Variables <a name="configuration"></a>

### Browserless container environment (production)

```env
# Core
TOKEN=<strong-random-token>      # Auth for all connections
CONCURRENT=10                     # Sessions per container
QUEUED=15                         # Queue buffer (1.5x concurrent)
TIMEOUT=300000                    # 5 min default timeout (ms)

# Health & limits
HEALTH=true                       # Enable health monitoring
MAX_CPU_PERCENT=85                # Reject new sessions above this CPU %
MAX_MEMORY_PERCENT=90             # Reject new sessions above this memory %

# Fargate-specific
CHROME_FLAGS=--disable-dev-shm-usage

# Optional: stealth for anti-bot sites
# DEFAULT_STEALTH=true
```

### Seer application environment (production)

```env
# Point to ALB, not individual container
BROWSERLESS_URL=ws://browserless-alb.internal:3000
BROWSERLESS_TOKEN=<same-strong-token>

# Match CONCURRENT × number_of_tasks for global limit
# E.g., 2 tasks × 10 concurrent = 20
BROWSER_POOL_MAX_CONCURRENT=20

# Longer timeout for HITL
BROWSER_POOL_DEFAULT_TIMEOUT_SECONDS=300
BROWSER_INTERACTIVE_TIMEOUT_SECONDS=1800

# Store in AWS Parameter Store:
# /prod/browserless_url
# /prod/browserless_token
# /prod/browser_pool_max_concurrent
```

---

## 15. Cost Estimation <a name="cost-estimation"></a>

### ECS Fargate (us-east-1 pricing)

| Component | Spec | Monthly Cost |
|-----------|------|-------------|
| 2 tasks (baseline) | 2 vCPU / 4 GB each, 24/7 | ~$140 |
| 4 tasks (moderate load) | 2 vCPU / 4 GB each, 24/7 | ~$280 |
| ALB | Fixed + LCU | ~$25–40 |
| CloudWatch Logs | ~10 GB/mo | ~$5 |
| **Total (2 tasks)** | | **~$170/mo** |
| **Total (4 tasks)** | | **~$325/mo** |

### ECS EC2 with Spot (us-east-1 pricing)

| Component | Spec | Monthly Cost |
|-----------|------|-------------|
| 2 × c6i.xlarge Spot | 4 vCPU / 8 GB each | ~$50 |
| 2 × c6i.xlarge On-Demand (fallback) | 4 vCPU / 8 GB each | ~$245 |
| ALB + Logs | | ~$30–45 |
| **Total (Spot, 2 instances)** | | **~$85/mo** |
| **Total (On-Demand, 2 instances)** | | **~$290/mo** |

### EKS

| Component | Spec | Monthly Cost |
|-----------|------|-------------|
| EKS control plane | | $73 |
| 2 × c6i.xlarge nodes | On-Demand | $245 |
| **Total** | | **~$320/mo** |

---

## 16. Migration Path <a name="migration-path"></a>

### Phase 1: Browserless infrastructure (Week 1) — NO code changes

**Goal:** Get Browserless running on ECS Fargate behind an ALB with autoscaling.

**App changes required: ZERO.** Just change `BROWSERLESS_URL` to point at the ALB. This works because your pool manager already connects to a single URL — the ALB transparently routes to healthy containers with sticky sessions.

```
1. Create ECS cluster (Fargate)
2. Create task definition (see Section 8)
3. Create ECS service (min=2, max=10)
4. Create ALB with WebSocket support + sticky sessions
5. Create target tracking autoscaling policy (CPU 65%)
6. Update Seer config: BROWSERLESS_URL=ws://alb-dns:3000
7. Deploy and test
```

**Limitation:** This gives you Browserless HA and autoscaling, but your app layer is still a single-process bottleneck. If you only run 1 API replica, this is fine.

### Phase 2: Connection resilience (Week 2) — Small code changes

**Goal:** Handle Browserless failures gracefully instead of crashing.

```
1. Add connection retry with backoff to pool_manager.py (Section 5, Change 5)
2. Add WebSocket disconnect handling in browser_service.py
3. Add Browserless /pressure health probe to health_status()
4. Add CloudWatch alarms: Browserless CPU > 80%, unhealthy targets, 429 rate
```

**Files changed:** `pool_manager.py`, `browser_service.py` (minor additions)

### Phase 3: Stateless app layer (Week 3–4) — Core refactor

**Goal:** Make the Python process disposable. Multiple API/worker replicas can coexist. See Section 5 for full design.

```
1. Add Redis session registry to pool_manager.py (Section 5, Change 1)
2. Add distributed concurrency limiting via Redis counter (Section 5, Change 2)
3. Add replica heartbeat + orphan cleanup (Section 5, Change 4)
4. Add config: browser_pool_max_concurrent_per_replica, browser_pool_max_concurrent_global
5. Test with 2+ API replicas behind a load balancer
```

**Files changed:** `pool_manager.py` (major), `config.py` (add 2 fields)

### Phase 4: Distributed HITL (Week 4–5) — Medium refactor

**Goal:** HITL survives process restarts. Any API replica can receive the human response.

```
1. Refactor hitl_bridge.py to use Redis + pub/sub (Section 5, Change 3)
2. Update browser_service.py to use new HITLBridge class
3. Add pub/sub subscriber startup to application lifecycle
4. Test: start HITL on replica A, respond via replica B
```

**Files changed:** `hitl_bridge.py` (rewrite), `browser_service.py` (minor)

### Phase 5: Optimization (Month 2+)

**Goal:** Cost and performance tuning.

```
1. Analyze CloudWatch metrics to right-size Fargate tasks
2. If shm issues: graduate to ECS EC2 (Section 9)
3. If custom metrics needed: graduate to EKS (Section 10)
4. Consider Spot instances for non-HITL workloads
5. Implement pressure-aware routing if running many Browserless instances
6. Consider separate browser pools for workflow (short) vs interactive/HITL (long)
```

---

## Decision Matrix: Quick Reference

| Question | Answer |
|----------|--------|
| **Can I ditch CDP WebSockets?** | No. CDP is the only protocol for browser automation. All tools (Playwright, Puppeteer, browser-use) use it. There is no REST alternative for interactive sessions. |
| **So what DO I change?** | Your **application layer** — move `_sessions`, semaphore, and HITL futures from in-memory to Redis/Valkey. This makes your Python processes stateless and horizontally scalable. |
| **Can I scale without ANY code changes?** | Yes for Phase 1 — point `BROWSERLESS_URL` at an ALB. But you're limited to 1 API replica unless you do Phase 3. |
| **Do I need sticky sessions?** | Yes, always. CDP WebSocket connections are stateful at the Browserless level. |
| **What metric should I autoscale on?** | CPU (65% target). Never memory — Chrome doesn't release it. |
| **Can I move a session between containers?** | No. Sessions are bound to the container that created them. Cookie export/import is your checkpoint mechanism. |
| **What about HITL during scale-in?** | Use ALB deregistration delay (120s+) and protect HITL-active tasks from scale-in. |
| **Fargate vs EC2?** | Start with Fargate. Graduate to EC2 if you need `--shm-size` or Spot pricing. |
| **How many sessions per container?** | ~8–10 at 2 vCPU/4 GB. Budget ~500 MB + 0.3 CPU per concurrent session. |
