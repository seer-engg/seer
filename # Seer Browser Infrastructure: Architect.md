# Seer Browser Infrastructure: Architecture Document

## The Big Picture

You're building a browser infrastructure layer that sits between your workflow engine (FastAPI workers + LangGraph) and the actual browser automation libraries (Browser Use via Playwright). This layer handles everything : browser lifecycle, session persistence, user authentication flows with live streaming, and historical replay.

The architecture breaks down into **five core subsystems** that work together.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SEER FRONTEND (TS)                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │ Workflow      │  │ Live Browser     │  │ Session Replay        │ │
│  │ Builder UI    │  │ Viewer (canvas + │  │ Player (rrweb-player) │ │
│  │              │  │ WebSocket)       │  │                       │ │
│  └──────┬───────┘  └────────┬─────────┘  └───────────┬───────────┘ │
└─────────┼──────────────────┼─────────────────────────┼─────────────┘
          │                  │                         │
          │ REST API         │ WebSocket               │ REST API
          │                  │ (frames + input)        │ (fetch events)
          ▼                  ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SEER BACKEND (FastAPI)                          │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  Browser Service API                           │ │
│  │  POST /browser/sessions      - create session                 │ │
│  │  DELETE /browser/sessions/:id - release session               │ │
│  │  GET /browser/sessions/:id/context - get cookies/storage      │ │
│  │  WS /browser/sessions/:id/stream - live screencast frames     │ │
│  │  WS /browser/sessions/:id/input  - receive user input events  │ │
│  │  GET /browser/sessions/:id/replay - get rrweb recording       │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │                 Browser Pool Manager                           │ │
│  │  - Manages Playwright browser instances                       │ │
│  │  - Concurrency control (semaphore-based)                      │ │
│  │  - Session lifecycle (create, track, cleanup, timeout)        │ │
│  │  - Health monitoring & auto-recovery                          │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │               Session Context Manager                          │ │
│  │  - Save/restore cookies per user per integration              │ │
│  │  - Save/restore localStorage/sessionStorage                   │ │
│  │  - Encrypt sensitive session data at rest                     │ │
│  │  - Context expiry & refresh logic                             │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │              Live Streaming Service                            │ │
│  │  - CDP Page.startScreencast → JPEG frames                    │ │
│  │  - Frame relay over WebSocket to frontend                     │ │
│  │  - Input event proxy (mouse/keyboard) from frontend → CDP    │ │
│  │  - Session-scoped: one stream per active viewer               │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │              Session Recording Service                         │ │
│  │  - Injects rrweb recorder script into pages                   │ │
│  │  - Collects DOM mutation events via CDP                       │ │
│  │  - Stores events in DB (compressed JSON)                      │ │
│  │  - Serves events for rrweb-player replay                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │
          │  Playwright CDP
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CHROMIUM INSTANCE(S)                             │
│                    (Running on same EC2)                            │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Session 1 │  │ Session 2 │  │ Session 3 │  │ Session N │          │
│  │ (user A   │  │ (user B   │  │ (user A   │  │           │          │
│  │  workflow) │  │  login)   │  │  workflow) │  │           │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Subsystem 1: Browser Pool Manager

This is the core — it manages the lifecycle of Playwright browser instances on your EC2.

### Responsibilities

The pool manager's job is to create browser contexts on demand, enforce concurrency limits so your EC2 doesn't run out of memory, track active sessions with metadata (which user, which integration, when created), enforce timeouts to kill abandoned sessions, and recover gracefully when a browser crashes.

### Key Design Decisions

**Single persistent browser, multiple contexts.** Don't launch a new Chromium process per session — that's expensive (~100-150MB per process). Instead, launch one Playwright browser instance at startup and create isolated `BrowserContext` objects per session. Each context has its own cookies, storage, and cache, so they're fully isolated. You can comfortably run 10-20 concurrent contexts on a 4GB EC2 instance.

**Concurrency via asyncio.Semaphore.** Set a max based on your EC2's memory. Each `create_session()` acquires the semaphore; `release_session()` releases it. If the pool is full, requests queue up rather than crashing.

**Timeout enforcement.** Every session gets a TTL (e.g., 30 minutes for interactive login, 5 minutes for automated workflow steps). A background task periodically sweeps and kills expired sessions. This prevents resource leaks from abandoned sessions.

```python
# Conceptual structure — not production code, but shows the shape

class BrowserPoolManager:
    def __init__(self, max_concurrent: int = 10, default_timeout: int = 300):
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.sessions: dict[str, BrowserSession] = {}
        self.browser: Browser | None = None

    async def initialize(self):
        """Launch a single persistent Playwright browser on startup."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=True,  # headless for automated workflows
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",  # important for Docker/EC2
                "--disable-gpu",
            ]
        )
        # Start the session reaper background task
        asyncio.create_task(self._session_reaper())

    async def create_session(
        self,
        user_id: str,
        integration_id: str,
        session_type: str,  # "interactive" | "automation"
        context_data: dict | None = None,  # saved cookies/storage to inject
        proxy: str | None = None,
        timeout: int | None = None,
    ) -> BrowserSession:
        """Create a new isolated browser session."""
        await self.semaphore.acquire()

        context_options = {
            "viewport": {"width": 1280, "height": 800},
            "user_agent": get_realistic_user_agent(),
        }
        if proxy:
            context_options["proxy"] = {"server": proxy}

        context = await self.browser.new_context(**context_options)

        # Inject saved cookies if resuming a session
        if context_data and context_data.get("cookies"):
            await context.add_cookies(context_data["cookies"])

        # Inject localStorage via CDP (Playwright doesn't natively support this)
        if context_data and context_data.get("local_storage"):
            # Navigate to the domain first, then inject
            # (localStorage is domain-scoped)
            pass

        page = await context.new_page()

        session = BrowserSession(
            id=str(uuid4()),
            user_id=user_id,
            integration_id=integration_id,
            session_type=session_type,
            context=context,
            page=page,
            created_at=datetime.utcnow(),
            timeout=timeout or self.default_timeout,
        )
        self.sessions[session.id] = session
        return session

    async def release_session(self, session_id: str) -> dict | None:
        """Close session, extract context, release resources."""
        session = self.sessions.pop(session_id, None)
        if not session:
            return None

        # Extract cookies and storage before closing
        context_data = await self._extract_context(session)

        await session.context.close()
        self.semaphore.release()
        return context_data

    async def _extract_context(self, session: BrowserSession) -> dict:
        """Pull cookies and localStorage from the session."""
        cookies = await session.context.cookies()

        # Extract localStorage via CDP
        cdp = await session.page.context.new_cdp_session(session.page)
        local_storage = {}
        try:
            # Get all frames' localStorage
            result = await cdp.send("DOMStorage.getDOMStorageItems", {
                "storageId": {"securityOrigin": session.page.url, "isLocalStorage": True}
            })
            local_storage = {item[0]: item[1] for item in result.get("entries", [])}
        except Exception:
            pass  # page may have navigated away

        return {
            "cookies": cookies,
            "local_storage": local_storage,
            "extracted_at": datetime.utcnow().isoformat(),
        }

    async def _session_reaper(self):
        """Background task: kill sessions that exceed their timeout."""
        while True:
            await asyncio.sleep(30)
            now = datetime.utcnow()
            expired = [
                sid for sid, s in self.sessions.items()
                if (now - s.created_at).total_seconds() > s.timeout
            ]
            for sid in expired:
                logger.warning(f"Reaping expired session {sid}")
                await self.release_session(sid)
```

### Stealth Integration

For basic anti-detection, install `playwright-stealth` (the Python port of `puppeteer-extra-plugin-stealth`). Apply it to each new page. This patches `navigator.webdriver`, WebGL fingerprinting, Chrome runtime checks, and other common bot signals. This is the same level of stealth that Steel's self-hosted version provides.

```python
from playwright_stealth import stealth_async

page = await context.new_page()
await stealth_async(page)
```

---

## Subsystem 2: Session Context Manager

This persists user authentication state across workflow runs. When a user logs into Gmail through your interactive browser, you save those cookies. Next time a workflow runs that needs Gmail access, you inject those cookies back.

### Storage Design

Store context in your existing database (Supabase/Postgres) with encryption at rest. The schema looks like this:

```sql
CREATE TABLE browser_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    integration_id VARCHAR(255) NOT NULL,  -- e.g., "gmail", "github", "shopify"
    cookies BYTEA NOT NULL,                 -- AES-encrypted JSON
    local_storage BYTEA,                    -- AES-encrypted JSON
    session_storage BYTEA,                  -- AES-encrypted JSON
    domain VARCHAR(255) NOT NULL,           -- primary domain for this context
    expires_at TIMESTAMPTZ,                 -- when cookies expire (earliest expiry)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, integration_id)         -- one context per user per integration
);
```

### Key Design Points

**Always update after workflow runs.** Websites refresh tokens and update cookies during sessions. After every workflow execution that uses a browser context, extract the updated cookies and save them back. If you only save once during login and never update, sessions will expire and users will need to re-authenticate frequently.

**Encrypt at rest.** You're storing user session cookies — these are essentially authentication tokens. Use Fernet (symmetric encryption) with a key from your environment variables. Never store raw cookies in the database.

**Domain-scoped context.** localStorage is scoped to a domain. When restoring it, you need to navigate to that domain first before injecting. Your context manager should track which domain the context belongs to and handle the navigation automatically.

**Context validation.** Before using a saved context, do a quick validation — navigate to the target site, check if the session is still valid (look for login prompts or auth redirects). If invalid, flag the context as stale and notify the user to re-authenticate.

```python
class SessionContextManager:
    def __init__(self, db, encryption_key: bytes):
        self.db = db
        self.fernet = Fernet(encryption_key)

    async def save_context(
        self, user_id: str, integration_id: str, context_data: dict
    ):
        """Encrypt and persist browser context."""
        encrypted_cookies = self.fernet.encrypt(
            json.dumps(context_data["cookies"]).encode()
        )
        encrypted_storage = self.fernet.encrypt(
            json.dumps(context_data.get("local_storage", {})).encode()
        )

        # Calculate earliest cookie expiry for proactive refresh
        expires_at = self._earliest_cookie_expiry(context_data["cookies"])

        await self.db.execute("""
            INSERT INTO browser_contexts (user_id, integration_id, cookies,
                local_storage, domain, expires_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (user_id, integration_id)
            DO UPDATE SET cookies = $3, local_storage = $4,
                expires_at = $6, updated_at = NOW()
        """, user_id, integration_id, encrypted_cookies,
            encrypted_storage, context_data.get("domain"), expires_at)

    async def load_context(self, user_id: str, integration_id: str) -> dict | None:
        """Decrypt and return saved browser context."""
        row = await self.db.fetchrow("""
            SELECT cookies, local_storage, domain, expires_at
            FROM browser_contexts
            WHERE user_id = $1 AND integration_id = $2
        """, user_id, integration_id)

        if not row:
            return None

        return {
            "cookies": json.loads(self.fernet.decrypt(row["cookies"])),
            "local_storage": json.loads(self.fernet.decrypt(row["local_storage"])),
            "domain": row["domain"],
            "expires_at": row["expires_at"],
        }
```

---

## Subsystem 3: Live Streaming Service

This is what lets users see and interact with the browser in real time — used during the "connect your account" flow where they need to log in manually.

### How It Works

The streaming pipeline uses CDP's built-in screencast capability. When a user initiates an interactive session, your backend starts capturing JPEG frames from Chrome and relays them to the frontend over a WebSocket. User input events (mouse clicks, keyboard presses, scrolling) flow in the reverse direction.

```
Frontend (canvas)          Backend (FastAPI)           Chrome (CDP)
     │                          │                          │
     │  WS connect              │                          │
     ├─────────────────────────►│                          │
     │                          │  Page.startScreencast    │
     │                          ├─────────────────────────►│
     │                          │                          │
     │                          │  screencastFrame event   │
     │                          │◄─────────────────────────┤
     │   binary JPEG frame      │                          │
     │◄─────────────────────────┤                          │
     │                          │  ack frame               │
     │                          ├─────────────────────────►│
     │                          │                          │
     │  user clicks (x,y)       │                          │
     ├─────────────────────────►│                          │
     │                          │  Input.dispatchMouseEvent│
     │                          ├─────────────────────────►│
     │                          │                          │
     │  user types "hello"      │                          │
     ├─────────────────────────►│                          │
     │                          │  Input.dispatchKeyEvent  │
     │                          ├─────────────────────────►│
```

### Backend Implementation

```python
# WebSocket endpoint for live streaming

@router.websocket("/browser/sessions/{session_id}/stream")
async def session_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = pool_manager.get_session(session_id)
    if not session or session.session_type != "interactive":
        await websocket.close(code=4004)
        return

    cdp = await session.page.context.new_cdp_session(session.page)

    # Frame queue — CDP fires events, we relay them
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=3)

    def on_screencast_frame(params):
        """CDP callback: new frame available."""
        frame_data = base64.b64decode(params["data"])
        session_id_internal = params["sessionId"]

        # Ack immediately so CDP sends the next frame
        asyncio.create_task(
            cdp.send("Page.screencastFrameAck", {"sessionId": session_id_internal})
        )

        # Drop frames if queue is full (backpressure)
        try:
            frame_queue.put_nowait(frame_data)
        except asyncio.QueueFull:
            pass  # skip frame, frontend is too slow

    cdp.on("Page.screencastFrame", on_screencast_frame)

    # Start screencast — quality and resolution are configurable
    await cdp.send("Page.startScreencast", {
        "format": "jpeg",
        "quality": 60,          # balance quality vs bandwidth
        "maxWidth": 1280,
        "maxHeight": 800,
        "everyNthFrame": 2,     # skip every other frame for performance
    })

    try:
        # Two concurrent tasks: send frames out, receive input in
        async def send_frames():
            while True:
                frame = await frame_queue.get()
                await websocket.send_bytes(frame)

        async def receive_input():
            while True:
                data = await websocket.receive_json()
                await _dispatch_input(cdp, data, session.page)

        await asyncio.gather(send_frames(), receive_input())

    except WebSocketDisconnect:
        pass
    finally:
        await cdp.send("Page.stopScreencast")


async def _dispatch_input(cdp, event: dict, page):
    """Route frontend input events to Chrome via CDP."""
    event_type = event.get("type")

    if event_type == "mousedown":
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": event["x"],
            "y": event["y"],
            "button": "left",
            "clickCount": 1,
        })
    elif event_type == "mouseup":
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": event["x"],
            "y": event["y"],
            "button": "left",
        })
    elif event_type == "mousemove":
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": event["x"],
            "y": event["y"],
        })
    elif event_type == "keydown":
        await cdp.send("Input.dispatchKeyEvent", {
            "type": "keyDown",
            "key": event["key"],
            "text": event.get("text", ""),
            "code": event.get("code", ""),
        })
    elif event_type == "keyup":
        await cdp.send("Input.dispatchKeyEvent", {
            "type": "keyUp",
            "key": event["key"],
            "code": event.get("code", ""),
        })
    elif event_type == "scroll":
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseWheel",
            "x": event["x"],
            "y": event["y"],
            "deltaX": event.get("deltaX", 0),
            "deltaY": event.get("deltaY", 0),
        })
```

### Frontend Implementation

The frontend renders frames on a canvas element and captures user input events, sending them back over the same WebSocket.

```tsx
// BrowserViewer.tsx — core concept

const BrowserViewer: React.FC<{ sessionId: string }> = ({ sessionId }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        const ws = new WebSocket(
            `${WS_BASE_URL}/browser/sessions/${sessionId}/stream`
        );
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        ws.onmessage = (event) => {
            // Received a JPEG frame — render it on canvas
            const blob = new Blob([event.data], { type: "image/jpeg" });
            const img = new Image();
            img.onload = () => {
                const ctx = canvasRef.current?.getContext("2d");
                if (ctx) {
                    ctx.drawImage(img, 0, 0, ctx.canvas.width, ctx.canvas.height);
                }
                URL.revokeObjectURL(img.src);
            };
            img.src = URL.createObjectURL(blob);
        };

        return () => ws.close();
    }, [sessionId]);

    // Calculate coordinates relative to the browser viewport
    const getScaledCoords = (e: React.MouseEvent) => {
        const canvas = canvasRef.current!;
        const rect = canvas.getBoundingClientRect();
        const scaleX = 1280 / rect.width;   // match CDP viewport
        const scaleY = 800 / rect.height;
        return {
            x: Math.round((e.clientX - rect.left) * scaleX),
            y: Math.round((e.clientY - rect.top) * scaleY),
        };
    };

    const sendInput = (data: object) => {
        wsRef.current?.send(JSON.stringify(data));
    };

    return (
        <canvas
            ref={canvasRef}
            width={1280}
            height={800}
            style={{ width: "100%", maxWidth: 1280, cursor: "default" }}
            onMouseDown={(e) => sendInput({ type: "mousedown", ...getScaledCoords(e) })}
            onMouseUp={(e) => sendInput({ type: "mouseup", ...getScaledCoords(e) })}
            onMouseMove={(e) => sendInput({ type: "mousemove", ...getScaledCoords(e) })}
            onKeyDown={(e) => {
                e.preventDefault();
                sendInput({ type: "keydown", key: e.key, text: e.key.length === 1 ? e.key : "", code: e.code });
            }}
            onKeyUp={(e) => sendInput({ type: "keyup", key: e.key, code: e.code })}
            onWheel={(e) => {
                const coords = getScaledCoords(e as any);
                sendInput({ type: "scroll", ...coords, deltaX: e.deltaX, deltaY: e.deltaY });
            }}
            tabIndex={0}  // makes canvas focusable for keyboard events
        />
    );
};
```

### Performance Considerations

CDP screencast at quality 60 and `everyNthFrame: 2` gives you roughly 10-15fps, which feels adequate for a login flow (not gaming, just clicking through forms). Each frame is roughly 30-80KB depending on page complexity. For a typical interactive session (1-3 minutes), you're looking at maybe 50-100MB of data transferred — very manageable. If you need smoother streaming later, you can upgrade to a noVNC-based approach or even implement WebRTC (more complex but gives you 25+ fps with much lower latency).

---

## Subsystem 4: Session Recording Service

This records what happens during browser sessions so users can replay them later — useful for debugging workflows and auditing what the automation did.

### Approach: rrweb Injection

rrweb is an open-source library that records DOM mutations, mouse movements, scrolling, and input events. You inject it into every page the browser visits, collect the events on the backend, and play them back using `rrweb-player`.

```python
class SessionRecordingService:
    RRWEB_SCRIPT_URL = "https://cdn.jsdelivr.net/npm/rrweb@latest/dist/rrweb-all.min.js"

    async def start_recording(self, session: BrowserSession):
        """Inject rrweb into the page and start recording."""
        session.recording_events = []

        # Expose a function the page can call to send events back
        await session.page.expose_function(
            "__rrwebEvent",
            lambda event_json: session.recording_events.append(
                json.loads(event_json)
            )
        )

        # Inject rrweb and start recording
        await session.page.add_init_script(f"""
            (async () => {{
                const script = document.createElement('script');
                script.src = '{self.RRWEB_SCRIPT_URL}';
                script.onload = () => {{
                    rrweb.record({{
                        emit(event) {{
                            window.__rrwebEvent(JSON.stringify(event));
                        }},
                        sampling: {{
                            mousemove: true,
                            mouseInteraction: true,
                            scroll: 150,     // throttle scroll events
                            input: 'last',   // only record final input value
                        }},
                        blockClass: 'rr-block',  // mask sensitive elements
                        maskInputOptions: {{
                            password: true,   // never record password values
                        }},
                    }});
                }};
                document.head.appendChild(script);
            }})();
        """)

    async def stop_recording(self, session: BrowserSession) -> list[dict]:
        """Stop recording and return all captured events."""
        return session.recording_events

    async def save_recording(
        self, session_id: str, user_id: str, events: list[dict]
    ):
        """Compress and store recording in the database."""
        compressed = gzip.compress(json.dumps(events).encode())
        await self.db.execute("""
            INSERT INTO session_recordings
                (session_id, user_id, events_compressed, event_count, created_at)
            VALUES ($1, $2, $3, $4, NOW())
        """, session_id, user_id, compressed, len(events))

    async def get_recording(self, session_id: str) -> list[dict]:
        """Retrieve and decompress recording events."""
        row = await self.db.fetchrow("""
            SELECT events_compressed FROM session_recordings
            WHERE session_id = $1
        """, session_id)
        if not row:
            return []
        return json.loads(gzip.decompress(row["events_compressed"]))
```

### Frontend Replay

Use the `rrweb-player` npm package to replay recordings in your UI.

```tsx
// SessionReplay.tsx

import rrwebPlayer from "rrweb-player";
import "rrweb-player/dist/style.css";

const SessionReplay: React.FC<{ sessionId: string }> = ({ sessionId }) => {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        async function loadReplay() {
            const res = await fetch(`/api/browser/sessions/${sessionId}/replay`);
            const events = await res.json();

            if (containerRef.current && events.length > 0) {
                new rrwebPlayer({
                    target: containerRef.current,
                    props: {
                        events,
                        width: 1280,
                        height: 800,
                        autoPlay: false,
                        showController: true,
                        speedOption: [1, 2, 4, 8],
                    },
                });
            }
        }
        loadReplay();
    }, [sessionId]);

    return <div ref={containerRef} />;
};
```

### Recording Schema

```sql
CREATE TABLE session_recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id),
    workflow_run_id UUID REFERENCES workflow_runs(id),   -- link to workflow execution
    events_compressed BYTEA NOT NULL,                     -- gzipped rrweb events JSON
    event_count INTEGER NOT NULL,
    duration_ms INTEGER,                                  -- calculated from first/last event
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_recordings_user ON session_recordings(user_id, created_at DESC);
CREATE INDEX idx_recordings_workflow ON session_recordings(workflow_run_id);
```

### Important Caveat with rrweb

rrweb records DOM mutations — it works well for standard web pages but has limitations. It won't capture canvas-based content (like Google Maps), video playback, or WebGL content. It also can't capture native browser dialogs (file upload pickers, alert boxes). For most workflow automation targets (SaaS dashboards, email, forms), rrweb works excellently. If you need pixel-perfect recording later, you can add CDP-based screenshot capture as a complement (take screenshots at key moments and store alongside the rrweb events).

---

## Subsystem 5: Integration with Browser Use / Stagehand

This is where your workflow engine connects to the browser infrastructure. Your LangGraph workflow nodes that need browser automation will call into the Browser Pool Manager to get a session, run their automation, and release.

```python
# How a workflow node uses the browser infrastructure

async def browser_automation_node(state: WorkflowState):
    """Example: a LangGraph node that automates a browser task."""

    user_id = state["user_id"]
    integration = state["integration"]  # e.g., "gmail"

    # 1. Load saved session context for this user + integration
    context_data = await context_manager.load_context(user_id, integration)

    if not context_data:
        # No saved context — user needs to login interactively first
        raise NeedsAuthenticationError(integration)

    # 2. Create an automated browser session with the saved context
    session = await pool_manager.create_session(
        user_id=user_id,
        integration_id=integration,
        session_type="automation",
        context_data=context_data,
        timeout=120,  # 2 min for automated tasks
    )

    # 3. Optionally start recording for audit trail
    await recording_service.start_recording(session)

    try:
        # 4. Connect Browser Use to this session's page
        # Browser Use just needs a Playwright page object
        from browser_use import Agent

        agent = Agent(
            task=state["task_description"],
            page=session.page,        # pass the pre-configured page
            llm=your_llm_instance,
        )
        result = await agent.run()

        # 5. Save updated context (cookies may have refreshed)
        updated_context = await pool_manager.release_session(session.id)
        await context_manager.save_context(user_id, integration, updated_context)

        # 6. Save recording
        events = await recording_service.stop_recording(session)
        await recording_service.save_recording(
            session.id, user_id, events
        )

        return {"result": result}

    except Exception as e:
        # Still save context and recording on failure
        updated_context = await pool_manager.release_session(session.id)
        if updated_context:
            await context_manager.save_context(user_id, integration, updated_context)
        raise
```

### Interactive Login Flow (User-Facing)

When a user needs to connect a new account, your frontend triggers an interactive session.

```python
# API endpoint: user initiates interactive login

@router.post("/integrations/{integration_id}/connect")
async def start_interactive_login(
    integration_id: str,
    current_user: User = Depends(get_current_user),
):
    """Create an interactive browser session for user to log in."""

    session = await pool_manager.create_session(
        user_id=str(current_user.id),
        integration_id=integration_id,
        session_type="interactive",
        timeout=600,  # 10 min for manual login
    )

    # Navigate to the login page
    login_url = get_login_url(integration_id)  # e.g., "https://accounts.google.com"
    await session.page.goto(login_url)

    # Start recording this login flow
    await recording_service.start_recording(session)

    return {
        "session_id": session.id,
        "stream_url": f"/browser/sessions/{session.id}/stream",
    }


@router.post("/browser/sessions/{session_id}/complete")
async def complete_interactive_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """User signals they've finished logging in. Save context."""

    # Extract and save the session context
    context_data = await pool_manager.release_session(session_id)

    if context_data:
        await context_manager.save_context(
            str(current_user.id),
            session.integration_id,
            context_data,
        )

    # Save recording
    events = await recording_service.stop_recording(session)
    await recording_service.save_recording(session_id, str(current_user.id), events)

    return {"status": "connected", "integration": session.integration_id}
```
