# 🔮 Seer - Event-Driven Multi-Agent Evaluation Platform

**Seer** is a Multi-Agent System (MAS) for evaluating AI agents through blackbox testing.

Agents communicate via an event bus, enabling modular, scalable, and traceable interactions.

---

## 🚀 Quick Start

```bash
# 1. Setup
./setup.sh

# 2. Start your agent (separate terminal)
cd /path/to/your/agent && langgraph dev --port 2024

# 3. Launch Seer 
python run.py

# 4. Open UI
# UI:                 http://localhost:8501
# Use the "Agent Threads" tab to debug individual agent conversations
```

Real LangGraph agents with proper structure, testable in isolation with `langgraph dev`.

**See [ARCHITECTURE_CHOICE.md](ARCHITECTURE_CHOICE.md) for detailed comparison.**

---

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[Simulation Flow](Simulation%20Evaluating%20Vertex%20wo%20debugging.txt)** - Example evaluation flow

---

## 🏗️ Architecture

```
Event Bus (FastAPI)
    ↓
    ├─→ Bridge → LangGraph Customer Success Agent (port 8001)
    └─→ Bridge → LangGraph Eval Agent (port 8002)
         ↓
    Your Agent (Blackbox A2A)
```

**Why This Architecture:**
- ✅ **Real agents** - Proper LangGraph structure with state, tools, workflows
- ✅ **Testable in isolation** - Use `langgraph dev` to test agents individually
- ✅ **Event-driven** - Event Bus enables agent coordination
- ✅ **Blackbox A2A testing** - Test your agent without accessing code
- ✅ **Full tracing** - See all events and agent threads in UI

**Agent Files:**
- Customer Success: `agents/customer_success/graph.py`
- Eval Agent: `agents/eval_agent/graph.py`

---

## 📁 Project Structure

```
seer/
├── event_bus/          # FastAPI event bus
├── agents/             # LangGraph agents with Event Bus bridges
│   ├── customer_success/
│   │   ├── eventbus_bridge.py   # Bridge to Event Bus
│   │   ├── graph.py             # LangGraph agent graph
│   │   └── langgraph.json
│   └── eval_agent/
│       ├── eventbus_bridge.py   # Bridge to Event Bus
│       ├── graph.py             # LangGraph agent graph
│       └── langgraph.json
├── shared/             # Shared utilities (schemas, prompts)
├── ui/                 # Streamlit UI
│   └── streamlit_app.py
├── run.py              # Launcher (starts everything)
└── requirements.txt    # Dependencies
```

---

## 💬 Example Usage

```
You: "Evaluate my agent at http://localhost:2024 (ID: abc123).
     It should recall memories and respond politely."

Seer: ✅ Generated 6 tests. Ready to run?

You: "Yes"

Seer: 📊 Results: 5/6 passed (83%)
```

---

## 📊 Monitoring & Debugging

### In Streamlit UI (Recommended)

Open http://localhost:8501 and use the tabs:

1. **💬 Chat** - Interact with Seer
2. **🤖 Agent Threads** - Debug individual agent conversations:
   - View what each agent receives and sends
   - See tool calls and responses
   - Side-by-side comparison of CS and Eval agents
   - Filter by thread ID
3. **📡 Event Bus** - Real-time event monitoring:
   - Event type filtering
   - Payload inspection
   - Color-coded events
   - Thread tracking

### Via Log Files

```bash
# Event bus activity
tail -f logs/event_bus.log

# Customer Success agent
tail -f logs/customer_success_agent.log

# Eval agent
tail -f logs/eval_agent.log
```

**What you'll see in logs:**
- 🔌 `SUBSCRIBE` - When agents join the event bus
- ✉️ `PUBLISH` - When agents send messages
- 📥 `POLL` - When agents receive messages
- 📨 Agent-specific activity with `[CS]` or `[EVAL]` prefixes

---

## 🛠️ Development

**Add a new agent:**
1. Copy `agents/eval_agent/` as template
2. Define state, tools, and workflow
3. Subscribe to event bus
4. Update `run.py` to launch it

**Add a new event type:**
1. Add to `EventType` in `event_bus/schemas.py`
2. Create payload schema
3. Publish and handle in agents

---

## 🙏 Built With

- **FastAPI** - Event bus
- **LangGraph** - Agents
- **LangChain** - LLM orchestration
- **Streamlit** - UI
- **OpenAI** - LLM models

---

**Questions? Check [README_FULL.md](README_FULL.md)** 🔮

