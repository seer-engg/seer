# 🔮 Seer - A2A Multi-Agent Evaluation Platform

**Seer** is a Multi-Agent System (MAS) for evaluating AI agents through blackbox testing.

Agents communicate via LangGraph's Agent-to-Agent (A2A) protocol with an orchestrator acting as a central hub, enabling modular, scalable, and traceable interactions.

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
# Use the "Orchestrator Monitor" tab to see message flow between all agents
# LangGraph Studio:   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8001
```

Real LangGraph agents with proper structure, testable in isolation with `langgraph dev`.

**See [ARCHITECTURE_CHOICE.md](ARCHITECTURE_CHOICE.md) for detailed comparison.**

---

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[DATABASE.md](DATABASE.md)** - Database schema and persistence layer
- **[Simulation Flow](Simulation%20Evaluating%20Vertex%20wo%20debugging.txt)** - Example evaluation flow

---

## 🏗️ Architecture

```
Your Agent (Blackbox A2A)
        ↓
Customer Success Agent ←→ Orchestrator Agent ←→ Eval Agent
    (port 8001)         (port 8000)         (port 8002)
        ↑                      ↑                      ↑
        └──────────────────────┴──────────────────────┘
                     A2A Protocol (Hub & Spoke)
```

**Why This Architecture:**
- ✅ **Real agents** - Proper LangGraph structure with state, tools, workflows
- ✅ **Central coordination** - Orchestrator agent acts as message router and data store
- ✅ **Testable in isolation** - Use `langgraph dev` to test agents individually
- ✅ **Blackbox A2A testing** - Test your agent without accessing code
- ✅ **Full tracing** - See all message flow through the Orchestrator Monitor
- ✅ **Persistent storage** - SQLite database stores all chat threads and eval data
- ✅ **Simplified deployment** - No bridge processes needed

**Agent Files:**
- Orchestrator: `agents/orchestrator/graph.py` (Central hub)
- Customer Success: `agents/customer_success/graph.py`
- Eval Agent: `agents/eval_agent/graph.py`

**Configuration:**
- `deployment-config.json` - Centralized agent UUIDs and ports
- `shared/config.py` - Configuration management utilities

---

## 📁 Project Structure

```
seer/
├── agents/             # LangGraph agents with A2A communication
│   ├── orchestrator/           # Central coordinating agent (group chat hub)
│   │   ├── graph.py             # Main orchestrator logic
│   │   └── langgraph.json
│   ├── customer_success/
│   │   ├── graph.py             # LangGraph agent graph
│   │   └── langgraph.json
│   └── eval_agent/
│       ├── graph.py             # LangGraph agent graph
│       └── langgraph.json
├── shared/             # Shared utilities (schemas, prompts, database, config)
├── data/               # SQLite database storage
├── ui/                 # Streamlit UI
│   └── streamlit_app.py
├── deployment-config.json  # Agent UUIDs and configuration
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

## 💾 Database & Persistence

Seer uses SQLite to persist all data:
- **Chat threads** - All conversations with complete history
- **Messages** - Every message from users and agents
- **Agent activities** - What each agent did in each thread (for debugging/tracing)
- **Eval suites** - Generated test cases
- **Test results** - Detailed test execution results

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
3. **🎛️ Orchestrator Monitor** - Real-time message flow monitoring:
   - See all messages between agents
   - Track message broadcasting
   - View agent registration status
   - Monitor conversation threads

### LangGraph Studio (Browser-based)

Access LangGraph Studio for advanced debugging:
- **Customer Success Agent**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8001
- **Eval Agent**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8002

Features:
- Visual graph execution
- Real-time conversation monitoring
- Tool call inspection
- State management debugging
- A2A communication traces

### Via Log Files

```bash
# Orchestrator agent (LangGraph)
tail -f logs/orchestrator_langgraph.log

# Customer Success agent (LangGraph)
tail -f logs/customer_success_langgraph.log

# Eval agent (LangGraph)
tail -f logs/eval_agent_langgraph.log
```

**What you'll see in logs:**
- 🎛️ Orchestrator activity and message broadcasting
- 🤖 A2A communication traces between all agents
- 📨 Agent-specific activity with `[ORCHESTRATOR]`, `[CS]`, or `[EVAL]` prefixes
- 🔄 Agent registration and status updates

---

## 🛠️ Development

**Add a new agent:**
1. Copy `agents/eval_agent/` as template
2. Define state, tools, and workflow
3. Register with Orchestrator agent (it will broadcast messages to you)
4. Update `run.py` to launch it

**Add new data types:**
1. Add schemas in `shared/schemas.py`
2. Update Orchestrator agent to handle the new data types
3. Update other agents to use Orchestrator for storage/retrieval

---

## 🙏 Built With

- **LangGraph** - All agents (Orchestrator, Customer Success, Eval)
- **LangChain** - LLM orchestration
- **Streamlit** - UI
- **OpenAI** - LLM models
- **SQLite** - Data persistence

---

**Questions? Check [README_FULL.md](README_FULL.md)** 🔮

