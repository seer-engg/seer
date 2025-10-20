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
# Chat directly with the orchestrator, which delegates to eval/coding agents
# LangGraph Studio:   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8000
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
User ←→ Streamlit UI ←→ Orchestrator Agent (Conversational)
                              (port 8000)
                                   ↓
                         ┌─────────┴─────────┐
                         ↓                   ↓
                  Eval Agent          Coding Agent
                  (port 8002)         (port 8003)
                         
              A2A Point-to-Point Communication
```

**Why This Architecture:**
- ✅ **Conversational orchestrator** - Users interact directly with orchestrator
- ✅ **Point-to-point A2A** - No broadcast, only targeted agent delegation
- ✅ **Real agents** - Proper LangGraph structure with state, tools, workflows
- ✅ **Quick acknowledgment** - Orchestrator acknowledges and relays agent responses
- ✅ **Testable in isolation** - Use `langgraph dev` to test agents individually
- ✅ **Blackbox A2A testing** - Test your agent without accessing code
- ✅ **Persistent storage** - SQLite database stores all chat threads and eval data
- ✅ **Simplified deployment** - No bridge or customer success agent needed

**Agent Files:**
- Orchestrator: `agents/orchestrator/simplified_graph.py` (Conversational hub with A2A routing)
- Eval Agent: `agents/eval_agent/simplified_graph.py` (Test generation and execution)
- Coding Agent: `agents/coding_agent/graph.py` (Code analysis and review)

**Configuration:**
- `deployment-config.json` - Centralized agent UUIDs and ports
- `shared/config.py` - Configuration management utilities

---

## 📁 Project Structure

```
seer/
├── agents/             # LangGraph agents with A2A communication
│   ├── orchestrator/           # Conversational orchestrator with A2A routing
│   │   ├── simplified_graph.py  # Main orchestrator logic
│   │   └── langgraph.json
│   ├── eval_agent/
│   │   ├── simplified_graph.py  # LangGraph agent graph
│   │   └── langgraph.json
│   └── coding_agent/
│       ├── graph.py             # LangGraph agent graph
│       └── langgraph.json
├── shared/             # Shared utilities (schemas, prompts, database, config)
├── data/               # SQLite database storage
├── ui/                 # Streamlit UI
│   └── streamlit_app.py
├── deployment-config.json  # Agent UUIDs and configuration
├── run.py              # Launcher (starts everything)
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

1. **💬 Chat** - Interact with the conversational orchestrator
   - Orchestrator handles user conversations directly
   - Automatically delegates to eval or coding agents as needed
2. **📊 Results** - View evaluation results and test suites
   - See test case results
   - Track evaluation progress
   - View historical test runs

### LangGraph Studio (Browser-based)

Access LangGraph Studio for advanced debugging:
- **Orchestrator Agent**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8000
- **Eval Agent**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8002
- **Coding Agent**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8003

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

# Eval agent (LangGraph)
tail -f logs/eval_agent_langgraph.log

# Coding agent (LangGraph)
tail -f logs/coding_agent_langgraph.log
```

**What you'll see in logs:**
- 🎛️ Orchestrator conversations and A2A delegation
- 🤖 Point-to-point A2A communication traces
- 📨 Agent-specific activity with tool calls and responses
- 🔄 Agent registration and status updates

---

## 🛠️ Development

**Add a new agent:**
1. Copy `agents/eval_agent/` as template
2. Define state, tools, and workflow
3. Add agent to `deployment-config.json` with UUID and port
4. Add delegation tool in orchestrator for the new agent
5. Update `run.py` to launch it

**Add new data types:**
1. Add schemas in `shared/schemas.py`
2. Update Orchestrator's data_manager to handle the new data types
3. Add tools in orchestrator for storing/retrieving the new data

---

## 🙏 Built With

- **LangGraph** - All agents (Orchestrator, Customer Success, Eval)
- **LangChain** - LLM orchestration
- **Streamlit** - UI
- **OpenAI** - LLM models
- **SQLite** - Data persistence

---

**Questions? Check [README_FULL.md](README_FULL.md)** 🔮

