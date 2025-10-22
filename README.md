# 🔮 Seer - Multi-Agent System for Evaluating AI Agents

**Seer** is a Multi-Agent System (MAS) for evaluating AI agents through blackbox testing.

---

## 🚀 Quick Start

```bash
# 1. Setup PostgreSQL
# Quick: docker run --name seer-postgres -e POSTGRES_DB=seer_db -e POSTGRES_USER=seer_user -e POSTGRES_PASSWORD=seer_password -p 5432:5432 -d postgres:15

# 2. Configure .env
# Add: OPENAI_API_KEY=your_key
#      POSTGRESQL_CONNECTION_STRING=postgresql://seer_user:seer_password@localhost:5432/seer_db

# 3. Install dependencies
./setup.sh

# 4. Start your agent (separate terminal)
cd /path/to/your/agent && langgraph dev --port 2024

# 5. Launch Seer 
python run.py

# 6. Open UI
# UI:                 http://localhost:8501
# Chat directly with the orchestrator, which delegates to eval/coding agents
# LangGraph Studio:   https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8000
```

**Agent Files:**
- Orchestrator: `agents/orchestrator/graph.py` (Conceirge)
- Eval Agent: `agents/eval_agent/graph.py` (Test generation and execution)
- Coding Agent: `agents/coding_agent/graph.py` (Code analysis and review)

**Configuration:**
- `deployment-config.json` - Centralized agent UUIDs and ports
- `shared/config.py` - Configuration management utilities

---

## 📁 Project Structure

```
seer/
├── agents/             # LangGraph agents with inter agent communication
│   ├── orchestrator/           # Conversational orchestrator with inter agent communication
│   │   ├── graph.py  # Main orchestrator logic
│   │   └── langgraph.json
│   ├── eval_agent/
│   │   ├── graph.py  # LangGraph agent graph
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

Seer uses **PostgreSQL with Peewee ORM** for scalable, production-ready data persistence:
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
- Inter agent communication traces

### Via Log Files

```bash
# Orchestrator agent (LangGraph)
tail -f logs/orchestrator_langgraph.log

# Eval agent (LangGraph)
tail -f logs/eval_agent_langgraph.log

# Coding agent (LangGraph)
tail -f logs/coding_agent_langgraph.log
```