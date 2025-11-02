# 🔮 Seer - Multi-Agent System for Evaluating AI Agents

**Seer** is a Multi-Agent System (MAS) for evaluating AI agents through blackbox testing.


## 🚀 Setup

```bash
# 1. Configure .env
cp .env.example .env

# 2. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -e .

# 3. Start Seer
python run.py
```

### Sample Eval Agent Input
"""
Evaluate my agent at buggy_coder at github https://github.com/seer-engg/buggy-coder. It should be fixing flawed python code correctly and sharing the fixed code too
"""

## Notes
- To enable codex handoff, set `CODEX_HANDOFF_ENABLED` to `true` in `.env` file.