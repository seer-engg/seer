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
1. Evaluate my agent at buggy_coder at github https://github.com/seer-engg/buggy-coder. It should be fixing flawed python code correctly and sharing the fixed code too
2. Evaluate my agent at buggy_coder at github https://github.com/seer-engg/buggy-coder/tree/seer/codex/20251108-125730-92e6b17/v1. It should be fixing flawed python code correctly and sharing the fixed code too
3. Evaluate my buggy_coder at https://github.com/seer-engg/buggy-coder for syncing Asana ticket updates when a GitHub PR is merged

## Notes
- To enable codex handoff, set `CODEX_HANDOFF_ENABLED` to `true` in `.env` file.