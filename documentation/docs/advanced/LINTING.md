---
sidebar_position: 3
---

# Linting Standards

Python code quality enforcement via pylint.

## Quick Start

### Setup
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Run Manually
```bash
# All files
pre-commit run --all-files

# Specific files
pylint path/to/file.py

# Single directory
pylint api/
```

---

## Configuration

Configured in `pyproject.toml`:

| Setting | Limit | Purpose |
|---------|-------|---------|
| `max-complexity` | 10 | Cyclomatic complexity per function |
| `max-line-length` | 100 | Characters per line |
| `max-module-lines` | 600 | Lines per file |
| `max-args` | 7 | Function parameters |
| `max-branches` | 15 | Conditional branches |
| `max-locals` | 20 | Local variables |
| `max-statements` | 60 | Statements per function |

**Disabled checks**:
- `missing-docstring` - Documentation optional
- `too-few-public-methods` - Utility classes allowed
- `fixme` - TODOs permitted
- `import-error` - CI runs isolated

---

## Pre-commit Integration

Runs automatically on `git commit`:
- Checks only staged files
- Blocks commit on violations
- Shows specific line numbers and errors

**Bypass** (emergency only):
```bash
git commit --no-verify
```

**Configuration**: `.pre-commit-config.yaml`

---

## CI/CD Integration

GitHub Actions runs on all PRs:
- File: `.github/workflows/pre-commit.yml`
- Runs: `pre-commit run --all-files`
- Python: 3.12
- Blocks merge on failures

**Local validation**:
```bash
# Replicate CI check
pre-commit run --all-files
```

---

## Fixing Violations

### Common Issues

**1. Line too long (C0301)**
```python
# Bad
some_function_call(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

# Good
some_function_call(
    arg1, arg2, arg3,
    arg4, arg5, arg6,
    arg7, arg8, arg9, arg10
)
```

**2. Too complex (R1260)**
```python
# Refactor: extract helper functions
# Refactor: use early returns
# Refactor: simplify conditionals
```

**3. Module too long (C0302)**
```python
# Split into multiple modules
# Move related functions to separate files
```

---

## Architecture Decision

See [ADR 001: Pylint for Code Quality](/adr/pylint-for-code-quality) for tool selection rationale.

---

## Related Documentation
- [Configuration Reference](./CONFIGURATION)
- [Development Workflow](/#development-workflow)
