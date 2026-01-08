# Pre-commit Hooks Setup

## Overview
This repository uses pre-commit hooks to enforce code quality standards before commits.

## Code Quality Standards
- **Max Complexity**: 10 (cyclomatic complexity)
- **Max Lines per File**: 600
- **Max Line Length**: 100 characters

## Tools Enforced
1. **Flake8**: Style guide enforcement
2. **Radon**: Complexity and maintainability metrics
3. **Pylint**: Comprehensive linting with configured limits

## Installation

### First Time Setup
```bash
# Install dev dependencies (includes pre-commit)
uv sync --dev

# Install pre-commit hooks
pre-commit install
```

### Manual Run (Optional)
```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

## What Happens on Commit
Before each commit, the following checks run automatically:
1. Flake8 checks for style violations and complexity
2. Radon verifies complexity grades (must be B or better)
3. Radon checks maintainability index
4. Pylint validates against configured limits
5. Basic file checks (trailing whitespace, file size, etc.)

**If any check fails, the commit will be blocked.**

## Bypassing Hooks (NOT Recommended)
```bash
# Only use in emergencies
git commit --no-verify -m "message"
```

## Troubleshooting

### Hook installation fails
```bash
# Reinstall pre-commit
uv pip install --force-reinstall pre-commit
pre-commit install
```

### False positives
Update `.pre-commit-config.yaml` and discuss with the team before modifying limits.
