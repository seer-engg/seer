---
sidebar_position: 1
---

# ADR 001: Pylint for Code Quality Enforcement

## Status
Accepted (2026-01-09)

## Context
Need consistent Python code quality enforcement with:
- Complexity metrics (cyclomatic complexity)
- Style consistency (line length, module size)
- Error detection (exception handling, common pitfalls)

Evaluated three options:
1. **Pylint**: Comprehensive linter with McCabe complexity plugin
2. **Flake8 + Radon**: Separate tools for style and complexity
3. **Radon standalone**: Complexity-only analysis

## Decision
Use **Pylint exclusively** with McCabe extension for all code quality checks.

**Rationale**:
- Single tool, single configuration
- Built-in complexity enforcement via `pylint.extensions.mccabe`
- Pre-commit integration prevents regressions
- CI enforcement in GitHub Actions

**Configuration** (`pyproject.toml`):
- max-complexity: 10
- max-line-length: 100
- max-module-lines: 600
- max-args: 7, max-branches: 15, max-locals: 20, max-statements: 60

## Consequences

**Positive**:
- Unified toolchain reduces maintenance
- Enforced standards in pre-commit prevent complexity creep
- CI catches violations before merge
- Clear metrics for refactoring priorities

**Negative**:
- Initial refactoring required to meet standards (completed)
- CI runs in isolated env → `import-error` disabled

**Migration**:
- Removed radon from dependencies
- Removed flake8 (not present)
- Blocked .pylintrc via pre-commit (enforces pyproject.toml)

## References
- Configuration: `pyproject.toml` lines 82-115
- Pre-commit: `.pre-commit-config.yaml`
- CI: `.github/workflows/pre-commit.yml`
- Related commit: 02c600a
