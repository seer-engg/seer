# Linting Guidelines for Seer

## Pylint Configuration

We use pylint with strict settings. The codebase maintains a 9.3+ rating.

To run pylint manually:
```bash
pylint api/ shared/ agents/ workflow_compiler/ worker/
```

## When Disables Are Acceptable

### ✅ Always Acceptable (with justification)

1. **import-outside-toplevel** - Avoiding circular imports or lazy loading
   ```python
   # Good
   # pylint: disable=import-outside-toplevel # Reason: Avoids circular import with services module
   from .services import some_function
   ```

2. **invalid-name** - Framework-required naming conventions (e.g., AST visitor pattern)
   ```python
   # Good - visit_* methods for AST NodeVisitor
   # pylint: disable=invalid-name # Reason: visit_* methods follow AST NodeVisitor naming convention
   def visit_Name(self, node):
       pass
   ```

3. **global-statement** - Singleton patterns with lazy initialization
   ```python
   # Good - lazy loading singleton
   _asyncpg = None
   def get_asyncpg():
       global _asyncpg  # pylint: disable=global-statement # Reason: Singleton lazy-loading pattern
       if _asyncpg is None:
           import asyncpg
           _asyncpg = asyncpg
       return _asyncpg
   ```

4. **eval-used** - Expression evaluators with sandboxed input
   ```python
   # Good - sandboxed eval with AST validation
   # pylint: disable=eval-used # Reason: Sandboxed expression evaluator with AST validation, disabled builtins, and whitelisted functions
   result = eval(compiled, {"__builtins__": {}}, safe_locals)
   ```

5. **broad-exception-caught** - Adapter boundaries converting exceptions to strings
   ```python
   # Good - LangChain adapter boundary
   except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Adapter boundary; convert all errors to user-friendly strings
       return f"Error: {str(e)}"
   ```

### ⚠️ Acceptable for Endpoints (with justification)

6. **too-many-arguments** - FastAPI/REST endpoint signatures matching API contracts
   ```python
   # Good - FastAPI endpoint with query parameters
   @router.get("/resources/{provider}/{resource_type}")
   async def browse_resources(  # pylint: disable=too-many-arguments # Reason: FastAPI endpoint signature matches REST API contract
       request: Request,
       provider: str,
       resource_type: str,
       q: Optional[str] = Query(None),
       page_token: Optional[str] = Query(None),
       page_size: int = Query(50),
   ):
       pass
   ```

### ❌ Never Acceptable Without Architectural Fix

1. **protected-access** - Use public APIs or try/except patterns instead
2. **too-many-locals/branches/statements** - Refactor into smaller functions
3. **broad-exception-caught** (non-boundary) - Narrow to specific exceptions

## Justification Format

Every disable MUST include a "Reason:" comment explaining WHY, not WHAT:

```python
# ✅ Good - explains architectural reason
# pylint: disable=import-outside-toplevel # Reason: Avoids circular import with services module

# ❌ Bad - just restates the violation
# pylint: disable=import-outside-toplevel

# ❌ Bad - explains what, not why
# pylint: disable=import-outside-toplevel # This is an import inside a function
```

## Pre-commit Hook

Pylint runs automatically in pre-commit. The hook will block commits that introduce new violations.

If you need to bypass the hook temporarily (not recommended):
```bash
git commit --no-verify
```

## Code Review Checklist

When reviewing PRs, ensure:
- [ ] No new `pylint: disable` without architectural justification in code comments
- [ ] Exception handling catches specific exception types (not bare `except Exception`)
- [ ] No dynamic attribute attachment to ORM/SQLAlchemy models
- [ ] New disables are documented in the PR description with reasoning
- [ ] Pylint score maintained or improved (check CI output)

## Common Patterns

### Circular Import Resolution

If you encounter circular imports, consider these solutions in order:

1. **Restructure imports** - Move shared code to a separate module
2. **Type hints only** - Use `from __future__ import annotations` and string type hints
3. **Import inside function** - Last resort, with proper justification

### Complexity Reduction

If you hit complexity limits (too-many-*):

1. **Extract functions** - Break large functions into smaller, focused ones
2. **Use validation helpers** - Chain validation functions that return early
3. **Dataclasses for parameters** - Group related parameters into dataclasses
4. **Nested dataclasses** - Group related attributes in dataclasses

### Exception Handling

Prefer specific exceptions over broad catches:

```python
# ✅ Good - specific exceptions
try:
    result = await api_call()
except (HTTPException, asyncio.TimeoutError, ValueError) as e:
    logger.error("API call failed: %s", e)
    return default_value
except Exception as e:
    # Unexpected error - surface to monitoring
    logger.exception("Unexpected error in api_call")
    raise

# ❌ Bad - catches everything silently
try:
    result = await api_call()
except Exception:
    return default_value
```

## Architecture Principles

1. **Import hygiene** - Avoid circular imports through proper module organization
2. **Boundary patterns** - Adapter/tool boundaries can use broad exception catching to convert to user-friendly messages
3. **Fail fast** - Let unexpected errors bubble up rather than silently catching them
4. **Simple over clever** - Prefer straightforward code over complex patterns
5. **Type safety** - Use type hints and let mypy catch issues

## Enforcement

- **Pre-commit hook** - Blocks commits with new violations
- **CI pipeline** - Runs pylint on all PRs
- **Code review** - Reviewers check for unjustified disables
- **Periodic audits** - Review existing disables quarterly

## Questions?

If you're unsure whether a disable is appropriate, ask in code review or consult the tech lead.
