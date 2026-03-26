# Backend Test Strategy Analysis

**Date:** 2026-03-25
**Context:** Following up on test audit report (2026-03-23) with focus on backend tests only.

---

## Current State (Post-Fixes)

| Layer | Files | Mock Usage | Skipped | Status |
|-------|-------|------------|---------|--------|
| Unit | 241 | 5,078 patch/mock calls | 0 | Heavy mocking - maintenance burden |
| Integration | 47 | Minimal | 2 | Good - tests real behavior |
| E2E | 10 | None | 2 | **Fixed** - now uses Testcontainers (Postgres + Redis) |

### What's Already Fixed

**R3 (Switch to Postgres) - COMPLETE**

The E2E tests now use real Testcontainers infrastructure:
- PostgreSQL container with real schema migrations
- Redis container for caching/queues
- LangGraph checkpointer initialized with real connection pool
- User emulation mode for auth (bypasses Clerk JWT)
- Taskiq direct executor for synchronous task execution

```
tests/e2e/
├── conftest.py          # Container fixtures, checkpointer, auth
├── fixtures/
│   ├── containers.py    # Postgres + Redis testcontainers
│   └── database.py      # DB session management
└── workflows/
    ├── test_workflow_lifecycle.py   # Create → Publish → Execute → Verify
    ├── test_trigger_execution.py    # Webhook triggers, catalog
    └── test_error_handling.py       # Validation, auth, edge cases
```

**E2E Test Results:** 26 passed, 2 skipped (trigger provisioning - requires external resources)

---

## The Problem: Mock-Heavy Unit Tests

The report identified **5,078 mock/patch usages** across 241 unit test files. This creates:

1. **False confidence** - Tests pass but prod breaks
2. **Refactor friction** - Every internal change breaks tests
3. **Maintenance tax** - At 10 features/day, mocks drift from reality within days
4. **Slow CI** - 241 files × setup overhead adds up

### Example Anti-Pattern

```python
# tests/unit/api/workflows/test_lifecycle.py
# Tests internal utilities that are implementation details
async def test_parse_workflow_id():
    assert parse_workflow_id("wf_abc123") == "abc123"

async def test_hash_spec():
    spec = {"version": "2", "nodes": []}
    assert _hash_spec(spec) == "expected_hash"
```

These utility functions are tested through every integration/E2E test that creates a workflow. Testing them separately:
- Adds maintenance burden
- Tests implementation, not behavior
- Provides no additional confidence

---

## Recommendation: Testing Trophy for Backend

The **Testing Trophy** (Kent C. Dodds) prioritizes confidence over coverage:

```
        ┌─────────────┐
        │   E2E       │  ← Critical user journeys (small count, high value)
        │  (10 files) │
        ├─────────────┤
        │ Integration │  ← Component interactions (the sweet spot)
        │ (47 files)  │
        ├─────────────┤
        │    Unit     │  ← Pure logic only (shrink this layer)
        │  (241→~50)  │
        └─────────────┘
```

### What Unit Tests to KEEP

| Category | Examples | Why Keep |
|----------|----------|----------|
| **Pure computation** | Workflow compiler stages, expression evaluator | Deterministic, no I/O, fast feedback |
| **Schema validation** | Workflow spec parsing, node type validation | Complex rules, many edge cases |
| **Business rules** | Usage limits, trial expiration, billing math | Critical correctness, easy to unit test |
| **Data transformation** | JSON→model conversion, serialization | Easy to test in isolation |

### What Unit Tests to DELETE

| Category | Examples | Why Delete |
|----------|----------|------------|
| **Mocked service wiring** | "Did router call service with args?" | Integration tests cover this better |
| **Mocked DB operations** | CRUD with mocked ORM | Integration tests use real DB |
| **Trivial property tests** | `test_cents_to_dollars()` | Covered by real usage |
| **Implementation details** | `test_parse_workflow_id()` | Internal helpers, not behavior |

---

## Action Plan

### Phase 1: Prune Unit Tests (2-3 days, incremental)

**Target:** Reduce from 241 files to ~50-80 files (keep only pure logic)

**Files to DELETE immediately:**

```
tests/unit/database/test_overage_models.py     # Trivial property tests
tests/unit/api/workflows/test_lifecycle.py     # Internal utility tests
tests/unit/api/workflows/test_execution.py     # Enum verification (if only testing enums)
```

**Directories to AUDIT and heavily prune:**

| Directory | Files | Action |
|-----------|-------|--------|
| `tests/unit/api/` | 48 | Keep schema validation, delete mocked router tests |
| `tests/unit/services/` | 29 | Keep business logic, delete mocked dependency tests |
| `tests/unit/tools/` | 27 | Keep validation, delete mocked execution tests |
| `tests/unit/agents/` | 11 | Keep tool logic, delete mocked LangGraph tests |

**Directories to KEEP mostly intact:**

| Directory | Files | Reason |
|-----------|-------|--------|
| `tests/unit/core/` | 45 | Compiler is pure logic, complex rules |
| `tests/unit/prompts/` | ? | Template validation, deterministic |

### Phase 2: Expand E2E Coverage (2-3 days)

Add critical backend user journeys to `tests/e2e/`:

1. **OAuth credential flow** - Connect → Store → Refresh → Use in workflow
2. **Workflow versioning** - Create v1 → Edit → Publish v2 → Rollback
3. **HITL execution** - Run → Pause at human step → Approve → Resume
4. **Trigger lifecycle** - Subscribe → Receive event → Execute → Verify
5. **Error recovery** - Fail mid-execution → Retry → Succeed

### Phase 3: CI Pipeline Tiers (1 day)

```yaml
# .github/workflows/backend-tests.yml

# Tier 1: Every push (~30s)
lint-and-type:
  - ruff check
  - mypy

# Tier 2: Every PR (~2min)
integration-smoke:
  - pytest tests/integration -x --tb=short
  - pytest tests/e2e -x --tb=short

# Tier 3: Nightly (~10min)
full-suite:
  - pytest tests/ --tb=long
```

---

## Metrics to Track

| Metric | Current | Target | Why |
|--------|---------|--------|-----|
| Unit test files | 241 | 50-80 | Less maintenance, faster CI |
| Mock usages | 5,078 | <500 | Higher confidence |
| E2E test files | 10 | 15-20 | Critical journeys covered |
| CI time (PR) | ? | <3min | Fast feedback loop |
| Skipped tests | 4 | 0 | No phantom coverage |

---

## Decision Matrix: Keep vs Delete

For each unit test file, ask:

```
┌─────────────────────────────────────────────────────────────┐
│ Does this test pure computation with no I/O?                │
│   YES → KEEP                                                │
│   NO  ↓                                                     │
├─────────────────────────────────────────────────────────────┤
│ Does this test use >3 mocks?                                │
│   YES → DELETE (replace with integration test if needed)    │
│   NO  ↓                                                     │
├─────────────────────────────────────────────────────────────┤
│ Would this test break if we refactored internals?           │
│   YES → DELETE (tests implementation, not behavior)         │
│   NO  ↓                                                     │
├─────────────────────────────────────────────────────────────┤
│ Is there an integration/E2E test covering the same path?    │
│   YES → DELETE (redundant)                                  │
│   NO  → KEEP or add integration test first                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

| Recommendation | Status | Effort | Impact |
|----------------|--------|--------|--------|
| R3: Switch to Postgres | **DONE** | - | E2E now catches real DB bugs |
| R4: Prune unit tests | TODO | 2-3 days | -80% unit tests, +confidence |
| R2: Add user journey E2E | TODO | 2-3 days | Catches real user bugs |
| R5: CI pipeline tiers | TODO | 1 day | Fast feedback + full coverage |

**Execution order:**
1. **Now:** Incremental pruning - delete 10-20 mock-heavy files per day during normal dev
2. **This week:** Add 5 critical E2E journeys
3. **Next week:** CI tier structure

---

## Appendix: Files Flagged for Deletion

Based on analysis, these files are candidates for immediate removal:

```bash
# Trivial/implementation tests
tests/unit/database/test_overage_models.py
tests/unit/api/workflows/test_lifecycle.py

# Heavy mock tests (audit individually)
tests/unit/api/workflows/*.py  # except validation tests
tests/unit/services/workflows/*.py  # except business rule tests
tests/unit/tools/test_registry.py  # consolidate with integration version
```

## Appendix B: Most Mock-Heavy Files (Audit Results)

**Top 20 files by mock usage** - prime candidates for deletion:

| Mocks | File | Recommendation |
|-------|------|----------------|
| 464 | `tests/unit/worker/test_chat_tasks.py` | DELETE - replace with integration test |
| 405 | `tests/unit/services/browser/test_browser_service.py` | DELETE - browser is external I/O |
| 201 | `tests/unit/worker/test_knowledge_tasks.py` | DELETE - replace with integration test |
| 178 | `tests/unit/services/workflows/test_execution.py` | AUDIT - keep business rules only |
| 162 | `tests/unit/services/browser/test_profile_manager.py` | DELETE - browser I/O |
| 131 | `tests/unit/api/workflows/test_execution.py` | AUDIT - keep validation only |
| 124 | `tests/unit/api/integrations/test_services.py` | AUDIT - OAuth needs real testing |
| 98 | `tests/unit/api/browser/test_ws_router.py` | DELETE - WebSocket needs integration |
| 88 | `tests/unit/api/workflows/test_triggers_service.py` | AUDIT - keep business rules |
| 84 | `tests/unit/observability/test_org_usage.py` | AUDIT - billing math is pure logic |
| 82 | `tests/unit/services/workflows/test_triggers.py` | AUDIT - keep idempotency tests |
| 82 | `tests/unit/services/browser/test_recording_service.py` | DELETE - browser I/O |
| 80 | `tests/unit/core/test_agent_node.py` | KEEP - LangGraph node logic |
| 74 | `tests/unit/api/agents/test_workflow_services.py` | AUDIT |
| 69 | `tests/unit/services/browser/test_pool_manager.py` | DELETE - browser I/O |
| 68 | `tests/unit/api/subscriptions/test_stripe_webhook.py` | KEEP - Stripe signatures critical |
| 61 | `tests/unit/core/test_image_gen_node.py` | KEEP - node logic |
| 60 | `tests/unit/api/browser/test_router.py` | DELETE - needs integration |
| 59 | `tests/unit/api/forms/test_router.py` | AUDIT |
| 59 | `tests/unit/api/core/test_usage_limit_middleware.py` | KEEP - billing rules |

**Total mock usages in top 20:** ~2,500 (50% of all mocks)

**Quick wins:** Deleting the browser/* and worker/* test files removes ~1,400 mock usages immediately.
