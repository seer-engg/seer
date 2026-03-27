# Backend Test Strategy Analysis

**Date:** 2026-03-27 (fresh analysis)
**Context:** Independent re-examination of backend test suite after 6 phases of test overhaul work.
**Scope:** Backend only (no Playwright/frontend).

---

## Executive Summary

The test overhaul has delivered **measurable structural improvements**. The suite went from a mock-heavy, SQLite-based setup to a Testing Trophy architecture with real PostgreSQL across all layers. The integration layer — previously hollow with 120+ mock sites in a single file — is now **genuinely integrating**: all 16 files with mocks use them only at appropriate external boundaries.

**Overall coverage: 62.4%** (20,679 / 33,165 statements) — up from 57% pre-overhaul.

The remaining work is not about test architecture (that's solved) but about **coverage breadth**: 6 modules with >200 statements sit below 40% coverage, totaling ~3,500 uncovered statements in user-facing code.

---

## Current State (Verified 2026-03-27)

### Test Suite Overview

| Layer | Files | Tests | Mock Sites | Skipped | Status |
|-------|-------|-------|------------|---------|--------|
| Unit | 178 | 3,306 | ~2,400 | 0 | All passing |
| Integration | 40 | 489 | ~243 | 2 (OPENAI_API_KEY) | All passing |
| E2E | 9 | 78 | 0 | 2 (trigger timeout) | All passing |
| **Total** | **227** | **3,873** | **~2,643** | **4** | **3,869 passed** |

**Runtime:** 306s (5m06s) for full suite. Unit-only: 42s.

### Test Infrastructure

| Component | Implementation | Notes |
|-----------|---------------|-------|
| Database (all layers) | PostgreSQL via Testcontainers (`pgvector/pgvector:pg17`) | Zero SQLite references |
| Redis/Valkey | Testcontainers (`valkey/valkey:7-alpine`) | E2E Taskiq broker |
| Test isolation | Truncation-based (`TRUNCATE CASCADE`) | Not transaction rollback — avoids PostgreSQL IntegrityError abort issue |
| Container scope | Session-scoped | ~5s startup, amortized across all tests |
| E2E task execution | In-process via Taskiq patch | Synchronous with execution tracking |

### Coverage by Module (All Layers Combined)

| Module | Covered/Total | Coverage | Assessment |
|--------|---------------|----------|------------|
| `core/schema` | 331/354 | 93.5% | Excellent |
| `core/expr` | 611/672 | 90.9% | Excellent |
| `core/compiler` | 636/708 | 89.8% | Excellent |
| `database` | 1,121/1,264 | 88.7% | Excellent |
| `services/workflows` | 343/389 | 88.2% | Excellent |
| `core/runtime` | 472/552 | 85.5% | Good |
| `core/nodes` | 1,238/1,492 | 83.0% | Good |
| `services/collaboration` | 173/211 | 82.0% | Good |
| `agents/nexus` | 737/902 | 81.7% | Good |
| `observability` | 669/876 | 76.4% | Good |
| `api/core` (middleware) | 460/617 | 74.6% | Good |
| `api/workflows` | 1,560/2,125 | 73.4% | Good |
| `core/triggers` | 849/1,174 | 72.3% | Good |
| `worker/tasks` | 545/888 | 61.4% | Moderate (was 0%) |
| `tools` (all) | 3,939/6,448 | 61.1% | Moderate |
| `api/agents` | 649/1,174 | 55.3% | Gap |
| `api/subscriptions` | 563/1,166 | 48.3% | Gap |
| `services/integrations` | 1,009/2,165 | 46.6% | Gap |
| `api/organizations` | 279/649 | 43.0% | Gap |
| `api/integrations` | 376/921 | 40.8% | Gap |
| `api/files` | 157/401 | 39.2% | Critical gap |
| `services/memory` | 200/541 | 37.0% | Critical gap |
| `services/browser` | 447/1,209 | 37.0% | Critical gap |
| `api/memory` | 106/304 | 34.9% | Critical gap |
| `api/browser` | 121/351 | 34.5% | Critical gap |
| `tools/postgres` | 112/352 | 31.8% | Critical gap |
| `tools/supabase` | 167/550 | 30.4% | Critical gap |
| `database/seed` | 0/337 | 0.0% | Not tested (seed data) |

---

## What's Working Well

1. **Testing Trophy architecture is correct.** E2E (0 mocks) → Integration (external-only mocks) → Unit (pure logic). The layers serve their intended purpose.

2. **Integration layer quality is high.** All 16 integration files with mocks were audited — every mock is at an appropriate external boundary (Stripe, OAuth, LLM agents, Taskiq, Redis). Zero internal-service mocking.

3. **Core engine is battle-tested.** Compiler (90%), runtime (86%), expressions (91%), schema (94%), nodes (83%). The workflow execution pipeline has excellent coverage.

4. **PostgreSQL everywhere.** All layers use real Postgres via Testcontainers. The SQLite divergence problem from the original recommendations is fully resolved.

5. **Worker tasks went from 0% → 61%.** The most dangerous coverage gap (background task logic) is now meaningfully tested with pure logic tests covering error detection, idempotency, status transitions.

6. **Zero skipped tests in unit layer.** 4 total skips are all justified (2 require OPENAI_API_KEY, 2 require external trigger provisioning).

---

## What Needs Work — Three Remaining Problems

### Problem 1: Large Untested Surface Areas (3,500+ uncovered statements)

Seven modules sit below 40% coverage with >200 statements each. These are user-facing features where bugs would ship undetected:

| Module | Uncovered Stmts | Impact | What's Missing |
|--------|----------------|--------|----------------|
| `services/browser/` | 762 of 1,209 | **High** | Browser pool management, CDP streaming, recording pipeline. Only data layer (encryption, profiles, session storage) is covered. |
| `tools/supabase/` | 383 of 550 | **Medium** | Database queries, storage operations, auth admin, edge functions. Zero dedicated tests. |
| `tools/postgres/` | 240 of 352 | **Medium** | Query execution, schema introspection, table operations. Only init/registry tested. |
| `api/files/` | 244 of 401 | **Medium** | File upload/download endpoints. |
| `api/memory/` | 198 of 304 | **Medium** | Memory bank API endpoints. |
| `api/browser/` | 230 of 351 | **Medium** | Browser session API endpoints. |
| `services/memory/` | 341 of 541 | **Medium** | Memory extraction, mem0 client, memory bank service logic. |

**Total: ~2,400 uncovered statements in user-facing code below 40%.**

Additionally, at the 40-50% range:
- `api/integrations/` (545 uncovered) — OAuth connection management
- `api/organizations/` (370 uncovered) — Team management, permissions
- `api/subscriptions/` (603 uncovered) — Billing, tier resolution
- `services/integrations/` (1,156 uncovered) — Provider implementations, resource management

**Total across all gaps: ~5,000+ uncovered statements.**

### Problem 2: Misclassified Unit Tests (Minor)

4 unit test files (13 tests) use the `db_engine` fixture, making them integration tests in disguise:

| File | Tests | Fixture |
|------|-------|---------|
| `unit/api/collaboration/test_lock_router.py` | 3 | `db_engine`, `api_client` |
| `unit/api/organizations/test_org_collaboration_events.py` | 1 | `db_engine`, `api_client` |
| `unit/api/workflows/services/test_collaboration_events.py` | 1 | `db_engine` |
| `unit/api/workflows/test_consultant_visibility.py` | 8 | `db_engine` |

These should be moved to `tests/integration/` for clarity, though they work correctly where they are.

### Problem 3: Mock-Heavy Unit Tests Still Exist (Lower Priority)

The top 10 most mock-heavy unit test files:

| File | Mocks | Category | Verdict |
|------|-------|----------|---------|
| `test_agent_node.py` | 66 | Core node | REVIEW — Could some test via `compile_workflow → ainvoke`? |
| `test_workflow_validation.py` | 54 | Tools | KEEP — Validation logic |
| `test_clerk_verifier.py` | 48 | Auth | KEEP — Security-critical JWT verification |
| `test_image_gen_node.py` | 43 | Core node | REVIEW — LLM mocking |
| `test_linkedin.py` | 39 | Tools | KEEP — External API |
| `test_knowledge_tasks.py` | 35 | Worker | KEEP — Pipeline logic with good assertions |
| `test_memory_tasks.py` | 34 | Worker | KEEP — Gating logic, extraction paths |
| `test_general_chat_task.py` | 32 | Worker | KEEP — Error isolation tests |
| `test_execution.py` (mcp) | 27 | MCP | KEEP — Tool execution |
| `test_chat_tasks.py` | 26 | Worker | KEEP — Error chain traversal, status machines |

Most of these are testing real logic (error chains, state machines, validation). The REVIEW candidates (~109 mocks in agent_node + image_gen_node) mock LLM responses — some could potentially move to core integration tests using the existing `compile_workflow → ainvoke` pattern, but the ROI is low.

---

## Progress vs. Original Recommendations

| # | Recommendation | Status | Details |
|---|---------------|--------|---------|
| R1 | Fix skipping Playwright tests | N/A | Backend-only scope |
| R2 | Add user journey E2E tests | **Done** | 5 → 9 files, 46 → 78 tests. Lifecycle, versioning, HITL, export/import, variables, multi-node, management |
| R3 | Switch backend E2E from SQLite to Postgres | **Done (ALL layers)** | E2E + Integration + Unit all use Testcontainers PostgreSQL |
| R4 | Prune mocked unit tests | **Done (2 rounds)** | 241 → 178 files, ~5,078 → ~2,400 mock sites (-53%) |
| R5 | CI pipeline tiers | **Not started** | No tiered CI structure yet |

### Cumulative Impact

| Metric | Before (2026-03-23) | After (2026-03-27) | Change |
|--------|---------------------|-------------------|--------|
| Unit test files | 241 | 178 | -63 (-26%) |
| Unit mock sites | ~5,078 | ~2,400 | -2,678 (-53%) |
| Integration test files | 35 | 40 | +5 |
| Integration tests | 359 | 489 | +130 (+36%) |
| E2E test files | 5 | 9 | +4 |
| E2E tests | 46 | 78 | +32 (+70%) |
| Worker coverage | 0% | 61.4% | **+61.4%** |
| Overall coverage | 57% | 62.4% | +5.4% |
| SQLite in tests | Yes | No | **Eliminated** |
| Integration mock quality | 25/40 files with mocks, many internal | 16/40, all external-only | **Structural fix** |

---

## Reliability Assessment — Where Are Bugs Hiding?

### Tier 1: High Risk (Bugs Would Cause User-Visible Failures)

| Risk Area | Stmts | Coverage | Why | Current Defense |
|-----------|-------|----------|-----|-----------------|
| Browser automation (`services/browser/`) | 1,209 | 37.0% | Core product feature. Pool mgmt, CDP streaming, recording — all untested | Data layer only (encryption, profiles) |
| OAuth/integration endpoints (`api/integrations/`) | 921 | 40.8% | First thing users do after signup. Connection management, scope validation | Unit tests for some services, no integration tests for endpoints |
| Organization API (`api/organizations/`) | 649 | 43.0% | Team management, permissions, billing | Minimal unit tests |
| Chat worker orchestration (`worker/tasks/chat.py`) | 510 | 40% | Redis locks, LLM streaming, checkpoint management — runs all conversations | Pure logic tests cover error paths; orchestration flow untested |

### Tier 2: Medium Risk (Bugs Would Cause Incorrect Data or Broken Features)

| Risk Area | Stmts | Coverage | Why | Current Defense |
|-----------|-------|----------|-----|-----------------|
| Memory service (`services/memory/`) | 541 | 37.0% | User memories — extraction, storage, retrieval | Some unit tests |
| Subscription management (`api/subscriptions/`) | 1,166 | 48.3% | Billing, tier resolution, upgrade/downgrade | Integration tests for billing math (Phase 6) |
| Services/integrations (`services/integrations/`) | 2,165 | 46.6% | Provider implementations, resource management | Partial unit tests |
| Supabase tools (`tools/supabase/`) | 550 | 30.4% | Database, storage, auth, edge functions | Zero dedicated tests |

### Tier 3: Lower Risk (Well-Defended Areas)

| Risk Area | Stmts | Coverage | Defense |
|-----------|-------|----------|---------|
| Workflow compilation pipeline | 708 | 89.8% | Unit + integration + E2E |
| Core runtime | 552 | 85.5% | Unit + integration |
| Database models | 1,264 | 88.7% | Unit + integration + E2E |
| Expression engine | 672 | 90.9% | Comprehensive unit tests |
| Trigger polling | 1,174 | 72.3% | Unit + integration pipeline tests |

---

## Action Plan — What Matters Next

### Priority 1: Cover Browser Service End-to-End (HIGH IMPACT)

**Current state:** 37% coverage. Only data layer (encryption, profiles, session storage) is tested.
**Gap:** 762 uncovered statements in `browser_service.py` (1,075 lines), `recording_service.py` (765 lines), `streaming_service.py` (369 lines), `pool_manager.py` (305 lines).

**Approach:** Create `tests/integration/services/test_browser_service.py` testing:
- Pool lifecycle (acquire → use → release → reuse)
- Session create → navigate → extract → close
- Recording start → capture → compress → store → retrieve
- Error paths (pool exhaustion, CDP disconnect, stale session)

**Blocker:** Requires real Chromium. Options: (a) Testcontainers with Playwright image, (b) mock CDP at protocol level, (c) test service orchestration logic with a stub browser.

**Effort:** 2-3 days | **Coverage impact:** +762 statements (~2.3% overall)

### Priority 2: Cover Supabase + Postgres Tools (MEDIUM IMPACT, LOW EFFORT)

**Current state:** Supabase 30.4% (zero tests), Postgres 31.8%.
**Gap:** 623 uncovered statements total.

**Approach:** Pure unit tests — these tools transform inputs to API calls. Mock the Supabase/Postgres client, test query building, error handling, response transformation.

**Effort:** 1 day | **Coverage impact:** +623 statements (~1.9% overall)

### Priority 3: Cover Integration/OAuth Endpoints (HIGH IMPACT)

**Current state:** 40.8% for `api/integrations/` (921 statements).
**Existing:** `test_integration_lifecycle.py` covers service layer (38 tests, 0 mocks).
**Gap:** Router endpoints — OAuth redirect/callback, resource browsing, multi-account management.

**Approach:** Integration tests calling router endpoints via `authenticated_client`. Mock OAuth provider HTTP responses only.

**Effort:** 1-2 days | **Coverage impact:** +545 statements (~1.6% overall)

### Priority 4: Cover Organization API (MEDIUM IMPACT)

**Current state:** 43.0% for `api/organizations/` (649 statements).
**Gap:** Team management, invitation flow, role permissions, billing association.

**Approach:** Integration tests with real DB. Test invitation → accept → role assignment → permission check.

**Effort:** 1 day | **Coverage impact:** +370 statements (~1.1% overall)

### Priority 5: CI Pipeline Tiers (MULTIPLIER)

**Structure:**
- **Tier 1 (every push):** Lint + type check + unit tests (~45s)
- **Tier 2 (every PR):** Integration tests (~3 min with containers)
- **Tier 3 (pre-merge):** E2E tests (~5 min)
- **Tier 4 (nightly):** Full suite + coverage report

**Effort:** 1 day | **Impact:** Makes all other testing improvements enforceable

### Priority 6: Move Misclassified Tests (LOW EFFORT, HOUSEKEEPING)

Move the 4 unit test files (13 tests) that use `db_engine` to `tests/integration/`. Quick cleanup for correctness.

**Effort:** 30 minutes

---

## The Testing Trophy — Current State

```
        +--------------------+
        |        E2E         |  9 files, 78 tests, 0 mocks
        |   Real Postgres    |  Testcontainers
        |   Real Redis       |  Gold standard layer
        +--------------------+
        |    Integration     |  40 files, 489 tests
        |   Real Postgres    |  Testcontainers (was SQLite)
        |  External mocks    |  All 16 mocked files audited — appropriate
        +--------------------+
        |       Unit         |  178 files, 3,306 tests
        |    Pure logic +    |  ~2,400 mocks (down 53% from start)
        |   billing + auth   |  Healthy after 2 rounds of pruning
        +--------------------+
```

---

## Metrics Dashboard

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Overall coverage | 62.4% | 70%+ | Need ~2,500 more covered statements |
| Unit test files | 178 | ~170 | Minor pruning remaining (node tests) |
| Unit mock sites | ~2,400 | <2,000 | Node tests are main candidates |
| Integration test files | 40 | 45+ | +5 needed (browser, OAuth, org, memory) |
| Integration tests | 489 | 550+ | |
| Integration mock quality | All appropriate | ✅ | **Achieved** |
| E2E test files | 9 | 10-12 | Nearly there |
| E2E tests | 78 | 80+ | Nearly there |
| Worker coverage | 61.4% | 65%+ | Close — chat.py orchestration remaining |
| Browser service coverage | 37.0% | 60%+ | Biggest single gap |
| Integration/OAuth coverage | 40.8% | 55%+ | Router endpoints remaining |
| Supabase tools coverage | 30.4% | 70%+ | Zero tests — low-hanging fruit |
| Skipped tests | 4 | 0-4 | All justified (external deps) |
| Misclassified tests | 4 files (13 tests) | 0 | Move to integration |
| SQLite in tests | 0 | 0 | ✅ **Eliminated** |

---

## Key Decisions Needed

1. **Browser service testing strategy:** Testcontainers with Chromium image, mock CDP at protocol level, or stub browser? The choice determines effort and coverage quality for the biggest gap (762 uncovered statements).

2. **Coverage target:** Is 70% the right goal? Reaching 70% from 62.4% requires ~2,500 more covered statements. Priorities 1-4 above would add ~2,300 statements, getting close.

3. **Node execution tests (109 mock sites in agent_node + image_gen_node):** Worth migrating to core integration tests using `compile_workflow → ainvoke`? ROI is debatable — they test real logic, just with mocked LLM responses.

4. **Supabase/Postgres tool test approach:** Pure unit tests with mocked clients (fast, catches transformation bugs) or integration tests hitting real Supabase/Postgres instances (higher confidence, more infrastructure)?

---

## Completed Phase History

### Phase 3: Worker Task Coverage ✅
Worker 0% → 61.4%. Created 6 unit test files covering error detection, payload builders, ownership checks, status transitions, idempotency, document pipeline.

### Phase 4: E2E Journey Expansion ✅
5 → 9 E2E files, 46 → 78 tests. Export/import, global variables, multi-node conditional execution, workflow management.

### Phase 5: Integration Layer Expansion ✅
Added 5 integration test files (+130 tests): org lifecycle, workflow execution, trigger processing, browser data layer, integration lifecycle, usage limits, polling pipeline.

### Phase 6: Unit Test Pruning + Integration Quality ✅
Deleted 63 unit test files. Migrated billing tests to integration. Audited all 40 integration files — reclassified 3 as pure wiring (deleted), confirmed remaining 16 mocked files use appropriate external-only mocks. Migrated all layers from SQLite to PostgreSQL Testcontainers.

---

*Date: 2026-03-27 | Author: Claude (fresh analysis)*
