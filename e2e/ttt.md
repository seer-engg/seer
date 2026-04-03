Now I have all the details. Here's a comprehensive explanation of all the e2e tests:

---

# E2E Test Suite Explanation

## Overview Architecture

`★ Insight ─────────────────────────────────────`
**Test Isolation Pattern:** Each test creates a fresh workflow via API (`createTestWorkflow`), runs in isolation, and deletes the workflow afterward (`deleteTestWorkflow`). This allows tests to run in parallel without interfering with each other.

**Defensive Skip Pattern:** Tests use `test.skip()` dynamically when preconditions aren't met (e.g., no edges exist). This prevents false failures but means some features aren't tested if the app state doesn't match expectations.
`─────────────────────────────────────────────────`

---

## 01-workflow-creation.spec.ts

**Purpose:** Tests the core workflow builder - creating workflows and adding blocks to the canvas.

| Test | Status | What It Does | Motive |
|------|--------|--------------|--------|
| **should create new workflow and add blocks** | ✅ | Searches for "LLM" in Build panel, clicks to add it, then adds an If/Else block. Verifies both appear on canvas. | Validates the fundamental user flow: creating a workflow and populating it with blocks. |
| **should connect blocks** | ⏭️ | Attempts to drag from one node's output handle to another's input handle to create an edge (connection). | Tests that users can wire blocks together to create a flow. |
| **should configure block and autosave** | ✅ | Adds LLM block, clicks it to select, opens Config tab, fills in a prompt field, verifies autosave indicator appears. | Ensures block configuration works and changes are automatically saved. |
| **should delete edge via hover button** | ⏭️ | Hovers over an edge, clicks the delete button that appears, verifies edge is removed. | Tests edge deletion UX - users should be able to remove connections easily. |

### Why Skipped:

| Test | Skip Reason |
|------|-------------|
| should connect blocks | `"Connection handles not found"` - Fresh workflows have no existing nodes with handles to connect. The test adds one LLM block but needs two positioned blocks to find both source and target handles. |
| should delete edge via hover button | `"No edges exist in workflow"` - Fresh workflows have no edges. Need to first create nodes AND connect them before testing edge deletion. |

---

## 02-workflow-execution.spec.ts

**Purpose:** Tests running workflows and viewing execution results.

| Test | Status | What It Does | Motive |
|------|--------|--------------|--------|
| **should execute workflow and view traces** | ✅ | Clicks Run button, handles any input dialog, switches to Executions tab, verifies execution entries appear. | Core feature: users must be able to run their workflows and see results. |
| **should display execution costs** | ⏭️ | Switches to Executions tab, looks for cost indicators like "$0.xx" or "Cost: xx". | Validates that LLM usage costs are tracked and displayed (billing transparency). |
| **should handle execution errors** | ✅ | Switches to Executions tab, looks for failed executions, clicks to view error details. | Ensures error states are properly displayed so users can debug issues. |

### Why Skipped:

| Test | Skip Reason |
|------|-------------|
| should display execution costs | `"No executions available to check costs"` - Fresh workflows have no execution history. Need past executions with LLM calls to show costs. |

---

## 03-oauth-integration.spec.ts

**Purpose:** Tests connecting third-party services (Gmail, etc.) via OAuth.

| Test | Status | What It Does | Motive |
|------|--------|--------------|--------|
| **should connect Gmail integration** | ⏭️ | Searches for Gmail in Build panel, clicks it, looks for Connect button, verifies OAuth popup URL contains "google". | Tests that OAuth flows initiate correctly for integrations. |
| **should disconnect integration** | ⏭️ | Goes to Settings page, looks for Integrations section, clicks Disconnect button, verifies success message. | Users should be able to revoke access to connected services. |

### Why Skipped:

| Test | Skip Reason |
|------|-------------|
| should connect Gmail integration | `"Gmail tool not found in build panel"` - The Gmail tool may not be visible in the search results (could be under a different name, category, or requires prior OAuth). |
| should disconnect integration | `"Integrations section not found"` - The Settings page may not have an Integrations section visible, or the selector doesn't match the current UI. |

---

## 04-trigger-setup.spec.ts

**Purpose:** Tests adding and configuring workflow triggers (what starts a workflow).

| Test | Status | What It Does | Motive |
|------|--------|--------------|--------|
| **should configure webhook trigger** | ✅ | Adds a webhook trigger, clicks the trigger node, switches to Config tab, verifies configuration options appear. | Webhooks are a primary trigger type - users need to configure them. |
| **should publish workflow with trigger** | ✅ | Finds or adds a trigger, clicks Publish button, handles confirmation, verifies success toast. | Publishing makes workflows live - critical feature. |
| **should configure Gmail trigger** | ✅ | Adds a Gmail email trigger, clicks the trigger node, verifies Gmail-specific config fields (Query, Label IDs). | Email triggers need specific configuration like filters. |

---

## 05-canvas-operations.spec.ts

**Purpose:** Tests React Flow canvas interactions (drag-drop, delete, branch handling).

| Test | Status | What It Does | Motive |
|------|--------|--------------|--------|
| **should drag-drop block from panel to canvas** | ✅ | Drags LLM block from Build panel to canvas (or clicks if drag fails), verifies node count increased. | Drag-drop is the primary way to add blocks - must work reliably. |
| **should delete node with confirmation** | ✅ | Adds LLM block, selects it, presses Delete key, handles confirmation dialog, verifies node count decreased. | Users need to remove blocks they no longer need. |
| **should handle if/else branch connections** | ✅ | Adds If/Else block, verifies it shows "True" and "False" branch labels. | If/Else blocks have multiple outputs - UI must clearly show branch labels. |
| **should handle for loop connections** | ✅ | Adds For Loop block, verifies it has multiple connection handles (input, loop, exit). | For loops need multiple outputs - validates correct handle structure. |

---

## 06-edge-insertion.spec.ts

**Purpose:** Tests the edge insertion feature - inserting a block between two connected nodes.

`★ Insight ─────────────────────────────────────`
This is a **UX power feature**: Instead of disconnecting edges, adding a block, and reconnecting, users can hover over an edge and click "Insert" to add a block in-between. The old edge splits into two new edges automatically.
`─────────────────────────────────────────────────`

| Test | Status | What It Does | Motive |
|------|--------|--------------|--------|
| **should show insert button on edge hover** | ⏭️ | Hovers over an edge, looks for Insert button to appear. | Validates hover UX reveals the insert option. |
| **should enter pending connection mode on insert click** | ⏭️ | Clicks Insert button, verifies "Select a block" indicator appears. | After clicking Insert, app enters a pending state waiting for block selection. |
| **should auto-expand build panel in pending state** | ⏭️ | Clicks Insert, verifies Build panel expands and search is visible. | UX optimization: Build panel should auto-open so user can pick a block. |
| **should insert block between nodes** | ⏭️ | Clicks Insert, adds If/Else block, verifies edge count increased by 1 (old edge removed, 2 new edges created). | Core functionality: block actually gets inserted. |
| **should cancel pending connection** | ⏭️ | Clicks Insert, then Cancel, verifies pending state clears. | Users should be able to abort the insertion. |
| **should NOT show insert button on trigger edges** | ⏭️ | Hovers over trigger edges (dashed lines), verifies Insert button doesn't appear. | Trigger edges are special - you can't insert blocks between trigger and first node. |
| **should preserve edge data (branch) during insertion** | ⏭️ | Looks for edges with "true/false" labels, verifies they exist. | When inserting on a branch edge, the branch label must be preserved. |
| **should mark canvas dirty and trigger autosave** | ⏭️ | Inserts a block, verifies autosave indicator appears. | Insertions should trigger save like any other edit. |

### Why ALL Skipped:

| Skip Reason |
|-------------|
| `"No edges exist in workflow"` - **All 8 tests skip because fresh workflows have no edges.** The tests need a workflow with at least two connected nodes to have edges to test. |

---

## 07-oauth-scope-upgrade.spec.ts

**Purpose:** Tests OAuth scope management - upgrading permissions when needed.

`★ Insight ─────────────────────────────────────`
**OAuth Scopes:** When you connect Gmail with "read-only" access, you can only read emails. If you later want to send emails, the app needs to request additional permissions (scope upgrade). This test suite validates that UX.
`─────────────────────────────────────────────────`

| Test | Status | What It Does | Motive |
|------|--------|--------------|--------|
| **should connect with readonly scope initially** | ✅ | Searches for Gmail, checks for scope selector, verifies Connect button is enabled. | Users should be able to choose minimal permissions initially. |
| **should upgrade scope from readonly to full access** | ✅ | Searches for Gmail, looks for "Missing scope" warning or "Upgrade" button. | When a tool needs more permissions, the UI should prompt for upgrade. |
| **should switch Google accounts** | ⏭️ | Goes to Settings, looks for Connected Accounts section, checks for account email and Switch button. | Users may need to change which Google account is connected. |
| **should consolidate scopes for multiple Google integrations** | ✅ | Searches for "google", verifies multiple Google tools (Gmail, Sheets, Drive) are visible. | When using multiple Google services, scopes should be combined into one OAuth request (better UX than multiple popups). |
| **should handle partial disconnect** | ⏭️ | Goes to Settings, looks for Integrations section with multiple connections, verifies Disconnect buttons exist. | Users should be able to disconnect one integration while keeping others. |

### Why Skipped:

| Test | Skip Reason |
|------|-------------|
| should switch Google accounts | `"Connected accounts section not found"` - Settings page may not have the expected section, or the selector doesn't match. |
| should handle partial disconnect | `"Integrations section not found"` - Same issue - Settings UI structure differs from test expectations. |

---

## Summary: Why Tests Skip

| Root Cause | Tests Affected | Solution |
|------------|----------------|----------|
| **Fresh workflows have no edges** | 10 tests (all of 06-edge-insertion + 2 from 01-workflow-creation) | Tests should first create nodes AND connect them before testing edge features |
| **Settings page UI doesn't match selectors** | 4 tests (03-oauth, 07-oauth) | Update selectors to match actual Settings page structure |
| **No execution history** | 1 test (02-workflow-execution) | Run a workflow first or use a workflow with existing executions |
| **Gmail tool not found** | 1 test (03-oauth-integration) | Check if Gmail is under a different testid or category |

---

## Test Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Coverage Intent** | Good | Tests cover core features: creation, execution, triggers, OAuth, canvas ops |
| **Isolation** | Excellent | Each test creates/deletes its own workflow |
| **Resilience** | Mixed | Defensive skips prevent false failures but also hide missing coverage |
| **Edge Feature Testing** | Poor | 100% skip rate due to missing setup - these tests can't actually run |
| **Selector Stability** | Needs Work | Several tests fail due to selector mismatches with current UI |

Would you like me to suggest improvements to fix the skipped tests so they actually run?