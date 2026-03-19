# Memory Banks for Workflow Agent Nodes

## Objective

Add first-class, user-managed memory banks that can be attached to workflow `agent` nodes, while keeping the current Nexus memory experience working.

The feature needs to support:

1. Multiple memory banks per workspace/organization instead of one implicit global store.
2. Attaching a selected memory bank to a workflow `agent` node.
3. Letting an attached `agent` node retrieve from and write to that bank during execution.
4. Exposing a dedicated `memory` tool family for workflow `tool` nodes with basic CRUD operations, where the bank is passed explicitly.

## Recommended V1 Scope

Use one attached memory bank per `agent` node in V1.

Why:

- It matches the current requirement wording well enough.
- It keeps retrieval and write semantics unambiguous.
- It avoids hard questions like "which bank should writes go to?" or "do we search all banks and merge results?".

If multi-bank attachment is needed later, we can expand from `memory_bank_id` to `memory_bank_ids` plus a `write_bank_id`.

## Current State

### Existing memory implementation

- `src/seer/services/memory/user_memory.py` wraps Mem0 and assumes a single user-scoped store.
- `src/seer/api/memory/router.py` exposes CRUD/search endpoints directly against that single store.
- `src/seer/agents/nexus/agent.py` injects memory context into the Nexus system prompt.
- `src/seer/agents/nexus/tools/memory_tools.py` gives Nexus read-only memory search/profile tools.

### Existing workflow agent implementation

- `src/seer/core/nodes/agent_node.py` has no concept of memory attachment.
- The node can bind regular tools and execute them through LangChain `create_agent`.
- `core` currently depends downward on `tools`, and must not import `services`.

That last point is the main architectural constraint for this feature.

## Design Principles

1. Do not let `src/seer/core/` import `src/seer/services/memory/`.
2. Keep memory-bank ownership and CRUD in `api/services/database`, not in `core`.
3. Keep workflow execution memory access behind a runtime adapter injected from `services`.
4. Reuse the existing `tool` node for memory CRUD tools instead of creating a brand-new workflow node type.
5. Follow the repo's organization-ownership pattern: banks are workspace resources, not purely user-private records.
6. Keep legacy Nexus behavior by auto-resolving a default bank for the active organization/workspace.

## Proposed Architecture

### 1. Introduce a first-class `MemoryBank` model

Add a DB-backed memory-bank entity so users can manage multiple banks explicitly.

Recommended model:

- File: `src/seer/database/memory_models.py`
- Table: `memory_banks`
- Fields:
  - `id`
  - `organization`
  - `created_by_user`
  - `name`
  - `description` nullable
  - `status` (`active`, `deleted`)
  - `is_default`
  - `created_at`
  - `updated_at`
  - `last_used_at` nullable
  - `namespace_key`

Recommended public ID prefix:

- `mb_`

Ownership model:

- `organization` is the primary owner, matching workflows/knowledge-base direction in this repo.
- `created_by_user` is for audit, attribution, and permission nuances.
- every operation should be scoped to the active organization first, then membership/role checks decide access.

Why store `namespace_key`:

- It decouples Mem0 namespacing from DB IDs.
- It gives a safe migration path for the current legacy single-store memory.

### 2. Keep Mem0 isolation by bank namespace, not by post-filtering only

Do not implement multi-bank support only by storing `memory_bank_id` in metadata and filtering after search.

That would produce weak search quality because Mem0 retrieval would still search across the wrong bank first.

Recommended strategy:

- Keep one shared Mem0 vector collection/table for Seer memory.
- Store bank membership in each memory row's payload metadata.
- Scope every retrieval/update/delete by both organization and memory bank.
- Use `namespace_key` as the logical bank namespace value stored in metadata, not as a dynamic Postgres table name.

Recommended rules:

- Personal workspace default bank namespace: `org:<organization_id>:default`
- Named bank namespace: `org:<organization_id>:bank:<memory_bank_public_id>`

Also store these in Mem0 payload metadata for audit/debugging/filtering:

- `organization_id`
- `memory_bank_id`
- `memory_bank_namespace`
- `workflow_id` / `session_id` when relevant

### 2.1 Physical storage model

This is the actual database-level shape for the current Seer + Mem0 setup.

#### Postgres / pgvector

Current repo behavior:

- Seer creates one Mem0 client with one configured `collection_name` in `src/seer/services/memory/mem0_client.py`.
- In Mem0's pgvector backend, `collection_name` maps to a single Postgres table.
- Each memory is one row in that table.

That table stores roughly:

- `id`
- `vector`
- `payload JSONB`

The actual memory text and metadata live inside `payload`.

Implication for memory banks:

- we should **not** create one new Postgres table per bank
- we should keep one shared vector table and separate banks logically through payload metadata filters

#### SQLite history DB

Mem0 also keeps a separate history database, by default SQLite-backed, for memory mutation history.

Implication:

- memory CRUD history is not stored in the same pgvector table
- if we want bank-level history inspection later, the bank id/namespace should also be present in the memory payload and any app-level audit records we create

#### What this means for V1

For V1, each memory bank is:

- a row in Seer's own `memory_banks` table
- a logical partition inside Mem0's shared vector table, enforced by metadata filters

So the answer to "does Mem0 create new DB tables per bank?" should be **no** in our design.

### 3. Refactor the memory service into bank-aware operations

Replace the current single-store-only service shape with a bank-aware service layer.

Recommended service split:

- `MemoryBankService`
  - create/list/update/delete banks
  - resolve default bank for an organization
  - enforce organization ownership and role-based permissions
- `MemoryBankMemoryService`
  - add/search/get/update/delete memories within a resolved bank namespace
  - format prompt context for a given bank

The current `UserMemoryService` can remain temporarily as a compatibility wrapper around "default bank for the active personal organization".

### 4. Add a runtime memory adapter to `WorkflowRuntimeContext`

This is the key boundary that keeps `core` clean.

Recommended addition:

- Extend `src/seer/core/runtime/context.py`
- Add optional field like `memory_access`

Define a small protocol or dataclass interface in `core`, for example:

- `get_prompt_context(memory_bank_id, current_query, max_memories) -> str`
- `search(memory_bank_id, query, limit) -> list[dict]`
- `add(memory_bank_id, content, metadata, infer=True) -> dict | None`
- `get(memory_bank_id, memory_id) -> dict | None`
- `update(memory_bank_id, memory_id, content, metadata=None) -> dict | None`
- `delete(memory_bank_id, memory_id) -> bool`

Implementation lives in `services`, but `core` only sees the interface.

### 5. Attach memory to `agent` nodes via node inputs

Add a new reserved input on `AgentNode`:

- `memory_bank_id: string | null`

Recommended behavior:

- If absent: current behavior, no memory injection and no automatic memory tools.
- If present: inject prompt context and attach bank-scoped memory tools automatically.

Optional future flags:

- `memory_context_max_memories`
- `memory_context_query_template`

For V1, keep it simple and derive retrieval query from the rendered agent prompt.

### 6. Expose a `memory` tool family through the existing tool system

Do not create a new workflow node type for this.

Instead:

- Add new shared tools under `src/seer/tools/memory/`
- Register them in `src/seer/tools/__init__.py`
- Make them appear in the catalog as normal tools, grouped under "Memory" in the frontend

Recommended V1 tools:

- `memory_add`
- `memory_get`
- `memory_search`
- `memory_update`
- `memory_delete`

Minimum required by your request is `add/update/delete`, but `get/search` will be needed quickly in practice and keeps the tool family usable.

All of these should require `memory_bank_id` explicitly in their schema.

The tool implementation should call `context.memory_access`, not `services.memory` directly.

## Detailed Implementation Plan

### Phase 1. Data model and migration

#### 1.1 Add the DB model

Create `MemoryBank` ORM model and export it from `src/seer/database/__init__.py`.

Recommended constraints:

- `unique_together = (("organization", "name"),)`
- partial/semantic rule in service layer: one active default bank per organization

#### 1.2 Add migration

Create migration using:

```bash
uv run aerich migrate --name add_memory_banks
```

Migration should:

- create `memory_banks`
- backfill one default bank for existing users only if needed, or handle lazily on first access

Recommended approach:

- lazy creation on first access is simpler and safer than backfilling every user immediately

#### 1.3 Legacy compatibility

Preserve existing memories without bulk Mem0 rewrites.

Recommended compatibility rule:

- first resolved default bank for an org gets a stable namespace key
- for personal orgs, migration logic can map legacy user memory into that org's default bank view
- all legacy Nexus memory should remain reachable through the default bank path

Because old memories were written with `user_id` only, the migration/compatibility layer should:

- resolve the user's personal organization
- treat that personal org default bank as the owner of old user-scoped memories
- optionally backfill `organization_id` metadata lazily when a memory is updated or touched

This avoids an expensive full Mem0 rewrite while still moving the product model to org ownership.

### Phase 2. Service layer refactor

#### 2.1 Introduce bank management service

Add service methods:

- `list_banks(user)`
- `create_bank(user, organization, name, description=None)`
- `get_bank_for_org(user, organization, bank_id)`
- `get_default_bank(organization)`
- `get_or_create_default_bank(user, organization)`
- `update_bank(user, organization, bank_id, ...)`
- `delete_bank(user, organization, bank_id)`

Deletion policy for V1:

- soft-delete the bank record
- block deletion of the default bank unless another bank is promoted first
- require `OWNER`/`ADMIN` for destructive org-wide bank management

#### 2.2 Introduce bank-scoped memory service

Add service methods that require a bank object or bank id:

- `add_memory(user, bank, content, metadata=None, infer=True)`
- `create_manual_memory(user, bank, content, metadata=None)`
- `search(user, bank, query, limit=5, filters=None)`
- `get_all(user, bank)`
- `get_memory(user, bank, memory_id)`
- `update_memory(user, bank, memory_id, content, metadata=None)`
- `delete_memory(user, bank, memory_id)`
- `get_context_for_prompt(user, bank, current_query, max_memories=None)`

Important validation:

- service must verify organization ownership before every operation
- `get/update/delete(memory_id)` should verify the memory belongs to the requested bank
- every Mem0 operation should include `organization_id` and `memory_bank_id` in metadata/filter construction

#### 2.3 Keep `UserMemoryService` as a wrapper during transition

To minimize churn:

- keep `UserMemoryService`
- internally resolve the user's active/personal organization default bank and delegate to the bank-aware service

This lets Nexus and existing `/memory` routes continue working while new APIs land.

### Phase 3. API surface

#### 3.1 Add memory-bank management endpoints

Recommended new routes:

- `GET /api/memory/banks`
- `POST /api/memory/banks`
- `GET /api/memory/banks/{bank_id}`
- `PATCH /api/memory/banks/{bank_id}`
- `DELETE /api/memory/banks/{bank_id}`
- `POST /api/memory/banks/{bank_id}/set-default`

Recommended response model fields:

- `memory_bank_id`
- `name`
- `description`
- `is_default`
- `memory_count`
- `created_at`
- `updated_at`

#### 3.2 Add bank-scoped memory endpoints

Recommended routes:

- `GET /api/memory/banks/{bank_id}/items`
- `GET /api/memory/banks/{bank_id}/items/search`
- `POST /api/memory/banks/{bank_id}/items`
- `GET /api/memory/banks/{bank_id}/items/{memory_id}`
- `PUT /api/memory/banks/{bank_id}/items/{memory_id}`
- `DELETE /api/memory/banks/{bank_id}/items/{memory_id}`

#### 3.3 Keep current `/api/memory` routes for default-bank compatibility

Current routes should continue to work, but internally operate on the user's default bank.

That keeps:

- Nexus behavior unchanged
- existing UI/API consumers working
- rollout safer

Important ownership rule:

- these compatibility routes should operate against the caller's active workspace context when available
- if no active org is available in that API surface, fall back to the caller's personal organization

### Phase 4. Workflow runtime wiring

#### 4.1 Extend `WorkflowRuntimeContext`

Add optional field:

- `memory_access`

This should be populated in:

- `src/seer/services/workflows/execution.py`

It should not be created inside `core`.

#### 4.2 Add a services-side adapter implementation

Add something like:

- `src/seer/services/memory/runtime_adapter.py`

Responsibilities:

- resolve a bank by public id for the current runtime organization
- call the bank-aware memory service
- normalize errors into domain-safe exceptions or `None`/`False` responses as appropriate

This object is what `core` and `tools` will use at runtime.

### Phase 5. Agent node memory attachment

#### 5.1 Extend schema validation

Update `src/seer/core/schema/models.py` for `AgentNode`:

- accept optional `memory_bank_id`
- validate it is a string if present

Also update:

- `src/seer/api/workflows/services/catalog.py`

Add a field descriptor so the frontend can render a selector, for example:

- `name="memory_bank_id"`
- `kind="memory_bank_select"`

#### 5.2 Update reserved inputs in `agent_node.py`

Treat `memory_bank_id` as a reserved input, not an auxiliary prompt variable.

Recommended agent-node flow:

1. Extract `memory_bank_id` from inputs.
2. Render the prompt normally.
3. If `memory_bank_id` exists and `ctx.runtime_context.memory_access` exists:
   - fetch prompt memory context using the rendered prompt as the search query
   - prepend context to the prompt, similar to Nexus
4. Auto-bind bank-scoped memory tools to the agent
5. Execute agent as usual

#### 5.3 Auto-bind memory tools for attached banks

When `memory_bank_id` is set, the agent should be able to read/write memory without the user separately listing memory tools in `inputs.tools`.

Recommended memory tools for agent auto-binding:

- `recall_memories`
- `remember_fact`
- `update_memory`
- `delete_memory`

These can be LangChain tools created directly inside `agent_node.py` or via a helper in `core/nodes/agent_memory_tools.py`, but they must call `ctx.runtime_context.memory_access`.

Important:

- these are not necessarily the same shape as the shared workflow `memory_*` BaseTools
- agent convenience tools can be bank-bound and therefore should not require `memory_bank_id` in their arguments

This gives the best UX for the autonomous agent.

### Phase 6. Dedicated workflow memory tools

#### 6.1 Add shared tools under `src/seer/tools/memory/`

Recommended files:

- `src/seer/tools/memory/__init__.py`
- `src/seer/tools/memory/base.py` if shared helpers are needed
- `src/seer/tools/memory/add.py`
- `src/seer/tools/memory/get.py`
- `src/seer/tools/memory/search.py`
- `src/seer/tools/memory/update.py`
- `src/seer/tools/memory/delete.py`

#### 6.2 Tool schemas

Recommended parameter schemas:

`memory_add`

- `memory_bank_id` required
- `content` required
- `infer` optional
- `metadata` optional

`memory_update`

- `memory_bank_id` required
- `memory_id` required
- `content` required
- `metadata` optional

`memory_delete`

- `memory_bank_id` required
- `memory_id` required

`memory_get`

- `memory_bank_id` required
- `memory_id` required

`memory_search`

- `memory_bank_id` required
- `query` required
- `limit` optional

#### 6.3 Tool execution path

These tools should:

1. Require `context` and `context.memory_access`.
2. Fail clearly if runtime context or memory adapter is missing.
3. Delegate to the adapter.
4. Return normalized JSON-safe payloads.

This lets users use the existing `tool` node to manage memory directly in workflows.

### Phase 7. Nexus migration

#### 7.1 Keep current Nexus behavior

Nexus should keep using memory by default, but via the default bank.

Changes:

- `src/seer/agents/nexus/agent.py`
- `src/seer/agents/nexus/tools/memory_tools.py`

Update them to resolve the user's default bank before reading memory.

#### 7.2 Optional future enhancement

Later, Nexus chat sessions could choose a non-default bank, but that is not required for this workflow feature.

## Suggested File-Level Change List

### Database

- `src/seer/database/memory_models.py` new
- `src/seer/database/__init__.py`
- `migrations/...` new migration

### Services

- `src/seer/services/memory/user_memory.py` refactor or compatibility wrapper
- `src/seer/services/memory/memory_bank_service.py` new
- `src/seer/services/memory/runtime_adapter.py` new
- `src/seer/services/memory/__init__.py`
- `src/seer/services/workflows/execution.py`
- org access helpers patterned after existing org-scoped services

### API

- `src/seer/api/memory/router.py`
- any new request/response models if split out

### Core workflow runtime

- `src/seer/core/runtime/context.py`
- `src/seer/core/schema/models.py`
- `src/seer/core/nodes/agent_node.py`
- possibly `src/seer/core/nodes/agent_memory_tools.py` new helper

### Shared tools

- `src/seer/tools/memory/...` new
- `src/seer/tools/__init__.py`

### Nexus

- `src/seer/agents/nexus/agent.py`
- `src/seer/agents/nexus/tools/memory_tools.py`
- `src/seer/agents/nexus/utils.py` only if tool registration changes

### Tests

- `tests/unit/services/memory/...`
- `tests/unit/api/test_memory_router.py`
- `tests/unit/core/test_agent_node.py`
- `tests/unit/agents/nexus/test_memory_tools.py`
- new tool tests for `src/seer/tools/memory`
- integration tests for end-to-end workflow execution with memory

## Testing Plan

### 1. Service tests

Add tests for:

- creating/listing/updating/deleting banks
- default-bank auto-creation
- namespace resolution for default vs non-default bank
- organization ownership and membership enforcement
- bank-scoped memory CRUD
- compatibility mapping from personal-org default bank to legacy user-scoped memories

### 2. API tests

Add tests for:

- new bank management routes
- bank-scoped memory routes
- backward-compatible `/api/memory` default-bank behavior
- rejecting access to another user's bank

### 3. Core unit tests

Add tests in `tests/unit/core/test_agent_node.py` for:

- `memory_bank_id` schema validation
- prompt context injection when bank attached
- no memory injection when bank not attached
- auto-bound memory tools present only when bank attached
- failure path when bank is configured but runtime adapter is missing

Because this touches `src/seer/core/`, add both unit tests and integration/spec coverage, per repo rules.

### 4. Shared tool tests

Add tests for:

- `memory_add`
- `memory_get`
- `memory_search`
- `memory_update`
- `memory_delete`

Mock the runtime adapter, not Mem0 directly, at the tool layer.

### 5. Nexus regression tests

Add/adjust tests so existing Nexus memory tools still work through the default bank.

## Rollout Plan

### Step 1

Ship DB model, services, and backward-compatible API changes first.

### Step 2

Wire default-bank behavior into Nexus so nothing regresses.

### Step 3

Add workflow runtime adapter and `agent` node attachment.

### Step 4

Add shared memory tools for workflow `tool` nodes.

### Step 5

Expose the frontend UI:

- memory bank management screens
- memory bank selector in `agent` node config
- "Memory" tool group in tool picker

## Main Risks and Mitigations

### Risk 1. Breaking the `core -> services` boundary

Mitigation:

- keep memory execution behind `WorkflowRuntimeContext.memory_access`
- implement the adapter in `services`

### Risk 2. Mem0 retrieval leaking across banks

Mitigation:

- use organization + bank metadata filters on every query
- keep `memory_bank_namespace` as a stable logical namespace value
- do not model each bank as its own physical pgvector table

### Risk 3. Legacy memory loss or migration complexity

Mitigation:

- keep the default bank mapped to the legacy namespace
- do not require an immediate Mem0 data rewrite

### Risk 4. Agent prompt bloat

Mitigation:

- reuse `config.memory_context_max_memories`
- optionally add truncation/summarization rules later

### Risk 5. Unauthorized memory access across users

Mitigation:

- enforce organization membership and role checks in service methods
- verify bank ownership on every memory operation, not only at bank resolution time
- always include `organization_id` in Mem0 filter construction

## Recommended Decisions To Lock Before Implementation

1. V1 uses one `memory_bank_id` per `agent` node, not multiple banks.
2. Existing `/api/memory` routes remain and map to the user's default bank.
3. The "dedicated memory tool node" will be implemented as a `Memory` tool family under the existing `tool` node, not as a brand-new workflow node type.
4. Default bank keeps the legacy namespace so old Nexus memories remain accessible immediately.
5. Workflow `agent` nodes auto-bind bank-scoped memory tools when `memory_bank_id` is set.
6. Memory banks are organization-owned resources, with creator attribution kept separately.
7. In pgvector, banks remain logical partitions inside one shared Mem0 collection/table rather than one table per bank.

## Suggested Execution Order

1. Add `MemoryBank` model and migration.
2. Build bank-aware services and legacy wrapper compatibility.
3. Expand memory API with bank endpoints.
4. Update Nexus to resolve the default bank through the new services.
5. Add runtime memory adapter and wire it into workflow execution.
6. Extend `AgentNode` schema/catalog and implement prompt injection plus auto memory tools.
7. Add shared `memory_*` tools for workflow `tool` nodes.
8. Add frontend support.
9. Run full relevant test suites.

## Minimum Acceptance Criteria

- A user can create at least two memory banks.
- Those banks belong to the active organization/workspace, not just the raw user record.
- Existing Nexus memory still works through the default bank.
- A workflow `agent` node with `memory_bank_id` set can retrieve relevant context from that bank.
- That same `agent` node can add new memory to that bank during execution.
- A workflow `tool` node can call `memory_add`, `memory_update`, and `memory_delete` with an explicit `memory_bank_id`.
- Access is correctly isolated per organization and per bank.
