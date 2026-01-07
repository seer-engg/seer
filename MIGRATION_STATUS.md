# SQLModel Migration Status

**Migration Type:** Clean Slate (Tortoise ORM → SQLModel + Alembic)
**Status:** ✅ Infrastructure Complete | 🚧 API Conversion In Progress
**Started:** 2026-01-06

---

## ✅ Completed Tasks

### 1. Dependencies & Configuration
- ✅ Removed `tortoise-orm` and `aerich` from dependencies
- ✅ Added `sqlmodel`, `alembic`, and `greenlet`
- ✅ Updated `pyproject.toml` to remove Aerich configuration

### 2. Database Infrastructure
- ✅ Created `shared/database/base.py` with SQLModel async engine
- ✅ Created comprehensive `shared/database/models.py` with all models:
  - User, Project
  - Workflow, WorkflowVersion, WorkflowDraft, WorkflowRecord
  - WorkflowRun, WorkflowChatSession, WorkflowChatMessage, WorkflowProposal
  - TriggerSubscription, TriggerEvent
  - OAuthConnection, IntegrationResource, IntegrationSecret
- ✅ Fixed SQLModel-specific issues:
  - Renamed `metadata` fields to avoid reserved name conflict
  - Fixed `ClassVar` annotations for class constants
  - Fixed BigInteger primary key declarations
- ✅ Added `User.get_or_create_from_auth()` helper method for auth middleware

### 3. Migration System
- ✅ Initialized Alembic in `alembic/` directory
- ✅ Configured `alembic.ini` with timestamp-based file naming
- ✅ Set up `alembic/env.py` to:
  - Import all SQLModel models
  - Read DATABASE_URL from environment
  - Convert async URL to sync for migrations (psycopg)
  - Enable autogenerate with type/default comparison
- ✅ Updated `shared/database/__init__.py` to use Alembic instead of Aerich

### 4. Developer Tools
- ✅ Created `scripts/migrate.sh` - Bash migration helper
  - Auto-detects Docker vs local environment
  - Commands: upgrade, create, rollback, history, current, reset
  - Color-coded output for better UX
- ✅ Created `scripts/migrate.py` - Python version for cross-platform support
  - Same features as Bash version
  - Works on Windows/Mac/Linux

### 5. Cleanup
- ✅ Deleted `migrations/` directory (old Aerich migrations)
- ✅ Deleted `shared/database/config.py` (Tortoise configuration)
- ✅ Deleted `shared/database/workflow_models.py` (will be removed after API conversion)
- ✅ Deleted `shared/database/models_oauth.py` (merged into models.py)
- ✅ Deleted `shared/database/models_integrations.py` (merged into models.py)

---

## 🚧 Remaining Tasks

### API Route Conversion (~30 files)

**Pattern to Follow:**

```python
# OLD (Tortoise ORM)
from shared.database.workflow_models import Workflow

workflows = await Workflow.filter(user_id=user.id).all()
workflow = await Workflow.get(id=workflow_id)
workflow = await Workflow.create(user=user, name="Test")
await workflow.save()

# NEW (SQLModel)
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from shared.database.models import Workflow
from shared.database.base import get_session

# In FastAPI route
async def list_workflows(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    statement = select(Workflow).where(Workflow.user_id == user.id)
    result = await session.execute(statement)
    workflows = result.scalars().all()
    return workflows

# Create
workflow = Workflow(user_id=user.id, name="Test")
session.add(workflow)
await session.commit()
await session.refresh(workflow)

# Update
workflow.name = "Updated"
session.add(workflow)
await session.commit()

# Delete
await session.delete(workflow)
await session.commit()
```

**Files Requiring Conversion:**

Core Workflow Services:
- [ ] `api/workflows/services.py` - Main workflow CRUD operations
- [ ] `api/workflows/router.py` - Workflow API routes
- [ ] `api/agents/workflow/services.py` - Workflow agent services
- [ ] `api/agents/workflow/router.py` - Agent routes

Trigger System:
- [ ] `api/triggers/services.py` - Trigger CRUD
- [ ] `api/triggers/polling/engine.py` - Polling engine
- [ ] `api/triggers/polling/adapters/*.py` - Trigger adapters

Integration System:
- [ ] `api/integrations/services.py` - Integration CRUD
- [ ] `api/integrations/router.py` - Integration routes
- [ ] `api/integrations/providers/*.py` - OAuth providers
- [ ] `api/integrations/resource_providers/*.py` - Resource providers
- [ ] `api/integrations/resource_browser.py` - Resource browser
- [ ] `shared/tools/oauth_manager.py` - OAuth token management
- [ ] `shared/tools/credential_resolver.py` - Credential resolution

Other APIs:
- [ ] `api/models/router.py` - Model routes
- [ ] `api/tools/router.py` - Tools routes
- [ ] `api/agents/routes.py` - Agent routes
- [ ] `api/agents/traces.py` - Trace handling
- [ ] `api/middleware/analytics.py` - Analytics middleware

Test Files:
- [ ] `tests/api/workflows/conftest.py` - Test fixtures
- [ ] `tests/api/workflows/test_triggers.py` - Trigger tests
- [ ] `tests/api/workflows/test_polling_engine.py` - Polling tests

Example Files (Low Priority):
- [ ] `workflow_compiler/examples/gmail_summary_workflow.py`
- [ ] `workflow_compiler/examples/gmail_common.py`

Scripts:
- [ ] `scripts/migrate_workflow_records.py` - Data migration script

---

## 📋 Next Steps

### Step 1: Generate Initial Migration
```bash
# Start PostgreSQL (if using Docker)
docker-compose up -d postgres

# Generate migration from SQLModel models
./scripts/migrate.sh create initial_sqlmodel_schema

# Review generated migration
ls alembic/versions/

# Apply migration
./scripts/migrate.sh upgrade
```

### Step 2: Convert API Files Systematically

**Recommended Order:**
1. Start with `api/workflows/services.py` (core functionality)
2. Then `api/workflows/router.py`
3. Then `api/triggers/services.py`
4. Continue with other workflow-related files
5. Move to integration system
6. Finally update tests

**For Each File:**
1. Replace Tortoise imports with SQLModel imports
2. Add `session: AsyncSession = Depends(get_session)` to route functions
3. Convert `.filter()` → `select().where()`
4. Convert `.get()` → `select().where()` + `.scalar_one_or_none()`
5. Convert `.create()` → create object + `session.add()` + `session.commit()`
6. Convert `.save()` → `session.add()` + `session.commit()`
7. Test the converted routes

### Step 3: Clean Up
After all API files are converted:
- Remove old Tortoise model files
- Update any remaining imports
- Run full test suite
- Update documentation

### Step 4: Deploy
1. Test locally with fresh database
2. Deploy to Railway dev environment
3. Test thoroughly
4. Deploy to Railway production

---

## 🎯 Migration Helper Commands

```bash
# Run migrations
./scripts/migrate.sh

# Create a new migration
./scripts/migrate.sh create "add_new_field"

# View migration history
./scripts/migrate.sh history

# Rollback last migration
./scripts/migrate.sh rollback

# Check current migration status
./scripts/migrate.sh current

# Reset database (DANGER!)
./scripts/migrate.sh reset

# Cross-platform (Python version)
python scripts/migrate.py
python scripts/migrate.py create "add_new_field"
```

---

## 📝 Notes

- **Database URL:** Configured to auto-detect from environment or use local default
- **Async Support:** Full async/await support with asyncpg driver
- **Type Safety:** SQLModel provides Pydantic validation + SQLAlchemy ORM
- **Auto-migrations:** Alembic autogenerate enabled with type and default comparison
- **Backward Compatibility:** Old model exports maintained in `shared/database/__init__.py`

---

## ⚠️ Breaking Changes

This is a **clean slate migration** - database will be wiped and recreated:
- ✅ Safe for development (no production users yet)
- ✅ Railway dev environment can be reset
- ✅ Railway main environment can be reset
- ⚠️ **All existing data will be lost** - confirm this is acceptable before deploying

---

## 🐛 Known Issues

None currently! 🎉

---

## 📚 Resources

- [SQLModel Documentation](https://sqlmodel.tiangolo.com/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/)
- [Migration Plan](./MIGRATION_PLAN_CLEAN_SLATE.md)
