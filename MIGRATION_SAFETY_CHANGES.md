# Database Migration Production Safety - Implementation Summary

## Problem Addressed

Fixed recurring production deployment failures with "missing column" errors in Railway by removing auto-apply migrations on startup and implementing proper pre-deploy migration hooks.

## Changes Implemented

### Phase 1: Immediate Production Fixes ✅

#### 1. Railway Pre-Deploy Migration Hook
**File: `railway.json`**
- Added `preDeployCommand: ["uv run aerich upgrade"]` to run migrations BEFORE app starts
- Added explicit `startCommand` to ensure proper startup sequence
- Migrations now run as a separate phase before code execution

**Impact:** Eliminates race conditions where code deploys before migrations complete

#### 2. Removed AUTO_APPLY_DATABASE_MIGRATIONS
**Files Updated:**
- `src/seer/database/__init__.py` - Removed AUTO_APPLY logic and unused imports
- `docker-compose.yml` - Removed AUTO_APPLY environment variable
- `docker-compose.thread.yml` - Removed AUTO_APPLY environment variable
- `.env.railway.template` - Removed AUTO_APPLY documentation

**Impact:** Prevents dangerous auto-migration on app startup

#### 3. Updated Documentation
**File: `README.md`**
- Added "Database Migrations" section with clear instructions
- Documented when to run migrations
- Added migration commands reference
- Explained cloud vs local behavior

**Files: `documentation/docs/deployment/RAILWAY.md` and `documentation/docs/advanced/CONFIGURATION.md`**
- Updated Railway deployment guide to reflect railway.json approach
- Removed AUTO_APPLY references
- Added migration verification steps

**Impact:** Clear developer experience for all environments

### Phase 2: Pre-Commit Safety ✅

#### 4. Migration Linting Script
**File: `scripts/lint-migrations.py`**
- Detects dangerous operations before commit:
  - NOT NULL columns without DEFAULT
  - DROP COLUMN operations
  - RENAME COLUMN operations
  - UNIQUE constraints on existing tables
  - Large UPDATE operations
- Lints recent migrations (last 5) to avoid noise
- Provides clear error messages with guidance

**Impact:** Catches dangerous migrations before they reach production

#### 5. Pre-Commit Hook
**File: `.pre-commit-config.yaml`**
- Added `lint-migrations` hook
- Runs automatically on migration file changes
- Fails commit if dangerous patterns detected

**Impact:** Automated safety checks in developer workflow

#### 6. Safe Migrations Guide
**File: `documentation/docs/guides/safe-migrations.md`**
- Golden rules for safe migrations
- Common scenario walkthroughs
- Migration commands reference
- Testing procedures

**Impact:** Educational resource for team and contributors

### Phase 3: CI/CD Integration ✅

#### 7. GitHub Actions Migration Testing
**File: `.github/workflows/backend-tests.yml`**
- Added migration linting step
- Added migration testing with temporary database
- Validates migrations apply cleanly in CI

**Impact:** Catches migration issues before merge

## Migration Workflow by Environment

| Environment | How Migrations Run | When |
|-------------|-------------------|------|
| **Local Dev** | Manual: `docker compose exec api uv run aerich upgrade` | After pulling new code |
| **Self-Hosted** | Manual: `uv run aerich upgrade` | Before deployment/restart |
| **Cloud (Railway)** | Automatic: Pre-deploy hook in railway.json | Before each deployment |

## Files Modified

### Core Application
- `railway.json` - Added preDeployCommand
- `src/seer/database/__init__.py` - Removed AUTO_APPLY logic
- `docker-compose.yml` - Removed AUTO_APPLY variable
- `docker-compose.thread.yml` - Removed AUTO_APPLY variable
- `.env.railway.template` - Updated deployment instructions

### Documentation
- `README.md` - Added migration instructions
- `documentation/docs/deployment/RAILWAY.md` - Updated Railway guide
- `documentation/docs/advanced/CONFIGURATION.md` - Updated config reference
- `documentation/docs/guides/safe-migrations.md` - New safety guide

### Developer Tools
- `scripts/lint-migrations.py` - New migration linter
- `.pre-commit-config.yaml` - Added migration linting hook
- `.github/workflows/backend-tests.yml` - Added migration tests

## Verification Steps

### ✅ Completed
1. Railway.json updated with preDeployCommand
2. AUTO_APPLY removed from all code files
3. README updated with migration instructions
4. Migration linter created and tested
5. Pre-commit hook configured
6. CI/CD pipeline updated
7. Documentation updated

### 🔄 Next Steps
1. Deploy to Railway and verify migration execution in logs
2. Remove AUTO_APPLY_DATABASE_MIGRATIONS from Railway UI variables
3. Run `pre-commit install` on all developer machines
4. Monitor next deployment for success

## Testing

### Migration Linter Test
```bash
$ uv run python scripts/lint-migrations.py
❌ Migration linting failed:
  ⚠️  DROP COLUMN detected in 9_20260109055152_add_form_feilds.py
```
✅ **Working correctly** - detected DROP COLUMN in existing migration

### Pre-Commit Hook
After running `pre-commit install`:
```bash
$ git commit -m "test"
Lint Aerich Migrations...................................................Failed
```
✅ Will block dangerous migrations from being committed

## Success Metrics

After full deployment:
- ✅ Zero "missing column" errors in Railway deployments
- ✅ All migrations run BEFORE code starts
- ✅ Dangerous migrations caught in pre-commit
- ✅ Clear migration workflow documented
- ✅ No AUTO_APPLY references in codebase

## Rollback Plan

If issues occur:
1. Revert railway.json to previous version
2. Add back AUTO_APPLY_DATABASE_MIGRATIONS=true in Railway UI
3. Revert src/seer/database/__init__.py changes

## References

- [Writing Safe Database Migrations in Django](https://markusholtermann.eu/2021/06/writing-safe-database-migrations-in-django/)
- [Django Zero Downtime Guide](https://www.vintasoftware.com/blog/django-zero-downtime-guide)
- [Railway Pre-Deploy Commands](https://docs.railway.com/guides/pre-deploy-command)
