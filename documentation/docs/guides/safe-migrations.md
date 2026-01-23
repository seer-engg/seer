# Safe Database Migration Practices

## Golden Rules

1. **Always add columns as nullable or with defaults**
   ```sql
   -- ❌ Dangerous
   ALTER TABLE users ADD COLUMN email VARCHAR(255) NOT NULL;

   -- ✅ Safe
   ALTER TABLE users ADD COLUMN email VARCHAR(255) NULL;
   -- Or with default:
   ALTER TABLE users ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE;
   ```

2. **Never drop columns immediately**
   - Phase 1: Stop using column in code
   - Phase 2 (next release): Drop column in migration

3. **Test migrations in local environment first**
   ```bash
   uv run aerich upgrade
   # Test app functionality
   # If issues, rollback: uv run aerich downgrade
   ```

4. **Check migration SQL before committing**
   - Read the generated SQL in migration files
   - Verify it matches your intent
   - Look for unintended side effects

5. **Keep data migrations small and batched**
   - Don't update millions of rows in one transaction
   - Consider writing a background task instead

## Common Scenarios

### Adding a Required Field

**Wrong**:
```python
# Model change
class User(Model):
    email: str  # New required field

# Migration will fail on existing data
```

**Right**:
```python
# Step 1: Add as nullable
class User(Model):
    email: str | None = None

# Step 2 (later): Backfill data via script/task
# Step 3 (even later): Make it required
```

### Renaming a Column

**Wrong**:
```sql
ALTER TABLE users RENAME COLUMN old_name TO new_name;
-- Old code breaks immediately
```

**Right**:
- Don't rename in database - use model aliasing instead
- Or accept that column name differs from model field name

### Removing a Column

**Phase 1** (Release N):
- Remove all code references to the column
- Deploy and verify

**Phase 2** (Release N+1):
- Create migration to drop column
- Deploy

## Testing Migrations

Always test locally:
```bash
# Apply migration
uv run aerich upgrade

# Run tests
uv run pytest

# If something breaks, rollback
uv run aerich downgrade

# Fix migration, try again
```

## Migration Commands Reference

```bash
# Create new migration after model changes
uv run aerich migrate

# Apply pending migrations
uv run aerich upgrade

# Rollback last migration
uv run aerich downgrade

# View migration history
uv run aerich history

# Check current migration status
uv run aerich heads
```
