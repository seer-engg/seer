# Nexus Recursion Limit & User Settings Implementation

## Summary

Implemented configurable recursion limits for Nexus chat to prevent LangGraph recursion limit errors (previously hitting the default 25 step limit). Users can now configure their max agent steps via a new settings API, with a global default of 75 steps.

## Changes Made

### 1. Configuration (`src/seer/config.py`)
- Added `nexus_max_agent_steps: int = Field(default=75)`
- Global default increased from 25 to 75 to accommodate complex workflows

### 2. Database Schema (`src/seer/database/models.py`)
- Added `UserSettings` model with:
  - `max_agent_steps: int` (nullable) - per-user override
  - `preferences: JSONField` - extensible key-value store
- Added `UserSettingsPublic` Pydantic model for API responses
- Updated `src/seer/database/__init__.py` to export new models

### 3. Database Migration (`migrations/002_add_user_settings.sql`)
```sql
CREATE TABLE user_settings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    max_agent_steps INTEGER,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_user_settings_user_id ON user_settings(user_id);
```

### 4. Settings API (`src/seer/api/users/settings.py`)
**Endpoints:**
- `GET /users/me/settings` - Get current user settings (auto-creates if not exists)
- `PATCH /users/me/settings` - Update settings with validation:
  - `max_agent_steps`: 10-200 (enforced)
  - `preferences`: arbitrary JSON (merged, not replaced)

**Features:**
- Auto-creates default settings on first access
- Validates max_agent_steps bounds (10-200)
- Merges preferences instead of replacing

### 5. Chat Router Updates (`src/seer/api/agents/workflow/router.py`)
**Modified functions:**
- `chat_with_workflow_endpoint()` (line ~229)
- `resume_chat_endpoint()` (line ~471)

**Logic added:**
```python
# Get user settings for max steps
try:
    user_settings = await UserSettings.get(user=user)
    max_agent_steps = user_settings.max_agent_steps or config.nexus_max_agent_steps
except DoesNotExist:
    max_agent_steps = config.nexus_max_agent_steps

config_dict = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": max_agent_steps,  # NEW
}
```

### 6. Enhanced Logging (`src/seer/api/agents/workflow/chat_services.py`)
- Added logging in `invoke_with_timeout()` to track recursion_limit usage:
```python
recursion_limit = config_dict.get("recursion_limit", 25)
logger.info("Invoking agent thread=%s recursion_limit=%d timeout=%ds",
            thread_id or 'unknown', recursion_limit, int(timeout))
```

### 7. API Router Registration (`src/seer/api/main.py`)
- Imported and registered `settings_router`
- Available at `/users/me/settings`

### 8. Tests (`tests/integration/api/`)
**Created:**
- `test_user_settings.py` - Settings API integration tests (6 tests)
- `test_chat_recursion_limit.py` - Recursion limit logic tests (3 tests)

**Coverage:**
- ✅ Default settings creation
- ✅ Update max_agent_steps
- ✅ Validation bounds (10-200)
- ✅ Preferences update and merge
- ✅ Recursion limit fallback logic
- ✅ User override behavior

All 9 tests pass.

## API Usage Examples

### Get Settings
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/users/me/settings
```

Response:
```json
{
  "max_agent_steps": null,
  "preferences": {},
  "updated_at": "2026-01-23T00:00:00Z"
}
```

### Update Max Agent Steps
```bash
curl -X PATCH \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"max_agent_steps": 100}' \
  http://localhost:8000/users/me/settings
```

### Update Preferences
```bash
curl -X PATCH \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"preferences": {"theme": "dark", "notifications": true}}' \
  http://localhost:8000/users/me/settings
```

## Behavior

### Before
- Fixed 25 step recursion limit (LangGraph default)
- Complex workflows failed with recursion errors
- No user configurability

### After
- Default 75 steps (global config)
- Per-user override (10-200 range)
- Falls back to global default if user setting is null
- Logged for observability

## Migration Steps

1. **Run migration:**
   ```bash
   uv run aerich migrate
   uv run aerich upgrade
   ```

2. **Verify:**
   ```bash
   # Check settings API
   curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/users/me/settings

   # Update and verify
   curl -X PATCH -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"max_agent_steps": 100}' \
     http://localhost:8000/users/me/settings
   ```

3. **Test chat:**
   - Send complex workflow chat that previously failed
   - Check logs for `recursion_limit` value
   - Verify no 500 errors

## Files Modified

**Core Implementation:**
- `src/seer/config.py` - Added config field
- `src/seer/database/models.py` - Added UserSettings model
- `src/seer/database/__init__.py` - Exported new models
- `src/seer/api/users/settings.py` - New settings API (NEW FILE)
- `src/seer/api/users/__init__.py` - Module init (NEW FILE)
- `src/seer/api/agents/workflow/router.py` - Apply recursion_limit
- `src/seer/api/agents/workflow/chat_services.py` - Enhanced logging
- `src/seer/api/main.py` - Register router
- `migrations/002_add_user_settings.sql` - Migration (NEW FILE)

**Tests:**
- `tests/integration/api/__init__.py` - Test module (NEW FILE)
- `tests/integration/api/test_user_settings.py` - Settings tests (NEW FILE)
- `tests/integration/api/test_chat_recursion_limit.py` - Logic tests (NEW FILE)

## Configuration

### Environment Variables
None required - uses sensible defaults.

Optional override via `.env`:
```bash
NEXUS_MAX_AGENT_STEPS=100  # Change global default from 75
```

### User Settings
Users can override via API (10-200 range enforced).

## Frontend Integration (Separate Repo)

Settings API is ready for frontend integration:

1. **GET** `/users/me/settings` to load current settings
2. **PATCH** `/users/me/settings` with `{max_agent_steps: number}` to update
3. Display in settings UI with:
   - Slider/input (10-200 range)
   - Default badge showing global config value
   - Save button calling PATCH endpoint

## Next Steps

- [ ] Run migration in production
- [ ] Monitor logs for recursion_limit usage patterns
- [ ] Adjust default (75) based on real usage
- [ ] Add frontend settings UI (separate PR)
- [ ] Consider workflow-level overrides (future enhancement)

## Validation

Run all tests:
```bash
uv run pytest tests/integration/api/ -v
```

Expected output: 9 passed
