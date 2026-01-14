> **[← Back to Documentation Index](../docs/README.md)**

# Seer Utility Scripts

This directory contains utility scripts for debugging, deployment, and maintenance.

## Available Scripts

### inspect_checkpoint_blob.py

**Purpose**: Diagnostic tool for inspecting LangGraph checkpoint data structures

**Usage**:
```bash
python scripts/inspect_checkpoint_blob.py <thread_id>
```

**Example**:
```bash
python scripts/inspect_checkpoint_blob.py run_7
```

**What it does**:
- Inspects checkpoint channel_values and full checkpoint structure
- Searches for trace keys in checkpoints
- Shows channel versions, metadata, and pending writes
- Useful for debugging checkpointer state and trace key persistence

**When to use**:
- Debugging workflow execution state
- Investigating checkpoint persistence issues
- Understanding LangGraph state structure
- Troubleshooting agent conversation history

---

### railway-migrate.sh

**Purpose**: Runs database migrations on Railway deployments

**Usage**:
```bash
./scripts/railway-migrate.sh
```

**What it does**:
- Executes `aerich upgrade` to apply pending database migrations
- Used during Railway deployment initialization
- Ensures database schema is up to date

**When to use**:
- During Railway deployment setup
- When manually applying migrations to cloud deployments
- After database schema changes

---

### ensure_stripe_catalog.py

**Purpose**: Ensure Stripe products and prices exist for all subscription tiers

**Usage**:
```bash
python scripts/ensure_stripe_catalog.py
```

**What it does**:
- Creates or reactivates Stripe products for each tier (Pro, Pro+, Ultra)
- Creates or reactivates recurring prices (monthly and annual) with lookup keys
- Prints the lookup key to price ID mapping after completion

**When to use**:
- Setting up Stripe for a new environment
- Rotating Stripe keys and recreating catalog objects
- Validating Stripe catalog consistency before deployments

---

## Related Documentation

- [Configuration Reference](../docs/advanced/CONFIGURATION.md)
- [Railway Deployment](../docs/deployment/RAILWAY.md)
- [Main README](../README.md)
