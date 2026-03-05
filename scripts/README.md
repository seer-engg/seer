> # Seer Utility Scripts

Maintenance and debugging scripts for Seer.

## Available Scripts

### `inspect_checkpoint_blob.py`
Debug LangGraph checkpoint state and trace persistence.

```bash
uv run python scripts/inspect_checkpoint_blob.py <thread_id>
```

### `lint-migrations.py`
Lint Aerich migrations for dangerous operations (runs in pre-commit).

```bash
uv run python scripts/lint-migrations.py
```

### `ensure_stripe_catalog.py`
Create/reactivate Stripe products and prices for subscription tiers.

```bash
uv run python scripts/ensure_stripe_catalog.py
```

### `test_imports.py`
Test Python imports to catch circular dependencies (pre-commit hook, currently disabled).

### `view_aws_parameters.sh`
View AWS Parameter Store configuration values.

```bash
./scripts/view_aws_parameters.sh <environment>
```
