# E2E Subscription Tests

Comprehensive end-to-end tests for the 14-day free trial implementation.

## Overview

These tests validate the complete subscription lifecycle using **real Stripe test mode** with test clocks for time manipulation.

### Test Coverage

- ✅ **Onboarding Trial** (5 tests) - Trial creation during onboarding flow
- ✅ **Trial Expiration** (5 tests) - Trial → Active conversion after 14 days
- ✅ **Trial Cancellation** (7 tests) - Cancellation scenarios and user blocking
- ✅ **Billing Cycles** (6 tests) - Recurring monthly/annual billing
- ✅ **Edge Cases** (10 tests) - Error handling, webhooks, sync recovery

**Total: 33 tests covering all critical subscription behaviors**

## Prerequisites

### 1. Stripe Test Mode Setup

```bash
# Set Stripe test API keys in .env
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

### 2. Stripe CLI (for webhook testing)

```bash
# Install Stripe CLI
# macOS
brew install stripe/stripe-cli/stripe

# Linux
wget https://github.com/stripe/stripe-cli/releases/download/v1.19.0/stripe_1.19.0_linux_x86_64.tar.gz
tar -xvf stripe_1.19.0_linux_x86_64.tar.gz
sudo mv stripe /usr/local/bin

# Login to Stripe
stripe login
```

### 3. Python Environment

```bash
# Install dependencies
uv sync

# Ensure test database is configured
# SQLite in-memory is used automatically for tests
```

## Running Tests

### Basic Execution

```bash
# Run all subscription E2E tests
uv run pytest tests/e2e/subscriptions/ -v

# Run specific test file
uv run pytest tests/e2e/subscriptions/test_trial_expiration.py -v

# Run specific test
uv run pytest tests/e2e/subscriptions/test_trial_expiration.py::test_trial_converts_to_active_after_14_days -v
```

### With Stripe CLI (for webhook testing)

```bash
# Terminal 1: Start webhook forwarding
stripe listen --forward-to http://localhost:8000/api/subscriptions/webhooks/stripe

# Terminal 2: Export webhook secret
export STRIPE_WEBHOOK_SECRET=$(stripe listen --print-secret)

# Terminal 3: Run tests
uv run pytest tests/e2e/subscriptions/ -v --tb=short
```

### With Coverage

```bash
# Run with coverage report
uv run pytest tests/e2e/subscriptions/ \
  --cov=src/seer/api/subscriptions \
  --cov-report=html \
  --cov-report=term

# Open coverage report
open htmlcov/index.html
```

### Parallel Execution

```bash
# Run tests in parallel (be mindful of Stripe rate limits)
uv run pytest tests/e2e/subscriptions/ -n 4 --dist loadscope
```

### Filter by Test Markers

```bash
# Run only critical tests
uv run pytest tests/e2e/subscriptions/ -k "critical" -v

# Run only trial expiration tests
uv run pytest tests/e2e/subscriptions/ -k "expiration" -v

# Run only cancellation tests
uv run pytest tests/e2e/subscriptions/ -k "cancel" -v
```

## Test Structure

### Directory Layout

```
tests/e2e/subscriptions/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures
├── test_onboarding_trial.py     # Onboarding + trial creation (5 tests)
├── test_trial_expiration.py     # Trial → Active conversion (5 tests)
├── test_trial_cancellation.py   # Cancellation scenarios (7 tests)
├── test_billing_cycles.py       # Monthly billing cycles (6 tests)
├── test_edge_cases.py           # Error scenarios (10 tests)
└── helpers/
    ├── __init__.py
    ├── stripe_helpers.py        # Test clock utilities
    ├── webhook_helpers.py       # Webhook verification
    └── assertions.py            # Custom assertions
```

### Key Fixtures

| Fixture | Description |
|---------|-------------|
| `user_with_payment_method` | User with Stripe customer and payment method attached |
| `stripe_test_clock` | Test clock manager for time manipulation |
| `trial_subscription_setup` | Full trial subscription with test clock |
| `webhook_verifier` | Helper for verifying webhook delivery |
| `authenticated_subscription_client` | API client with auth headers |

## Critical Tests

### ⭐ Most Important Tests

These tests validate core requirements:

1. **`test_trial_converts_to_active_after_14_days`**
   - Verifies trial ends after 14 days
   - First payment collected
   - Status changes to active

2. **`test_canceled_subscription_ends_at_trial_end`**
   - User cancels during trial
   - No charge at trial end
   - Reverted to free tier

3. **`test_second_monthly_billing_cycle`**
   - Recurring billing works
   - Multiple cycles process correctly

## Test Environment

### Stripe Test Clocks

Tests use Stripe test clocks to simulate time progression:

```python
# Example: Advance time by 14 days
stripe_test_clock.advance_clock(clock_id, days=14, hours=1)
```

### Test Cards

Common test cards used:

| Card | Purpose |
|------|---------|
| `4242424242424242` | Successful payment |
| `4000000000000002` | Declined payment |
| `4000000000009995` | Insufficient funds |
| `4000002500003155` | Requires 3DS authentication |

### Database

- Tests use SQLite in-memory database
- Fresh database for each test
- Automatic cleanup after tests

## Debugging

### Verbose Output

```bash
# Show all output (print statements, logs)
uv run pytest tests/e2e/subscriptions/ -v -s

# Show full stack traces
uv run pytest tests/e2e/subscriptions/ -v --tb=long

# Stop at first failure
uv run pytest tests/e2e/subscriptions/ -v -x
```

### Inspecting Stripe Objects

```bash
# View test customer in Stripe dashboard
stripe customers list --limit 5

# View test subscriptions
stripe subscriptions list --limit 5

# View test invoices
stripe invoices list --limit 5
```

### Logs

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uv run pytest tests/e2e/subscriptions/ -v -s
```

## Troubleshooting

### Common Issues

1. **Stripe API Key Not Found**
   ```
   Error: config.stripe_secret_key is None
   ```
   **Fix:** Set `STRIPE_SECRET_KEY` in `.env` file

2. **Webhook Signature Verification Failed**
   ```
   Error: Invalid signature
   ```
   **Fix:** Ensure Stripe CLI is running and `STRIPE_WEBHOOK_SECRET` is set

3. **Test Timeout**
   ```
   Error: asyncio timeout after 10s
   ```
   **Fix:** Increase timeout in `webhook_verifier` or test clock operations

4. **Rate Limit Exceeded**
   ```
   Error: 429 Too Many Requests
   ```
   **Fix:** Run tests sequentially or reduce parallelism

### Clean Up Test Data

```bash
# Delete test customers (after tests complete)
stripe customers list --limit 100 | grep test_ | xargs -I {} stripe customers delete {}

# Delete test subscriptions
stripe subscriptions list --limit 100 | grep test_ | xargs -I {} stripe subscriptions cancel {}

# Delete test clocks
# Test clocks are automatically cleaned up by fixtures
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Subscription Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync

      - name: Run E2E tests
        env:
          STRIPE_SECRET_KEY: ${{ secrets.STRIPE_SECRET_KEY_TEST }}
          STRIPE_PUBLISHABLE_KEY: ${{ secrets.STRIPE_PUBLISHABLE_KEY_TEST }}
        run: |
          uv run pytest tests/e2e/subscriptions/ -v \
            --cov=src/seer/api/subscriptions \
            --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Success Criteria

Tests pass if:

- ✅ All 33 tests passing
- ✅ Code coverage >85% for subscription module
- ✅ All critical tests (⭐) passing
- ✅ No flaky tests (must pass 10/10 runs)
- ✅ Webhooks processed within 5 seconds
- ✅ DB always in sync with Stripe

## Resources

- [Stripe Test Mode](https://stripe.com/docs/testing)
- [Stripe Test Clocks](https://stripe.com/docs/billing/testing/test-clocks)
- [Stripe CLI](https://stripe.com/docs/stripe-cli)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)

## Support

For issues or questions:

1. Check Stripe Dashboard → Developers → Logs
2. Review test output with `-v -s` flags
3. Check `tests/e2e/subscriptions/helpers/` for utility functions
4. Verify Stripe webhook forwarding is active
