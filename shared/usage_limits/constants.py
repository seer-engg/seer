"""
Numerical constants for usage limits across subscription tiers.

These values define the hard limits for each feature dimension across
Self-Hosted, Cloud Free, Cloud Pro, Cloud Pro+, and Cloud Ultra tiers.

Convention:
  - -1 means unlimited
  - 0 means disabled/not allowed
"""

# ============================================================================
# Workflow Limits
# ============================================================================

WORKFLOWS_SELF_HOSTED = -1  # Unlimited
WORKFLOWS_FREE = 3
WORKFLOWS_PRO = -1  # Unlimited
WORKFLOWS_PRO_PLUS = -1  # Unlimited
WORKFLOWS_ULTRA = -1  # Unlimited

# ============================================================================
# Workflow Run Limits (Monthly)
# ============================================================================

RUNS_MONTHLY_SELF_HOSTED = -1  # Unlimited
RUNS_MONTHLY_FREE = 100
RUNS_MONTHLY_PRO = 1_000_000
RUNS_MONTHLY_PRO_PLUS = 5_000_000
RUNS_MONTHLY_ULTRA = 20_000_000

# ============================================================================
# Chat AI Message Limits (Total per User, across all workflows)
# ============================================================================

CHAT_MESSAGES_TOTAL_SELF_HOSTED = 0  # Disabled
CHAT_MESSAGES_TOTAL_FREE = 50  # 50 total messages across all workflows
CHAT_MESSAGES_TOTAL_PRO = -1  # Unlimited
CHAT_MESSAGES_TOTAL_PRO_PLUS = -1  # Unlimited
CHAT_MESSAGES_TOTAL_ULTRA = -1  # Unlimited

# ============================================================================
# Account Day Limits
# ============================================================================

ACCOUNT_DAY_LIMIT_SELF_HOSTED = -1  # Unlimited
ACCOUNT_DAY_LIMIT_FREE = 14  # 14-day trial
ACCOUNT_DAY_LIMIT_PRO = -1  # No limit
ACCOUNT_DAY_LIMIT_PRO_PLUS = -1  # No limit
ACCOUNT_DAY_LIMIT_ULTRA = -1  # No limit

# ============================================================================
# Polling Frequency Limits (Minimum Interval in Seconds)
# ============================================================================

POLL_MIN_INTERVAL_SELF_HOSTED = 1  # 1 second minimum
POLL_MIN_INTERVAL_FREE = 900  # 15 minutes
POLL_MIN_INTERVAL_PRO = 60  # 1 minute
POLL_MIN_INTERVAL_PRO_PLUS = 30  # 30 seconds
POLL_MIN_INTERVAL_ULTRA = 10  # 10 seconds

# ============================================================================
# LLM Credit Limits (Monthly, in USD)
# ============================================================================

LLM_CREDITS_SELF_HOSTED = -1  # BYOK (Bring Your Own Key), unlimited
LLM_CREDITS_FREE = 5.00
LLM_CREDITS_PRO = 20.00
LLM_CREDITS_PRO_PLUS = 50.00
LLM_CREDITS_ULTRA = 100.00

# ============================================================================
# Credit Thresholds
# ============================================================================

# Soft warning threshold (percentage of monthly credits)
CREDIT_WARNING_THRESHOLD = 0.80  # Warn at 80% usage

# Hard block threshold (percentage of monthly credits)
CREDIT_BLOCK_THRESHOLD = 1.20  # Block at 120% usage (allow 20% overage)
