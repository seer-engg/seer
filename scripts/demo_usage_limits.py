"""
Demonstration script for the Usage Limits system.

This script shows how to use the usage limits API to check tier limits,
resolve user tiers, and track resource usage.

Run: uv run python scripts/demo_usage_limits.py
"""
from seer.database.subscription_models import SubscriptionTier
from seer.observability import get_limits_for_tier
from seer.observability.models import SELF_HOSTED_LIMITS


def print_tier_limits(tier: SubscriptionTier, limits) -> None:
    """Pretty print tier limits."""
    print(f"\n{'='*60}")
    print(f"Tier: {tier.value.upper()}")
    print(f"{'='*60}")
    print(f"  Workflows:              {limits.workflows if limits.workflows != -1 else 'Unlimited'}")
    print(f"  Runs per month:         {limits.runs_monthly if limits.runs_monthly != -1 else 'Unlimited'}")
    print(f"  Chat messages/workflow: {limits.chat_messages_per_workflow if limits.chat_messages_per_workflow > 0 else 'Disabled' if limits.chat_messages_per_workflow == 0 else 'Unlimited'}")  # pylint: disable=line-too-long  # reason: demonstration output
    print(f"  Account day limit:      {limits.account_day_limit if limits.account_day_limit != -1 else 'No limit'}")
    print(f"  Min poll interval:      {limits.poll_min_interval_seconds}s")
    print(f"  LLM credits/month:      ${limits.llm_credits_monthly if limits.llm_credits_monthly != -1 else 'BYOK (Bring Your Own Key)'}")  # pylint: disable=line-too-long  # reason: demonstration output
    print(f"{'='*60}")


def main():
    """Demonstrate usage limits system."""
    print("\n" + "="*60)
    print(" Usage Limits System - Phase 1 Demonstration")
    print("="*60)

    # Show limits for each cloud tier
    for tier in [SubscriptionTier.FREE, SubscriptionTier.PRO, SubscriptionTier.PRO_PLUS, SubscriptionTier.ULTRA]:
        limits = get_limits_for_tier(tier)
        print_tier_limits(tier, limits)

    # Show self-hosted limits
    print(f"\n{'='*60}")
    print("Self-Hosted Mode")
    print(f"{'='*60}")
    print(f"  Workflows:              {'Unlimited'}")
    print(f"  Runs per month:         {'Unlimited'}")
    print(f"  Chat messages/workflow: {'Disabled (BYOK only)'}")
    print(f"  Account day limit:      {'No limit'}")
    print(f"  Min poll interval:      {SELF_HOSTED_LIMITS.poll_min_interval_seconds}s")
    print(f"  LLM credits/month:      {'BYOK (Bring Your Own Key)'}")
    print(f"{'='*60}")

    # Show tier progression
    print("\n" + "="*60)
    print(" Tier Progression Analysis")
    print("="*60)

    free = get_limits_for_tier(SubscriptionTier.FREE)
    pro = get_limits_for_tier(SubscriptionTier.PRO)
    pro_plus = get_limits_for_tier(SubscriptionTier.PRO_PLUS)
    ultra = get_limits_for_tier(SubscriptionTier.ULTRA)

    print("\nRuns per month progression:")
    print(f"  FREE → PRO:      {free.runs_monthly} → {pro.runs_monthly:,} ({pro.runs_monthly / free.runs_monthly}x increase)")
    print(f"  PRO → PRO+:      {pro.runs_monthly:,} → {pro_plus.runs_monthly:,} ({pro_plus.runs_monthly / pro.runs_monthly}x increase)")
    print(f"  PRO+ → ULTRA:    {pro_plus.runs_monthly:,} → {ultra.runs_monthly:,} ({ultra.runs_monthly / pro_plus.runs_monthly}x increase)")

    print("\nPolling frequency (faster is better):")
    print(f"  FREE:  {free.poll_min_interval_seconds}s (every 15 minutes)")
    print(f"  PRO:   {pro.poll_min_interval_seconds}s (every minute)")
    print(f"  PRO+:  {pro_plus.poll_min_interval_seconds}s (every 30 seconds)")
    print(f"  ULTRA: {ultra.poll_min_interval_seconds}s (every 10 seconds)")

    print("\nLLM credits per month:")
    print(f"  FREE:  ${free.llm_credits_monthly}")
    print(f"  PRO:   ${pro.llm_credits_monthly} ({pro.llm_credits_monthly / free.llm_credits_monthly}x increase)")
    print(f"  PRO+:  ${pro_plus.llm_credits_monthly} ({pro_plus.llm_credits_monthly / pro.llm_credits_monthly}x increase)")
    print(f"  ULTRA: ${ultra.llm_credits_monthly} ({ultra.llm_credits_monthly / pro_plus.llm_credits_monthly}x increase)")

    # Show upgrade incentives
    print("\n" + "="*60)
    print(" Key Upgrade Incentives")
    print("="*60)
    print("\nFREE → PRO benefits:")
    print("  ✓ Unlimited workflows (vs 3)")
    print("  ✓ 10,000x more runs (1M vs 100)")
    print("  ✓ Unlimited chat messages (vs 5 per workflow)")
    print("  ✓ No trial expiration (vs 14 days)")
    print("  ✓ 15x faster polling (1 min vs 15 min)")
    print("  ✓ 4x more LLM credits ($20 vs $5)")

    print("\nPRO → PRO+ benefits:")
    print("  ✓ 5x more runs (5M vs 1M)")
    print("  ✓ 2x faster polling (30s vs 1 min)")
    print("  ✓ 2.5x more LLM credits ($50 vs $20)")

    print("\nPRO+ → ULTRA benefits:")
    print("  ✓ 4x more runs (20M vs 5M)")
    print("  ✓ 3x faster polling (10s vs 30s)")
    print("  ✓ 2x more LLM credits ($100 vs $50)")

    print("\n" + "="*60)
    print(" ✓ Phase 1 Implementation Complete")
    print("="*60)
    print("\nNext: Phase 2 - Enforcement Points")
    print("  - Workflow creation gate")
    print("  - Workflow run gate")
    print("  - Chat message gate")
    print("  - Trial expiration gate")
    print("  - Polling frequency validation")
    print("  - LLM credit tracking")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
