"""
Clerk metadata synchronization.

Syncs Stripe customer ID and billing organization to Clerk user metadata for frontend access.
"""
from typing import Optional

import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger("api.subscriptions.clerk_sync")


async def sync_stripe_customer_to_clerk(
    clerk_user_id: str,
    stripe_customer_id: str,
    billing_organization_id: Optional[int] = None,
) -> bool:
    """
    Sync Stripe customer ID and billing org to Clerk user's public metadata.

    This allows the frontend to access billing information without
    additional API calls to the backend.

    Args:
        clerk_user_id: The Clerk user ID (e.g., user_xxx)
        stripe_customer_id: The Stripe customer ID (e.g., cus_xxx)
        billing_organization_id: Optional organization ID that owns this billing (V2 model)

    Returns:
        True if sync was successful, False otherwise
    """
    if not config.clerk_secret_key:
        logger.debug("Clerk secret key not configured, skipping metadata sync")
        return False

    try:
        # Build metadata payload
        metadata: dict = {
            "stripe_customer_id": stripe_customer_id,
        }

        # V2: Include billing organization context
        if billing_organization_id is not None:
            metadata["billing_organization_id"] = str(billing_organization_id)

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.patch(
                f"https://api.clerk.com/v1/users/{clerk_user_id}",
                headers={
                    "Authorization": f"Bearer {config.clerk_secret_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "public_metadata": metadata
                }
            )
            response.raise_for_status()

            logger.info(
                "Synced Stripe customer %s to Clerk user %s (org=%s)",
                stripe_customer_id, clerk_user_id, billing_organization_id
            )
            return True

    except httpx.HTTPStatusError as e:
        logger.error(
            "Failed to sync Stripe customer to Clerk: %s %s",
            e.response.status_code, e.response.text[:200]
        )
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Clerk webhook processing should not crash on errors
        logger.error("Failed to sync Stripe customer to Clerk: %s", str(e))
        return False


async def update_clerk_billing_organization(
    clerk_user_id: str,
    billing_organization_id: int,
) -> bool:
    """
    Update the billing organization ID in Clerk user's public metadata.

    Called when subscription transfers between organizations to keep
    Clerk metadata in sync.

    Args:
        clerk_user_id: The Clerk user ID (e.g., user_xxx)
        billing_organization_id: The organization ID that now owns billing

    Returns:
        True if sync was successful, False otherwise
    """
    if not config.clerk_secret_key:
        logger.debug("Clerk secret key not configured, skipping metadata sync")
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.patch(
                f"https://api.clerk.com/v1/users/{clerk_user_id}",
                headers={
                    "Authorization": f"Bearer {config.clerk_secret_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "public_metadata": {
                        "billing_organization_id": str(billing_organization_id),
                    }
                }
            )
            response.raise_for_status()

            logger.info(
                "Updated billing org %s for Clerk user %s",
                billing_organization_id, clerk_user_id
            )
            return True

    except httpx.HTTPStatusError as e:
        logger.error(
            "Failed to update Clerk billing org: %s %s",
            e.response.status_code, e.response.text[:200]
        )
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Clerk sync should not crash on errors
        logger.error("Failed to update Clerk billing org: %s", str(e))
        return False
