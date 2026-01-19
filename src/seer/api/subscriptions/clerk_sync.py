"""
Clerk metadata synchronization.

Syncs Stripe customer ID to Clerk user metadata for frontend access.
"""
import httpx

from seer.config import config
from seer.logger import get_logger

logger = get_logger("api.subscriptions.clerk_sync")


async def sync_stripe_customer_to_clerk(clerk_user_id: str, stripe_customer_id: str) -> bool:
    """
    Sync Stripe customer ID to Clerk user's public metadata.

    This allows the frontend to access the Stripe customer ID without
    additional API calls to the backend.

    Args:
        clerk_user_id: The Clerk user ID (e.g., user_xxx)
        stripe_customer_id: The Stripe customer ID (e.g., cus_xxx)

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
                        "stripe_customer_id": stripe_customer_id,
                    }
                }
            )
            response.raise_for_status()

            logger.info(
                "Synced Stripe customer %s to Clerk user %s",
                stripe_customer_id, clerk_user_id
            )
            return True

    except httpx.HTTPStatusError as e:
        logger.error(
            "Failed to sync Stripe customer to Clerk: %s %s",
            e.response.status_code, e.response.text[:200]
        )
        return False
    except Exception as e:
        logger.error("Failed to sync Stripe customer to Clerk: %s", str(e))
        return False
