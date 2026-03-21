from __future__ import annotations

import logging
from typing import Any


async def check_runtime_credit_limit(context: Any, logger: logging.Logger) -> None:
    """Run credit checks using the runtime organization context when available."""
    if not context or not getattr(context, "user", None):
        return

    from seer.observability.credit_gate import check_credit_limit  # pylint: disable=import-outside-toplevel  # Reason: Late import avoids optional runtime dependency cycles

    organization = None
    get_organization = getattr(context, "get_organization", None)
    if callable(get_organization):
        organization = await get_organization()

    try:
        await check_credit_limit(context.user, organization)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: propagate credit failures only; log unexpected resolution issues
        if exc.__class__.__name__ == "CreditLimitExceeded":
            raise
        logger.error("Credit limit check failed: %s", exc)
