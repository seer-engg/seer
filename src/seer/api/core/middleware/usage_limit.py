"""
Centralized usage limit enforcement middleware.

This middleware enforces subscription-based usage limits BEFORE route handlers execute,
providing a single source of truth for all limit checks.
"""
import json
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from seer.api.core.middleware.path_allowlist import is_public_path, is_payment_exempt_path
from seer.database import User
from seer.database.organization_models import Organization, OrganizationType
from seer.logger import get_logger
from seer.observability import (
    RunLimitExceeded,
    WorkflowLimitExceeded,
    get_effective_limits,
    get_effective_tier,
    get_monthly_run_count,
    get_org_monthly_run_count,
    get_org_workflow_count,
    get_workflow_count,
    resolve_user_tier,
)
from seer.config import config
from seer.observability.credit_gate import check_credit_limit
from seer.observability.exceptions import CreditLimitExceeded

logger = get_logger(__name__)


class UsageLimitMiddleware(BaseHTTPMiddleware):
    """
    Centralized usage limit enforcement for all subscription tiers.

    Checks limits before route handlers execute based on:
    - Request path pattern matching
    - Current user usage counters
    - User's subscription tier

    Returns HTTP 402 Payment Required with upgrade prompt if limit exceeded.

    Enforcement Points:
    1. Workflow creation: POST /api/v1/workflows
    2. Workflow runs: POST /api/v1/workflows/{id}/run
    3. Polling intervals: POST /api/v1/trigger-subscriptions (soft enforcement)
    4. Chat LLM credits: POST /nexus/{id}/chat and /nexus/{id}/chat/resume
    """

    # pylint: disable=too-complex,too-many-return-statements # Reason: Middleware dispatch with multiple path-based enforcement points
    async def dispatch(self, request: Request, call_next):
        """
        Main middleware dispatch method.

        Checks usage limits based on request path and method before proceeding to handler.
        """
        path = request.url.path
        if request.method == "OPTIONS":
            return await call_next(request)

        if config.disable_usage_limits:
            return await call_next(request)

        # Public paths skip all checks
        if is_public_path(path, include_docs=True):
            return await call_next(request)

        # Payment-exempt paths skip usage limits but require auth
        if is_payment_exempt_path(path):
            user: Optional[User] = getattr(request.state, "db_user", None)
            if user is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"},
                )
            # Skip usage limit checks for payment/billing endpoints
            return await call_next(request)

        # Regular authenticated paths - check usage limits
        user: Optional[User] = getattr(request.state, "db_user", None)
        if user is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )

        # Get organization context (may be None for personal workspace)
        organization: Optional[Organization] = getattr(request.state, "organization", None)

        # Get effective limits (org limits for team orgs, user limits otherwise)
        limits = await get_effective_limits(user, organization)

        # Check limits based on request path
        method = request.method

        # 1. Workflow Creation Limit
        if method == "POST" and path == "/api/v1/workflows":
            response = await self._check_workflow_creation_limit(user, organization, limits)
            if response:
                return response

        # 2. Workflow Run Limit
        elif method == "POST" and "/run" in path and "/api/v1/workflows/" in path:
            response = await self._check_workflow_run_limit(user, organization, limits)
            if response:
                return response

        # 3. Polling Frequency Validation (reads request body for soft enforcement)
        elif method == "POST" and "trigger-subscriptions" in path:
            await self._validate_polling_interval(request, user, limits)

        # 4. Chat LLM Credit Limit (before chat execution starts)
        elif method == "POST" and "/nexus/" in path and "/chat" in path:
            response = await self._check_llm_credit_limit(user, organization)
            if response:
                return response

        # All checks passed - proceed to handler
        return await call_next(request)

    async def _check_workflow_creation_limit(
        self, user: User, organization: Optional[Organization], limits
    ) -> Optional[JSONResponse]:
        """Check if user/org has exceeded workflow creation limit."""
        if not limits.has_unlimited_workflows:
            use_org_limits = organization and organization.type == OrganizationType.TEAM
            logger.info(
                "Checking workflow creation limit for user %s (org=%s)",
                user.id, organization.id if organization else None
            )

            # Get current count: org-level for team orgs, user-level otherwise
            if use_org_limits and organization:
                current = await get_org_workflow_count(organization)
            else:
                current = await get_workflow_count(user)

            if current >= limits.workflows:
                tier = await get_effective_tier(user, organization)
                error = WorkflowLimitExceeded(limits.workflows, current, tier)
                logger.warning(
                    "Workflow creation limit exceeded for user %s (org=%s, tier=%s, current=%d, limit=%d)",
                    user.id,
                    organization.id if organization else None,
                    tier.value,
                    current,
                    limits.workflows,
                )
                return JSONResponse(status_code=402, content=error.to_dict())
        return None

    async def _check_workflow_run_limit(
        self, user: User, organization: Optional[Organization], limits
    ) -> Optional[JSONResponse]:
        """Check if user/org has exceeded workflow run limit."""
        if not limits.has_unlimited_runs:
            use_org_limits = organization and organization.type == OrganizationType.TEAM

            # Get current count: org-level for team orgs, user-level otherwise
            if use_org_limits and organization:
                current = await get_org_monthly_run_count(organization)
            else:
                current = await get_monthly_run_count(user)

            if current >= limits.runs_monthly:
                tier = await get_effective_tier(user, organization)
                error = RunLimitExceeded(limits.runs_monthly, current, tier)
                logger.warning(
                    "Workflow run limit exceeded for user %s (org=%s, tier=%s, current=%d, limit=%d)",
                    user.id,
                    organization.id if organization else None,
                    tier.value,
                    current,
                    limits.runs_monthly,
                )
                return JSONResponse(status_code=402, content=error.to_dict())
        return None

    async def _check_llm_credit_limit(
        self, user: User, organization: Optional[Organization] = None
    ) -> Optional[JSONResponse]:
        """Check if user/org has exceeded LLM credit limit before chat execution."""
        # BYOK users bypass credit limits — they bear the LLM cost directly
        if user.active_organization_id:
            from seer.database.byok_models import LLMApiKey  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
            has_byok = await LLMApiKey.exists(
                organization_id=user.active_organization_id, is_active=True, status="active",
            )
            if has_byok:
                return None

        try:
            await check_credit_limit(user, organization)
            return None
        except CreditLimitExceeded as exc:
            logger.warning(
                "LLM credit limit exceeded for user %s (org=%s, period=%s, current=$%.2f, limit=$%.2f)",
                user.id,
                organization.id if organization else None,
                exc.period.value,
                exc.current,
                exc.limit,
            )
            return JSONResponse(status_code=402, content=exc.to_dict())

    async def _validate_polling_interval(self, request: Request, user: User, limits) -> None:
        """
        Soft enforcement of polling interval limits.

        Reads request body, checks poll_interval_seconds, and logs warning if too fast.
        Does not block the request, but logs for monitoring.
        """
        try:
            body = await request.body()
            # Must reset body for downstream handlers
            request._body = body  # pylint: disable=protected-access # Reason: Required pattern to reset request body in middleware

            data = json.loads(body.decode("utf-8"))
            requested_interval = data.get("poll_interval_seconds")

            if requested_interval is not None:
                min_interval = limits.poll_min_interval_seconds
                if requested_interval < min_interval:
                    tier = await resolve_user_tier(user)
                    logger.warning(
                        "Poll interval too fast for user %s (tier=%s, requested=%ds, min=%ds)",
                        user.id,
                        tier.value,
                        requested_interval,
                        min_interval,
                        extra={
                            "user_id": user.id,
                            "tier": tier.value,
                            "requested_interval": requested_interval,
                            "min_interval": min_interval,
                        },
                    )
                    # Note: We log but don't block. Service layer will clamp the value.
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Body parsing is non-critical, log and proceed with request
            # If body parsing fails, let it through (validation will catch it downstream)
            logger.debug("Failed to parse request body for polling interval check: %s", e)
