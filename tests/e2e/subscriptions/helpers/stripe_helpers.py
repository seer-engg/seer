"""
Stripe Test Clock Manager for E2E subscription testing.

Provides utilities for time-based testing using Stripe test clocks.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import stripe

from seer.config import config
from seer.logger import get_logger

logger = get_logger("tests.stripe_helpers")


class StripeTestClockManager:
    """
    Manages Stripe test clocks for time-based subscription testing.

    Test clocks allow advancing time in Stripe's test environment to simulate
    trial expirations, billing cycles, and subscription lifecycle events.
    """

    def __init__(self):
        """Initialize the test clock manager."""
        self.created_clocks: list[str] = []
        stripe.api_key = config.stripe_secret_key

    def create_clock(self, frozen_time: Optional[datetime] = None) -> stripe.test_helpers.TestClock:
        """
        Create a new Stripe test clock.

        Args:
            frozen_time: The initial frozen time. Defaults to current time.

        Returns:
            The created test clock object.
        """
        if frozen_time is None:
            frozen_time = datetime.now(timezone.utc)

        # Convert datetime to Unix timestamp
        frozen_timestamp = int(frozen_time.timestamp())

        test_clock = stripe.test_helpers.TestClock.create(
            frozen_time=frozen_timestamp,
            name=f"test-clock-{frozen_time.isoformat()}",
        )

        self.created_clocks.append(test_clock.id)
        logger.info(
            "Created test clock %s at %s", test_clock.id, frozen_time.isoformat()
        )

        return test_clock

    def advance_clock(
        self,
        clock_id: str,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
    ) -> stripe.test_helpers.TestClock:
        """
        Advance a test clock by the specified duration.

        Args:
            clock_id: The test clock ID to advance
            days: Number of days to advance
            hours: Number of hours to advance
            minutes: Number of minutes to advance

        Returns:
            The updated test clock object.
        """
        # Retrieve current clock
        current_clock = stripe.test_helpers.TestClock.retrieve(clock_id)
        current_time = datetime.fromtimestamp(
            current_clock.frozen_time, tz=timezone.utc
        )

        # Calculate new time
        delta = timedelta(days=days, hours=hours, minutes=minutes)
        new_time = current_time + delta
        new_timestamp = int(new_time.timestamp())

        # Advance the clock
        advanced_clock = stripe.test_helpers.TestClock.advance(
            clock_id,
            frozen_time=new_timestamp,
        )

        logger.info(
            "Advanced clock %s from %s to %s",
            clock_id,
            current_time.isoformat(),
            new_time.isoformat(),
        )

        return advanced_clock

    def get_clock(self, clock_id: str) -> stripe.test_helpers.TestClock:
        """
        Retrieve a test clock by ID.

        Args:
            clock_id: The test clock ID

        Returns:
            The test clock object.
        """
        return stripe.test_helpers.TestClock.retrieve(clock_id)

    def cleanup(self):
        """Delete all test clocks created by this manager."""
        for clock_id in self.created_clocks:
            try:
                stripe.test_helpers.TestClock.delete(clock_id)
                logger.info("Deleted test clock %s", clock_id)
            except stripe.error.StripeError as exc:
                logger.warning("Failed to delete test clock %s: %s", clock_id, exc)

        self.created_clocks.clear()


def create_customer_with_test_clock(
    email: str, test_clock_id: str
) -> stripe.Customer:
    """
    Create a Stripe customer associated with a test clock.

    Args:
        email: Customer email address
        test_clock_id: Test clock ID to associate with the customer

    Returns:
        The created Stripe customer.
    """
    customer = stripe.Customer.create(
        email=email,
        test_clock=test_clock_id,
    )

    logger.info(
        "Created customer %s with test clock %s", customer.id, test_clock_id
    )

    return customer


def attach_test_payment_method(customer_id: str) -> stripe.PaymentMethod:
    """
    Attach a test payment method to a customer.

    Uses Stripe's test card (4242 4242 4242 4242) which always succeeds.

    Args:
        customer_id: The customer ID to attach the payment method to

    Returns:
        The attached payment method.
    """
    # Create a test payment method
    payment_method = stripe.PaymentMethod.create(
        type="card",
        card={
            "token": "tok_visa",  # Stripe test token for Visa card
        },
    )

    # Attach to customer
    payment_method.attach(customer=customer_id)

    # Set as default
    stripe.Customer.modify(
        customer_id,
        invoice_settings={"default_payment_method": payment_method.id},
    )

    logger.info(
        "Attached payment method %s to customer %s",
        payment_method.id,
        customer_id,
    )

    return payment_method


def create_test_card_token(
    card_number: str = "4242424242424242",
    exp_month: int = 12,
    exp_year: int = 2030,
    cvc: str = "123",
) -> stripe.Token:
    """
    Create a Stripe card token for testing.

    Args:
        card_number: Card number (default: Visa test card)
        exp_month: Expiration month
        exp_year: Expiration year
        cvc: Card verification code

    Returns:
        The created token.
    """
    token = stripe.Token.create(
        card={
            "number": card_number,
            "exp_month": exp_month,
            "exp_year": exp_year,
            "cvc": cvc,
        }
    )

    return token


# Test card numbers for different scenarios
TEST_CARDS = {
    "visa_success": "4242424242424242",
    "visa_declined": "4000000000000002",
    "visa_insufficient_funds": "4000000000009995",
    "visa_3ds_required": "4000002500003155",
}
