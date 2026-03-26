# pylint: disable=import-outside-toplevel
# Reason: Test file with lazy imports
"""
Unit tests for overage billing pure computation.

Tests cover:
- OverageSettings property calculations (cents→dollars conversion)
- Cap reached detection logic
- Margin calculation math
"""
from decimal import Decimal

import pytest


pytestmark = pytest.mark.unit


# =============================================================================
# OverageSettings Model Tests
# =============================================================================


class TestOverageSettingsModel:
    """Tests for OverageSettings model."""

    def test_spending_cap_dollars_property(self):
        """Test spending_cap_dollars converts cents to dollars."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 5000

        assert settings.spending_cap_dollars == Decimal("50.00")

    def test_spending_cap_dollars_handles_zero(self):
        """Test spending_cap_dollars handles zero."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 0

        assert settings.spending_cap_dollars == Decimal("0.00")

    def test_spending_cap_dollars_handles_fractional_cents(self):
        """Test spending_cap_dollars handles odd cent amounts."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 5001

        assert settings.spending_cap_dollars == Decimal("50.01")

    def test_current_period_overage_dollars_property(self):
        """Test current_period_overage_dollars converts cents to dollars."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.current_period_overage_cents = 1500

        assert settings.current_period_overage_dollars == Decimal("15.00")

    def test_current_period_overage_dollars_handles_zero(self):
        """Test current_period_overage_dollars handles zero."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.current_period_overage_cents = 0

        assert settings.current_period_overage_dollars == Decimal("0.00")

    def test_remaining_cap_cents_property(self):
        """Test remaining_cap_cents calculates remaining budget."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 5000
        settings.current_period_overage_cents = 1500

        assert settings.remaining_cap_cents == 3500

    def test_remaining_cap_cents_at_zero(self):
        """Test remaining_cap_cents returns zero when at cap."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 5000
        settings.current_period_overage_cents = 5000

        assert settings.remaining_cap_cents == 0

    def test_remaining_cap_cents_over_cap_returns_zero(self):
        """Test remaining_cap_cents returns zero when over cap."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 5000
        settings.current_period_overage_cents = 6000  # Over cap

        assert settings.remaining_cap_cents == 0

    def test_is_cap_reached_returns_false_below_cap(self):
        """Test is_cap_reached returns False when below cap."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 5000
        settings.current_period_overage_cents = 4999

        assert settings.is_cap_reached() is False

    def test_is_cap_reached_returns_true_at_cap(self):
        """Test is_cap_reached returns True when at cap."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 5000
        settings.current_period_overage_cents = 5000

        assert settings.is_cap_reached() is True

    def test_is_cap_reached_returns_true_over_cap(self):
        """Test is_cap_reached returns True when over cap."""
        from seer.database.overage_models import OverageSettings

        settings = OverageSettings()
        settings.spending_cap_cents = 5000
        settings.current_period_overage_cents = 6000

        assert settings.is_cap_reached() is True


# =============================================================================
# Margin Calculation Tests
# =============================================================================


class TestMarginCalculation:
    """Tests for margin calculation logic used in overage billing."""

    def test_standard_margin_calculation(self):
        """Test 30% margin calculation."""
        base_cost_cents = 100
        margin_multiplier = Decimal("1.30")

        billed_amount_cents = int(float(base_cost_cents) * float(margin_multiplier))

        assert billed_amount_cents == 130

    def test_margin_on_small_amounts(self):
        """Test margin calculation on small amounts rounds correctly."""
        base_cost_cents = 1  # 1 cent
        margin_multiplier = Decimal("1.30")

        billed_amount_cents = int(float(base_cost_cents) * float(margin_multiplier))

        assert billed_amount_cents == 1  # Rounds down

    def test_margin_on_large_amounts(self):
        """Test margin calculation on larger amounts."""
        base_cost_cents = 10000  # $100
        margin_multiplier = Decimal("1.30")

        billed_amount_cents = int(float(base_cost_cents) * float(margin_multiplier))

        assert billed_amount_cents == 13000  # $130

    def test_custom_margin_calculation(self):
        """Test custom margin multiplier."""
        base_cost_cents = 100
        margin_multiplier = Decimal("1.50")  # 50% margin

        billed_amount_cents = int(float(base_cost_cents) * float(margin_multiplier))

        assert billed_amount_cents == 150
