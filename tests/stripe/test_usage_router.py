from datetime import datetime, timezone
from decimal import Decimal

from seer.api.usage.router import _build_usage_metric


def test_build_usage_metric_handles_unlimited_limits():
    metric = _build_usage_metric(
        used=5,
        limit_value=-1,
        is_unlimited=True,
        unit="messages",
    )

    assert metric.limit is None
    assert metric.remaining is None
    assert metric.is_unlimited is True
    assert metric.used == 5.0
    assert metric.unit == "messages"


def test_build_usage_metric_handles_disabled_limits():
    metric = _build_usage_metric(
        used=3,
        limit_value=0,
        is_unlimited=False,
        disabled=True,
    )

    assert metric.disabled is True
    assert metric.limit == 0
    assert metric.remaining == 0


def test_build_usage_metric_coerces_decimal_and_remaining():
    metric = _build_usage_metric(
        used=Decimal("1.5"),
        limit_value=5,
        is_unlimited=False,
    )

    assert metric.used == 1.5
    assert metric.remaining == 3.5
