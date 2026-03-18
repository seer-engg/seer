from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "clone_stripe_products_to_sandbox.py"
SPEC = importlib.util.spec_from_file_location("clone_stripe_products_to_sandbox", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_price_payload_rewrites_meter_id() -> None:
    source_price = {
        "id": "price_live_123",
        "currency": "usd",
        "billing_scheme": "per_unit",
        "metadata": {"type": "overage"},
        "recurring": {
            "aggregate_usage": "sum",
            "interval": "month",
            "interval_count": 1,
            "meter": "mtr_live_123",
            "usage_type": "metered",
        },
        "unit_amount_decimal": "1",
    }

    payload = MODULE.build_price_payload(
        source_price=source_price,
        target_product_id="prod_sandbox_123",
        copy_lookup_keys=False,
        target_meter_id="mtr_sandbox_123",
    )

    assert payload["product"] == "prod_sandbox_123"
    assert payload["recurring"]["meter"] == "mtr_sandbox_123"
    assert "aggregate_usage" not in payload["recurring"]
    assert payload["metadata"]["seer_cloned_from_live_price_id"] == "price_live_123"


def test_build_meter_payload_keeps_required_meter_fields() -> None:
    source_meter = {
        "id": "mtr_live_123",
        "display_name": "LLM Credit Overage",
        "event_name": "llm_overage_usage",
        "default_aggregation": {"formula": "sum"},
        "value_settings": {"event_payload_key": "value"},
        "customer_mapping": {"type": "by_id", "event_payload_key": "stripe_customer_id"},
    }

    payload = MODULE.build_meter_payload(source_meter)

    assert payload == {
        "display_name": "LLM Credit Overage",
        "event_name": "llm_overage_usage",
        "default_aggregation": {"formula": "sum"},
        "value_settings": {"event_payload_key": "value"},
        "customer_mapping": {"type": "by_id", "event_payload_key": "stripe_customer_id"},
    }


def test_find_existing_meter_matches_by_event_name() -> None:
    live_meter = {
        "id": "mtr_live_123",
        "event_name": "llm_overage_usage",
    }
    sandbox_meters = [
        {"id": "mtr_other", "event_name": "other_usage"},
        {"id": "mtr_sandbox_123", "event_name": "llm_overage_usage"},
    ]

    meter = MODULE.find_existing_meter(sandbox_meters, live_meter)

    assert meter is not None
    assert meter["id"] == "mtr_sandbox_123"
