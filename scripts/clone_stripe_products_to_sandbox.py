#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


API_BASE_URL = "https://api.stripe.com"
DEFAULT_TIMEOUT_SECONDS = 30
PRODUCT_CLONE_METADATA_KEY = "seer_cloned_from_live_product_id"
PRICE_CLONE_METADATA_KEY = "seer_cloned_from_live_price_id"


class StripeAPIError(RuntimeError):
    """Raised when Stripe returns a non-success response."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone products and prices from a Stripe live account into a sandbox/test account. "
            "The script is idempotent for resources it creates by tagging sandbox metadata with the live resource ids."
        ),
    )
    parser.add_argument(
        "--live-env",
        default="main",
        help="Seer environment to load the live Stripe key from. Defaults to 'main'.",
    )
    parser.add_argument(
        "--sandbox-env",
        default="dev",
        help="Seer environment to load the sandbox Stripe key from. Defaults to 'dev'.",
    )
    parser.add_argument(
        "--stripe-api-version",
        default=os.getenv("STRIPE_API_VERSION"),
        help="Optional Stripe API version override sent with every request.",
    )
    parser.add_argument(
        "--product-id",
        action="append",
        dest="product_ids",
        default=[],
        help="Specific live product id to copy. Pass multiple times to copy a subset.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Copy only active products from the live account.",
    )
    parser.add_argument(
        "--copy-lookup-keys",
        action="store_true",
        help="Also copy price lookup keys. Leave disabled if the sandbox already has overlapping lookup keys.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned creates without making POST requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds. Defaults to {DEFAULT_TIMEOUT_SECONDS}.",
    )
    return parser.parse_args()


@dataclass(slots=True)
class StripeClient:
    api_key: str
    account_label: str
    api_version: str | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    def request(self, method: str, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        encoded_data: bytes | None = None
        url = f"{API_BASE_URL}{path}"

        if method.upper() == "GET" and data:
            url = f"{url}?{urlencode(list(flatten_form_fields(data)))}"
        elif data:
            encoded_data = urlencode(list(flatten_form_fields(data))).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if encoded_data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if self.api_version:
            headers["Stripe-Version"] = self.api_version

        request = Request(url=url, data=encoded_data, headers=headers, method=method.upper())

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise StripeAPIError(build_http_error_message(self.account_label, method, path, exc.code, body)) from exc
        except URLError as exc:
            raise StripeAPIError(f"{self.account_label}: request failed for {method} {path}: {exc.reason}") from exc

        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise StripeAPIError(f"{self.account_label}: received non-JSON response for {method} {path}") from exc

    def list_all(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        request_params = dict(params or {})
        request_params.setdefault("limit", 100)

        while True:
            response = self.request("GET", path, request_params)
            batch = response.get("data", [])
            items.extend(batch)
            if not response.get("has_more") or not batch:
                return items
            request_params["starting_after"] = batch[-1]["id"]


def flatten_form_fields(data: Any, prefix: str | None = None) -> Iterable[tuple[str, str]]:
    if data is None:
        return

    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}[{key}]" if prefix else key
            yield from flatten_form_fields(value, next_prefix)
        return

    if isinstance(data, list):
        for index, value in enumerate(data):
            next_prefix = f"{prefix}[{index}]"
            yield from flatten_form_fields(value, next_prefix)
        return

    if isinstance(data, bool):
        yield prefix or "", "true" if data else "false"
        return

    yield prefix or "", str(data)


def build_http_error_message(account_label: str, method: str, path: str, status_code: int, body: str) -> str:
    message = body
    try:
        parsed_body = json.loads(body)
        message = parsed_body.get("error", {}).get("message", body)
    except json.JSONDecodeError:
        pass
    return f"{account_label}: Stripe returned HTTP {status_code} for {method} {path}: {message}"


def require_api_key(value: str | None, label: str) -> str:
    if value:
        return value
    raise SystemExit(f"Missing {label}. Pass it explicitly or export the matching environment variable.")


def resolve_stripe_secret_key(environment: str) -> str:
    """
    Resolve stripe_secret_key for a specific Seer environment.

    `SeerConfig(env=...)` does not by itself retarget the AWS SSM source because the SSM
    loader reads ENV from process env. We therefore fetch the environment-specific SSM
    values explicitly and fall back to the standard config object for local .env/env usage.
    """
    # pylint: disable=import-outside-toplevel  # Reason: avoid importing the global config singleton on CLI startup
    from seer.config import SeerConfig
    from seer.utilities.aws.parameter_store import AwsSsmSettingsSource

    environment = environment.lower()

    ssm_values = AwsSsmSettingsSource(SeerConfig, ssm_path_prefix=f"/{environment}/")()
    stripe_secret_key = ssm_values.get("stripe_secret_key")
    if stripe_secret_key:
        return stripe_secret_key

    config = SeerConfig(env=environment)
    if config.env.lower() == environment and config.stripe_secret_key:
        return config.stripe_secret_key

    raise SystemExit(
        f"Missing stripe_secret_key for environment '{environment}'. "
        f"Expected it in AWS SSM under /{environment}/stripe_secret_key or via Seer config sources."
    )


def filter_present(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key in keys:
        value = data.get(key)
        if value in (None, "", [], {}):
            continue
        filtered[key] = value
    return filtered


def merge_metadata(source_metadata: dict[str, str] | None, marker_key: str, marker_value: str) -> dict[str, str]:
    metadata = dict(source_metadata or {})
    metadata[marker_key] = marker_value
    return metadata


def build_product_payload(source_product: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": source_product["name"],
        "metadata": merge_metadata(source_product.get("metadata"), PRODUCT_CLONE_METADATA_KEY, source_product["id"]),
    }
    payload.update(
        filter_present(
            source_product,
            [
                "active",
                "description",
                "images",
                "shippable",
                "statement_descriptor",
                "tax_code",
                "unit_label",
                "url",
            ],
        )
    )

    package_dimensions = filter_present(source_product.get("package_dimensions") or {}, ["height", "length", "weight", "width"])
    if package_dimensions:
        payload["package_dimensions"] = package_dimensions

    return payload


def build_meter_payload(source_meter: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "display_name": source_meter["display_name"],
        "event_name": source_meter["event_name"],
        "default_aggregation": filter_present(source_meter.get("default_aggregation") or {}, ["formula"]),
        "value_settings": filter_present(source_meter.get("value_settings") or {}, ["event_payload_key"]),
        "customer_mapping": filter_present(source_meter.get("customer_mapping") or {}, ["type", "event_payload_key"]),
    }
    if source_meter.get("event_time_window"):
        payload["event_time_window"] = source_meter["event_time_window"]
    return payload


def build_price_payload(
    source_price: dict[str, Any],
    target_product_id: str,
    copy_lookup_keys: bool,
    target_meter_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "product": target_product_id,
        "currency": source_price["currency"],
        "metadata": merge_metadata(source_price.get("metadata"), PRICE_CLONE_METADATA_KEY, source_price["id"]),
    }
    payload.update(filter_present(source_price, ["active", "billing_scheme", "nickname", "tax_behavior"]))

    recurring = filter_present(
        source_price.get("recurring") or {},
        ["interval", "interval_count", "meter", "trial_period_days", "usage_type"],
    )
    if target_meter_id:
        recurring["meter"] = target_meter_id
    if recurring:
        payload["recurring"] = recurring

    transform_quantity = filter_present(source_price.get("transform_quantity") or {}, ["divide_by", "round"])
    if transform_quantity:
        payload["transform_quantity"] = transform_quantity

    custom_unit_amount = filter_present(source_price.get("custom_unit_amount") or {}, ["enabled", "maximum", "minimum", "preset"])
    if custom_unit_amount:
        payload["custom_unit_amount"] = custom_unit_amount

    if source_price.get("billing_scheme") == "tiered":
        payload["tiers_mode"] = source_price["tiers_mode"]
        payload["tiers"] = [
            filter_present(tier, ["flat_amount", "flat_amount_decimal", "unit_amount", "unit_amount_decimal", "up_to"])
            for tier in source_price.get("tiers", [])
        ]
    else:
        if source_price.get("unit_amount_decimal") is not None:
            payload["unit_amount_decimal"] = source_price["unit_amount_decimal"]
        elif source_price.get("unit_amount") is not None:
            payload["unit_amount"] = source_price["unit_amount"]

    if copy_lookup_keys and source_price.get("lookup_key"):
        payload["lookup_key"] = source_price["lookup_key"]

    return payload


def find_existing_product(sandbox_products: list[dict[str, Any]], live_product_id: str) -> dict[str, Any] | None:
    for product in sandbox_products:
        metadata = product.get("metadata") or {}
        if metadata.get(PRODUCT_CLONE_METADATA_KEY) == live_product_id:
            return product
    return None


def find_existing_price(sandbox_prices: list[dict[str, Any]], live_price_id: str) -> dict[str, Any] | None:
    for price in sandbox_prices:
        metadata = price.get("metadata") or {}
        if metadata.get(PRICE_CLONE_METADATA_KEY) == live_price_id:
            return price
    return None


def find_existing_meter(sandbox_meters: list[dict[str, Any]], live_meter: dict[str, Any]) -> dict[str, Any] | None:
    live_event_name = live_meter.get("event_name")
    for meter in sandbox_meters:
        if meter.get("event_name") == live_event_name:
            return meter
    return None


def print_create(resource_label: str, source_id: str, payload: dict[str, Any], dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}create {resource_label} from {source_id}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def clone_products_and_prices(args: argparse.Namespace) -> int:
    live_api_key = require_api_key(resolve_stripe_secret_key(args.live_env), f"Stripe key for env '{args.live_env}'")
    sandbox_api_key = require_api_key(resolve_stripe_secret_key(args.sandbox_env), f"Stripe key for env '{args.sandbox_env}'")

    live_client = StripeClient(
        api_key=live_api_key,
        account_label="live",
        api_version=args.stripe_api_version,
        timeout_seconds=args.timeout_seconds,
    )
    sandbox_client = StripeClient(
        api_key=sandbox_api_key,
        account_label="sandbox",
        api_version=args.stripe_api_version,
        timeout_seconds=args.timeout_seconds,
    )

    live_product_filters: dict[str, Any] = {}
    if args.active_only:
        live_product_filters["active"] = True

    live_products = live_client.list_all("/v1/products", live_product_filters)
    sandbox_products = sandbox_client.list_all("/v1/products")
    sandbox_meters = sandbox_client.list_all("/v1/billing/meters")
    live_meter_cache: dict[str, dict[str, Any]] = {}
    sandbox_meter_cache: dict[str, dict[str, Any]] = {}

    if args.product_ids:
        requested_ids = set(args.product_ids)
        live_products = [product for product in live_products if product["id"] in requested_ids]

    if not live_products:
        print("No matching live products found.")
        return 0

    created_products = 0
    skipped_products = 0
    created_prices = 0
    skipped_prices = 0
    created_meters = 0
    skipped_meters = 0

    for live_product in live_products:
        sandbox_product = find_existing_product(sandbox_products, live_product["id"])

        if sandbox_product is None:
            product_payload = build_product_payload(live_product)
            print_create("product", live_product["id"], product_payload, args.dry_run)
            if args.dry_run:
                sandbox_product = {"id": f"dryrun_{live_product['id']}"}
            else:
                sandbox_product = sandbox_client.request("POST", "/v1/products", product_payload)
                sandbox_products.append(sandbox_product)
            created_products += 1
        else:
            print(f"skip product {live_product['id']} -> {sandbox_product['id']} (already cloned)")
            skipped_products += 1

        live_prices = live_client.list_all("/v1/prices", {"product": live_product["id"]})
        sandbox_prices = sandbox_client.list_all("/v1/prices", {"product": sandbox_product["id"]}) if not args.dry_run else []

        for live_price in live_prices:
            existing_price = find_existing_price(sandbox_prices, live_price["id"])
            if existing_price is not None:
                print(f"skip price {live_price['id']} -> {existing_price['id']} (already cloned)")
                skipped_prices += 1
                continue

            target_meter_id: str | None = None
            live_meter_id = (live_price.get("recurring") or {}).get("meter")
            if live_meter_id:
                live_meter = live_meter_cache.get(live_meter_id)
                if live_meter is None:
                    live_meter = live_client.request("GET", f"/v1/billing/meters/{live_meter_id}")
                    live_meter_cache[live_meter_id] = live_meter

                sandbox_meter = sandbox_meter_cache.get(live_meter_id)
                if sandbox_meter is None:
                    sandbox_meter = find_existing_meter(sandbox_meters, live_meter)
                    if sandbox_meter is None:
                        meter_payload = build_meter_payload(live_meter)
                        print_create("billing meter", live_meter_id, meter_payload, args.dry_run)
                        if args.dry_run:
                            sandbox_meter = {"id": f"dryrun_{live_meter_id}", "event_name": live_meter["event_name"]}
                        else:
                            sandbox_meter = sandbox_client.request("POST", "/v1/billing/meters", meter_payload)
                            sandbox_meters.append(sandbox_meter)
                        created_meters += 1
                    else:
                        if sandbox_meter.get("status") == "inactive" and not args.dry_run:
                            sandbox_meter = sandbox_client.request("POST", f"/v1/billing/meters/{sandbox_meter['id']}/reactivate")
                        print(f"skip billing meter {live_meter_id} -> {sandbox_meter['id']} (matched by event_name)")
                        skipped_meters += 1
                    sandbox_meter_cache[live_meter_id] = sandbox_meter
                target_meter_id = sandbox_meter["id"]

            price_payload = build_price_payload(live_price, sandbox_product["id"], args.copy_lookup_keys, target_meter_id)
            print_create("price", live_price["id"], price_payload, args.dry_run)

            if not args.dry_run:
                created_price = sandbox_client.request("POST", "/v1/prices", price_payload)
                sandbox_prices.append(created_price)
            created_prices += 1

    print(
        "Finished. "
        f"products created={created_products}, products skipped={skipped_products}, "
        f"meters created={created_meters}, meters skipped={skipped_meters}, "
        f"prices created={created_prices}, prices skipped={skipped_prices}"
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return clone_products_and_prices(args)
    except StripeAPIError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
