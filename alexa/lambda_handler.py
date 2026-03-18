"""
AWS Lambda handler for the Target Groceries Alexa skill.

Receives voice commands from Alexa, parses grocery items, and triggers
the target-grocery-order workflow in Seer.

Environment variables:
    SEER_API_URL: Base URL for Seer API (e.g., https://api.getseer.dev)
    SEER_API_KEY: API key for authenticating with Seer
    SEER_WORKFLOW_ID: ID of the target-grocery-order workflow
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict
from urllib.request import Request, urlopen
from urllib.error import URLError


SEER_API_URL = os.environ.get("SEER_API_URL", "https://api.getseer.dev")
SEER_API_KEY = os.environ.get("SEER_API_KEY", "")
SEER_WORKFLOW_ID = os.environ.get("SEER_WORKFLOW_ID", "")


def _parse_items(raw: str) -> list[str]:
    """Parse comma/and-separated item list from Alexa slot value."""
    # "milk, eggs, and bread" -> ["milk", "eggs", "bread"]
    items = re.split(r",\s*(?:and\s+)?|(?:\s+and\s+)", raw.strip())
    return [item.strip() for item in items if item.strip()]


def _trigger_workflow(items: list[str]) -> Dict[str, Any]:
    """Call Seer API to trigger the grocery order workflow."""
    url = f"{SEER_API_URL}/api/v1/workflows/{SEER_WORKFLOW_ID}/runs"
    payload = json.dumps({
        "inputs": {"items": items},
        "config": {},
    }).encode()

    req = Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {SEER_API_KEY}")

    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except URLError as exc:
        return {"error": str(exc)}


def _build_response(speech: str, should_end: bool = True) -> Dict[str, Any]:
    """Build Alexa JSON response."""
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": speech,
            },
            "shouldEndSession": should_end,
        },
    }


def _handle_add_groceries(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the AddGroceriesIntent."""
    slots = intent.get("slots", {})
    raw_items = slots.get("items", {}).get("value", "")

    if not raw_items:
        return _build_response(
            "I didn't catch the items. Try saying: add milk, eggs, and bread.",
            should_end=False,
        )

    items = _parse_items(raw_items)
    result = _trigger_workflow(items)

    if "error" in result:
        return _build_response(
            f"Sorry, I couldn't start the order. {result['error']}"
        )

    item_list = ", ".join(items)
    return _build_response(
        f"Adding {item_list} to your Target cart. "
        "I'll send you a notification when it's ready for approval."
    )


def _handle_intent(request: Dict[str, Any]) -> Dict[str, Any]:
    """Route intent requests."""
    intent = request.get("intent", {})
    intent_name = intent.get("name", "")

    if intent_name == "AddGroceriesIntent":
        return _handle_add_groceries(intent)

    if intent_name in ("AMAZON.HelpIntent",):
        return _build_response(
            "Say something like: add milk, eggs, and bread. "
            "I'll add them to your Target cart and notify you for approval.",
            should_end=False,
        )

    if intent_name in ("AMAZON.StopIntent", "AMAZON.CancelIntent"):
        return _build_response("Goodbye.")

    return _build_response("Sorry, I didn't understand that.", should_end=False)


def handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    """Lambda entry point for Alexa skill requests."""
    request = event.get("request", {})
    request_type = request.get("type", "")

    if request_type == "LaunchRequest":
        return _build_response(
            "Welcome to Target Groceries. Tell me what to add to your cart.",
            should_end=False,
        )

    if request_type == "IntentRequest":
        return _handle_intent(request)

    if request_type == "SessionEndedRequest":
        return _build_response("")

    return _build_response("Sorry, I didn't understand that.", should_end=False)
