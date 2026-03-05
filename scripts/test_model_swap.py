"""A/B test: GLM-4.7 vs Kimi K2.5 (structured output) and Qwen3-VL vs Gemini 2.5 Flash (vision)."""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# --- Test 1: Structured Output ---

STRUCTURED_PROMPT = """\
Extract the following information from this text and return valid JSON matching the schema below.

Text: "Acme Corp was founded in 2019 by Jane Doe and John Smith. They raised $5M in Series A \
from Sequoia Capital in March 2021. The company has 45 employees across offices in San Francisco \
and London. Their main products are AcmeDB (a distributed database, $299/mo enterprise tier) and \
AcmeML (an ML platform, $199/mo pro tier). Key competitors include DataBricks and Snowflake."

Schema:
{
  "company": {
    "name": "string",
    "founded_year": "integer",
    "founders": ["string"],
    "employee_count": "integer",
    "offices": ["string"]
  },
  "funding": [
    {
      "round": "string",
      "amount_usd": "integer",
      "lead_investor": "string",
      "date": "string (YYYY-MM)"
    }
  ],
  "products": [
    {
      "name": "string",
      "category": "string",
      "pricing": {"tier": "string", "monthly_usd": "integer"}
    }
  ],
  "competitors": ["string"]
}

Return ONLY valid JSON, no markdown fences."""

STRUCTURED_MODELS = [
    ("moonshotai/kimi-k2.5", "Kimi K2.5 (current)"),
    ("z-ai/glm-4.7", "GLM-4.7 (candidate)"),
]

# --- Test 2: Vision/Browser ---

# We'll use a simple text-based vision proxy test since we can't easily send screenshots
VISION_PROMPT = """\
You are a browser automation agent. Given this description of a webpage, identify the interactive \
elements and describe what actions you would take.

Page description: A login page with a centered card. At the top is a logo "Acme Corp". Below it \
are two input fields labeled "Email" and "Password". The Email field has placeholder "you@example.com". \
Below the inputs is a blue "Sign In" button spanning full width. Below that is a "Forgot password?" \
link in gray text. At the bottom is "Don't have an account? Sign up" with "Sign up" as a link.

List all interactive elements with their type, selector suggestion, and recommended action. \
Return as JSON array."""

VISION_MODELS = [
    ("google/gemini-2.5-flash", "Gemini 2.5 Flash (current)"),
    ("qwen/qwen3-vl-8b-thinking", "Qwen3-VL 8B Thinking (candidate)"),
]


async def call_model(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    """Call a model via OpenRouter and return result with timing."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2000,
    }
    start = time.monotonic()
    try:
        resp = await client.post(BASE_URL, json=payload, headers=HEADERS, timeout=60)
        elapsed = time.monotonic() - start
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {"content": content, "latency": elapsed, "usage": usage, "error": None}
    except (httpx.HTTPError, KeyError, ValueError) as e:
        return {"content": "", "latency": time.monotonic() - start, "usage": {}, "error": str(e)}


def try_parse_json(text: str) -> tuple[bool, any]:
    """Try to parse JSON from text, stripping markdown fences if needed."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return True, json.loads(cleaned)
    except json.JSONDecodeError as e:
        return False, str(e)


def print_result(label: str, result: dict, check_json: bool = True):
    """Print a model result."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if result["error"]:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  Latency: {result['latency']:.2f}s")
    usage = result["usage"]
    print(f"  Tokens: in={usage.get('prompt_tokens', '?')} out={usage.get('completion_tokens', '?')}")
    if check_json:
        valid, parsed = try_parse_json(result["content"])
        print(f"  Valid JSON: {valid}")
        if valid:
            print(f"  Output:\n{json.dumps(parsed, indent=2)[:1500]}")
        else:
            print(f"  Parse error: {parsed}")
            print(f"  Raw (first 500 chars): {result['content'][:500]}")
    else:
        print(f"  Output:\n{result['content'][:1500]}")


async def main():
    async with httpx.AsyncClient() as client:
        # Test 1: Structured output
        print("\n" + "#" * 60)
        print("  TEST 1: Structured Output — GLM-4.7 vs Kimi K2.5")
        print("#" * 60)

        results_structured = await asyncio.gather(
            *[call_model(client, model, STRUCTURED_PROMPT) for model, _ in STRUCTURED_MODELS]
        )
        for (model, label), result in zip(STRUCTURED_MODELS, results_structured):
            print_result(f"{label} ({model})", result, check_json=True)

        # Test 2: Vision/browser
        print("\n" + "#" * 60)
        print("  TEST 2: Vision/Browser — Qwen3-VL vs Gemini 2.5 Flash")
        print("#" * 60)

        results_vision = await asyncio.gather(
            *[call_model(client, model, VISION_PROMPT) for model, _ in VISION_MODELS]
        )
        for (model, label), result in zip(VISION_MODELS, results_vision):
            print_result(f"{label} ({model})", result, check_json=True)

        # Summary
        print("\n" + "#" * 60)
        print("  SUMMARY")
        print("#" * 60)
        for test_name, models, results in [
            ("Structured Output", STRUCTURED_MODELS, results_structured),
            ("Vision/Browser", VISION_MODELS, results_vision),
        ]:
            print(f"\n  {test_name}:")
            for (model, label), result in zip(models, results):
                if result["error"]:
                    print(f"    {label}: ERROR")
                else:
                    valid, _ = try_parse_json(result["content"])
                    print(f"    {label}: {result['latency']:.2f}s, valid_json={valid}")


if __name__ == "__main__":
    asyncio.run(main())
