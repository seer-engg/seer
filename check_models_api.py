#!/usr/bin/env python3
"""
Check which models are available via the /api/models endpoint.
This verifies that only Kimi models appear when OpenAI key is not configured.
"""

import json
import traceback

import requests

from src.seer.config import config

def check_models_api():
    """Check the models API endpoint."""
    print("=" * 60)
    print("MODEL API ENDPOINT CHECK")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  OpenAI key configured: {config.openai_api_key is not None}")
    print(f"  Anthropic key configured: {config.anthropic_api_key is not None}")
    print(f"  OpenRouter key configured: {config.openrouter_api_key is not None}")
    print(f"  Default model: {config.default_llm_model}")

    # You'll need to start the backend server first
    url = "http://localhost:8000/api/models"
    print(f"\nFetching models from: {url}")
    print("(Make sure backend server is running: uv run main.py)\n")

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        models = response.json()

        print(f"✅ API Response (Status {response.status_code}):")
        print(json.dumps(models, indent=2))

        print(f"\nTotal models available: {len(models)}")

        # Check providers
        providers = set(m['provider'] for m in models)
        print(f"\nProviders present: {', '.join(providers)}")

        # Check if Kimi is present
        kimi_models = [m for m in models if 'kimi' in m['id'].lower() or m['provider'] == 'openrouter']
        if kimi_models:
            print("\n✅ OpenRouter/Kimi models found:")
            for model in kimi_models:
                print(f"   • {model['name']} ({model['id']})")
        else:
            print("\n⚠️  No OpenRouter/Kimi models found!")

        # Check if OpenAI is present (should not be if key is commented out)
        openai_models = [m for m in models if m['provider'] == 'openai']
        if openai_models:
            print("\n⚠️  OpenAI models found (you may want to remove OPENAI_API_KEY):")
            for model in openai_models:
                print(f"   • {model['name']} ({model['id']})")
        else:
            print("\n✅ No OpenAI models (OPENAI_API_KEY not configured)")

    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the backend server running?")
        print("   Start it with: cd /workspace/backend && uv run main.py")
    except Exception as e:  # pylint: disable=broad-except  # CLI script needs to catch all exceptions for user-friendly error display
        print(f"❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    check_models_api()
