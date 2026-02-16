"""Browser stealth configuration for anti-detection.

Uses Chrome's new headless mode (--headless=new) which is much harder
to detect than the old headless mode. Works on cloud/AWS without Xvfb.

The key insight is that Chrome 109+ introduced a "new headless" mode that:
1. Shares code with headed Chrome (identical rendering)
2. Doesn't expose headless-specific JavaScript artifacts
3. Passes most bot detection including Google's
4. Runs without a display (perfect for containers/serverless)
"""
from __future__ import annotations

import platform
from typing import Any, Dict, List

# Stealth Chrome args - key is --headless=new for undetectable headless
STEALTH_CHROME_ARGS: List[str] = [
    "--headless=new",  # Chrome 109+ new headless mode (critical!)
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-popup-blocking",
]

# Realistic Chrome user agents (update periodically)
CHROME_USER_AGENTS: Dict[str, str] = {
    "linux": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "darwin": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}


def get_platform_user_agent() -> str:
    """Get a realistic user agent for the current platform.

    Returns:
        User agent string appropriate for the current OS.
    """
    system = platform.system().lower()
    return CHROME_USER_AGENTS.get(system, CHROME_USER_AGENTS["linux"])


def get_stealth_profile_kwargs() -> Dict[str, Any]:
    """Get BrowserUseProfile kwargs with stealth settings.

    Uses new headless mode which works identically to headed Chrome
    and is undetectable by most bot detection systems including Google.

    We set headless=False but pass --headless=new via args. This bypasses
    browser-use's default headless handling and uses Chrome's new
    undetectable headless mode directly.

    Returns:
        Dict of kwargs to pass to BrowserUseProfile constructor.
    """
    return {
        "headless": False,  # We handle headless via --headless=new in args
        "args": STEALTH_CHROME_ARGS,
        "user_agent": get_platform_user_agent(),
        "viewport": {"width": 1280, "height": 800},
    }


def get_headed_stealth_args() -> List[str]:
    """Get stealth args for headed mode (excludes --headless=new).

    For local development where a visible browser window is desired.

    Returns:
        List of Chrome args without headless flag.
    """
    return [arg for arg in STEALTH_CHROME_ARGS if arg != "--headless=new"]
