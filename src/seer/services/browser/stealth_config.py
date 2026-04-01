"""Browser profile configuration for Browserless connections.

Provides user agent strings and BrowserProfile kwargs for remote
browser sessions connected via CDP to a Browserless service.
"""
from __future__ import annotations

import platform
from typing import Any, Dict

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


def get_remote_profile_kwargs() -> Dict[str, Any]:
    """Get BrowserUseProfile kwargs for remote Browserless connections.

    When connecting to a remote Browserless service via CDP, Chrome
    launch args are managed by Browserless itself. Only user_agent
    and viewport are forwarded via BrowserProfile.

    Returns:
        Dict of kwargs to pass to BrowserUseProfile constructor.
    """
    return {
        "user_agent": get_platform_user_agent(),
        "viewport": {"width": 1280, "height": 800},
    }
