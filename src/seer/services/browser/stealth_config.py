"""Browser profile configuration for Browserless connections.

Provides user agent strings and BrowserProfile kwargs for remote
browser sessions connected via CDP to a Browserless service.
"""
from __future__ import annotations

import glob
import os
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


def find_local_chromium_executable() -> str | None:
    """Find the locally installed Chromium/Chrome executable.

    Checks the PLAYWRIGHT_BROWSERS_PATH environment variable first,
    then falls back to common system paths. Returns the path to the
    executable or None if not found.
    """
    playwright_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    system = platform.system()

    if system == "Linux":
        search_patterns = []
        if playwright_path:
            search_patterns += [
                f"{playwright_path}/chromium-*/chrome-linux*/chrome",
                f"{playwright_path}/chromium_headless_shell-*/chrome-linux*/chrome",
            ]
        search_patterns += [
            "/usr/bin/google-chrome-stable",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ]
    elif system == "Darwin":
        search_patterns = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
        if playwright_path:
            search_patterns += [
                f"{playwright_path}/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
            ]
    else:
        return None

    for pattern in search_patterns:
        if "*" in pattern:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]
        elif os.path.isfile(pattern) and os.access(pattern, os.X_OK):
            return pattern

    return None


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


def get_local_profile_kwargs() -> Dict[str, Any]:
    """Get BrowserUseProfile kwargs for local browser sessions.

    When no remote Browserless service is configured, browser_use
    launches a local Chrome/Chromium subprocess. This sets executable_path
    explicitly so the LocalBrowserWatchdog doesn't need to search for it
    (which requires PLAYWRIGHT_BROWSERS_PATH to be set in the worker env).

    Returns:
        Dict of kwargs to pass to BrowserUseProfile constructor.
    """
    kwargs: Dict[str, Any] = {
        "user_agent": get_platform_user_agent(),
        "viewport": {"width": 1280, "height": 800},
    }
    executable = find_local_chromium_executable()
    if executable:
        kwargs["executable_path"] = executable
    return kwargs
