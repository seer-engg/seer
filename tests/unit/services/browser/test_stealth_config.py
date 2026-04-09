"""Tests for browser stealth configuration."""
import platform
from unittest.mock import patch

import pytest

from seer.services.browser.stealth_config import (
    CHROME_USER_AGENTS,
    STEALTH_SCRIPTS,
    get_platform_user_agent,
    get_remote_profile_kwargs,
    get_stealth_scripts_combined,
)


@pytest.mark.unit
class TestUserAgents:
    """Test Chrome user agent strings."""

    def test_all_platforms_have_agents(self):
        """All common platforms should have user agents."""
        assert "linux" in CHROME_USER_AGENTS
        assert "windows" in CHROME_USER_AGENTS
        assert "darwin" in CHROME_USER_AGENTS

    def test_user_agents_not_headless(self):
        """User agents should not contain 'Headless'."""
        for ua in CHROME_USER_AGENTS.values():
            assert "Headless" not in ua, f"User agent should not contain 'Headless': {ua}"

    def test_user_agents_contain_chrome(self):
        """User agents should identify as Chrome."""
        for ua in CHROME_USER_AGENTS.values():
            assert "Chrome/" in ua

    def test_user_agents_version_is_current(self):
        """User agents should use a recent Chrome version (130+)."""
        for ua in CHROME_USER_AGENTS.values():
            # Extract Chrome/NNN from the UA string
            chrome_part = [p for p in ua.split() if p.startswith("Chrome/")][0]
            major_version = int(chrome_part.split("/")[1].split(".")[0])
            assert major_version >= 130, f"Chrome version {major_version} is outdated"


@pytest.mark.unit
class TestGetPlatformUserAgent:
    """Test get_platform_user_agent function."""

    @patch.object(platform, "system", return_value="Linux")
    def test_linux_platform(self, mock_system):
        """Should return Linux user agent on Linux."""
        ua = get_platform_user_agent()
        assert "Linux" in ua or "X11" in ua

    @patch.object(platform, "system", return_value="Darwin")
    def test_darwin_platform(self, mock_system):
        """Should return macOS user agent on Darwin."""
        ua = get_platform_user_agent()
        assert "Macintosh" in ua

    @patch.object(platform, "system", return_value="Windows")
    def test_windows_platform(self, mock_system):
        """Should return Windows user agent on Windows."""
        ua = get_platform_user_agent()
        assert "Windows" in ua

    @patch.object(platform, "system", return_value="UnknownOS")
    def test_unknown_platform_fallback(self, mock_system):
        """Should fallback to Linux user agent on unknown platforms."""
        ua = get_platform_user_agent()
        assert ua == CHROME_USER_AGENTS["linux"]


@pytest.mark.unit
class TestGetRemoteProfileKwargs:
    """Test get_remote_profile_kwargs function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        kwargs = get_remote_profile_kwargs()
        assert isinstance(kwargs, dict)

    def test_contains_user_agent(self):
        """Should contain a user agent string."""
        kwargs = get_remote_profile_kwargs()
        assert "user_agent" in kwargs
        assert "Chrome/" in kwargs["user_agent"]

    def test_contains_viewport(self):
        """Should contain viewport settings."""
        kwargs = get_remote_profile_kwargs()
        assert "viewport" in kwargs
        assert kwargs["viewport"]["width"] == 1280
        assert kwargs["viewport"]["height"] == 800

    def test_no_chrome_args(self):
        """Remote profile should not include Chrome launch args."""
        kwargs = get_remote_profile_kwargs()
        assert "args" not in kwargs

    def test_no_headless_setting(self):
        """Remote profile should not include headless setting."""
        kwargs = get_remote_profile_kwargs()
        assert "headless" not in kwargs


@pytest.mark.unit
class TestStealthScripts:
    """Test stealth JavaScript scripts."""

    def test_stealth_scripts_not_empty(self):
        """Should have at least one stealth script."""
        assert len(STEALTH_SCRIPTS) > 0

    def test_combined_scripts_is_string(self):
        """Combined scripts should return a non-empty string."""
        combined = get_stealth_scripts_combined()
        assert isinstance(combined, str)
        assert len(combined) > 0

    def test_webdriver_override_present(self):
        """Should include navigator.webdriver override."""
        combined = get_stealth_scripts_combined()
        assert "navigator" in combined
        assert "webdriver" in combined

    def test_chrome_runtime_stub_present(self):
        """Should include chrome.runtime stub."""
        combined = get_stealth_scripts_combined()
        assert "chrome.runtime" in combined

    def test_plugins_override_present(self):
        """Should include navigator.plugins override."""
        combined = get_stealth_scripts_combined()
        assert "navigator.plugins" in combined or "plugins" in combined

    def test_webgl_spoof_present(self):
        """Should include WebGL vendor/renderer spoofing."""
        combined = get_stealth_scripts_combined()
        assert "WebGLRenderingContext" in combined
        assert "UNMASKED_VENDOR_WEBGL" in combined or "37445" in combined

    def test_all_scripts_are_strings(self):
        """Each script should be a non-empty string."""
        for i, script in enumerate(STEALTH_SCRIPTS):
            assert isinstance(script, str), f"Script {i} is not a string"
            assert len(script.strip()) > 0, f"Script {i} is empty"
