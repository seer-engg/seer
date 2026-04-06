"""Tests for browser profile configuration."""
import platform
from unittest.mock import patch

import pytest

from seer.services.browser.stealth_config import (
    CHROME_USER_AGENTS,
    find_local_chromium_executable,
    get_local_profile_kwargs,
    get_platform_user_agent,
    get_remote_profile_kwargs,
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
class TestFindLocalChromiumExecutable:
    """Tests for find_local_chromium_executable."""

    @patch("seer.services.browser.stealth_config.platform.system", return_value="Linux")
    @patch("seer.services.browser.stealth_config.glob.glob")
    @patch.dict("os.environ", {"PLAYWRIGHT_BROWSERS_PATH": "/opt/pw-browsers"})
    def test_finds_via_playwright_browsers_path(self, mock_glob, _mock_system):
        """Should find Chromium using PLAYWRIGHT_BROWSERS_PATH env var."""
        mock_glob.side_effect = lambda pattern: (
            ["/opt/pw-browsers/chromium-1194/chrome-linux/chrome"]
            if "chromium-*" in pattern
            else []
        )
        result = find_local_chromium_executable()
        assert result == "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

    @patch("seer.services.browser.stealth_config.platform.system", return_value="Linux")
    @patch("seer.services.browser.stealth_config.glob.glob", return_value=[])
    @patch("seer.services.browser.stealth_config.os.path.isfile")
    @patch("seer.services.browser.stealth_config.os.access")
    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_to_system_chrome(self, mock_access, mock_isfile, _mock_glob, _mock_system):
        """Should fall back to system Chrome when playwright path not set."""
        mock_isfile.side_effect = lambda p: p == "/usr/bin/google-chrome-stable"
        mock_access.return_value = True
        result = find_local_chromium_executable()
        assert result == "/usr/bin/google-chrome-stable"

    @patch("seer.services.browser.stealth_config.platform.system", return_value="Linux")
    @patch("seer.services.browser.stealth_config.glob.glob", return_value=[])
    @patch("seer.services.browser.stealth_config.os.path.isfile", return_value=False)
    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_when_not_found(self, _mock_isfile, _mock_glob, _mock_system):
        """Should return None when no browser is found."""
        result = find_local_chromium_executable()
        assert result is None

    @patch("seer.services.browser.stealth_config.platform.system", return_value="Windows")
    def test_returns_none_on_unsupported_platform(self, _mock_system):
        """Should return None on unsupported platforms (e.g. Windows)."""
        result = find_local_chromium_executable()
        assert result is None

    @patch("seer.services.browser.stealth_config.platform.system", return_value="Linux")
    @patch("seer.services.browser.stealth_config.glob.glob")
    @patch.dict("os.environ", {"PLAYWRIGHT_BROWSERS_PATH": "/opt/pw-browsers"})
    def test_picks_highest_version(self, mock_glob, _mock_system):
        """Should pick the alphanumerically highest version when multiple are installed."""
        mock_glob.side_effect = lambda pattern: (
            [
                "/opt/pw-browsers/chromium-1100/chrome-linux/chrome",
                "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
            ]
            if "chromium-*" in pattern
            else []
        )
        result = find_local_chromium_executable()
        assert result == "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


@pytest.mark.unit
class TestGetLocalProfileKwargs:
    """Tests for get_local_profile_kwargs."""

    @patch(
        "seer.services.browser.stealth_config.find_local_chromium_executable",
        return_value="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    )
    def test_includes_executable_path_when_found(self, _mock_find):
        """Should include executable_path when browser is found."""
        kwargs = get_local_profile_kwargs()
        assert kwargs["executable_path"] == "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

    @patch(
        "seer.services.browser.stealth_config.find_local_chromium_executable",
        return_value=None,
    )
    def test_omits_executable_path_when_not_found(self, _mock_find):
        """Should omit executable_path when no browser is found (let browser_use search)."""
        kwargs = get_local_profile_kwargs()
        assert "executable_path" not in kwargs

    @patch(
        "seer.services.browser.stealth_config.find_local_chromium_executable",
        return_value="/some/chrome",
    )
    def test_still_contains_viewport_and_user_agent(self, _mock_find):
        """Local profile should always include viewport and user_agent."""
        kwargs = get_local_profile_kwargs()
        assert "user_agent" in kwargs
        assert "viewport" in kwargs
        assert kwargs["viewport"] == {"width": 1280, "height": 800}
