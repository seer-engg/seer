"""Tests for browser profile configuration."""
import platform
from unittest.mock import patch

import pytest

from seer.services.browser.stealth_config import (
    CHROME_USER_AGENTS,
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
