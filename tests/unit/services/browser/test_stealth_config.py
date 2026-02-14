"""Tests for browser stealth configuration."""
import platform
from unittest.mock import patch

import pytest

from seer.services.browser.stealth_config import (
    CHROME_USER_AGENTS,
    STEALTH_CHROME_ARGS,
    get_headed_stealth_args,
    get_platform_user_agent,
    get_stealth_profile_kwargs,
)


@pytest.mark.unit
class TestStealthChromeArgs:
    """Test stealth Chrome arguments."""

    def test_contains_headless_new(self):
        """Stealth args should contain --headless=new."""
        assert "--headless=new" in STEALTH_CHROME_ARGS

    def test_contains_automation_controlled_disable(self):
        """Stealth args should disable AutomationControlled."""
        assert "--disable-blink-features=AutomationControlled" in STEALTH_CHROME_ARGS

    def test_no_empty_args(self):
        """All args should be non-empty strings."""
        for arg in STEALTH_CHROME_ARGS:
            assert isinstance(arg, str)
            assert len(arg) > 0


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
class TestGetStealthProfileKwargs:
    """Test get_stealth_profile_kwargs function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        kwargs = get_stealth_profile_kwargs()
        assert isinstance(kwargs, dict)

    def test_headless_is_false(self):
        """headless should be False (we use --headless=new in args instead)."""
        kwargs = get_stealth_profile_kwargs()
        assert kwargs["headless"] is False

    def test_contains_args(self):
        """Should contain stealth args."""
        kwargs = get_stealth_profile_kwargs()
        assert "args" in kwargs
        assert "--headless=new" in kwargs["args"]

    def test_contains_user_agent(self):
        """Should contain a user agent string."""
        kwargs = get_stealth_profile_kwargs()
        assert "user_agent" in kwargs
        assert "Chrome/" in kwargs["user_agent"]

    def test_contains_viewport(self):
        """Should contain viewport settings."""
        kwargs = get_stealth_profile_kwargs()
        assert "viewport" in kwargs
        assert kwargs["viewport"]["width"] == 1280
        assert kwargs["viewport"]["height"] == 800


@pytest.mark.unit
class TestGetHeadedStealthArgs:
    """Test get_headed_stealth_args function."""

    def test_excludes_headless_new(self):
        """Should exclude --headless=new for headed mode."""
        args = get_headed_stealth_args()
        assert "--headless=new" not in args

    def test_keeps_other_stealth_args(self):
        """Should keep other stealth args."""
        args = get_headed_stealth_args()
        assert "--disable-blink-features=AutomationControlled" in args

    def test_returns_list(self):
        """Should return a list."""
        args = get_headed_stealth_args()
        assert isinstance(args, list)
