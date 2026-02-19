"""Tests for Langfuse tracing utilities."""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


class TestLangfuseUserContext:
    """Tests for langfuse_user_context context manager."""

    @patch("seer.utilities.langfuse_tracing.config")
    def test_passthrough_when_langfuse_disabled(self, mock_config):
        """Should be a passthrough when Langfuse is disabled."""
        from seer.utilities.langfuse_tracing import langfuse_user_context

        mock_config.langfuse_enabled = False

        # Create mock langfuse module
        mock_propagate = MagicMock()
        mock_langfuse = MagicMock()
        mock_langfuse.propagate_attributes = mock_propagate

        # When langfuse_enabled is False, we should not call propagate_attributes
        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            executed = False
            with langfuse_user_context("user_123"):
                executed = True
            assert executed
            mock_propagate.assert_not_called()

    @patch("seer.utilities.langfuse_tracing.config")
    def test_passthrough_when_user_id_is_none(self, mock_config):
        """Should be a passthrough when user_id is None."""
        from seer.utilities.langfuse_tracing import langfuse_user_context

        mock_config.langfuse_enabled = True

        mock_propagate = MagicMock()
        mock_langfuse = MagicMock()
        mock_langfuse.propagate_attributes = mock_propagate

        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            executed = False
            with langfuse_user_context(None):
                executed = True
            assert executed
            mock_propagate.assert_not_called()

    @patch("seer.utilities.langfuse_tracing.config")
    def test_passthrough_when_user_id_is_empty(self, mock_config):
        """Should be a passthrough when user_id is empty string."""
        from seer.utilities.langfuse_tracing import langfuse_user_context

        mock_config.langfuse_enabled = True

        mock_propagate = MagicMock()
        mock_langfuse = MagicMock()
        mock_langfuse.propagate_attributes = mock_propagate

        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            executed = False
            with langfuse_user_context(""):
                executed = True
            assert executed
            mock_propagate.assert_not_called()

    @patch("seer.utilities.langfuse_tracing.config")
    def test_calls_propagate_attributes_with_user_id(self, mock_config):
        """Should call propagate_attributes with user_id."""
        mock_config.langfuse_enabled = True

        # Create a proper context manager mock
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_propagate = MagicMock(return_value=mock_cm)
        mock_langfuse = MagicMock()
        mock_langfuse.propagate_attributes = mock_propagate

        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            from seer.utilities import langfuse_tracing
            import importlib
            importlib.reload(langfuse_tracing)

            with patch.object(langfuse_tracing, "config", mock_config):
                executed = False
                with langfuse_tracing.langfuse_user_context("user_abc123"):
                    executed = True
                assert executed
                mock_propagate.assert_called_once_with(user_id="user_abc123")

    @patch("seer.utilities.langfuse_tracing.config")
    def test_handles_import_error_gracefully(self, mock_config):
        """Should handle ImportError gracefully when langfuse is not installed."""
        from seer.utilities.langfuse_tracing import langfuse_user_context

        mock_config.langfuse_enabled = True

        # Simulate langfuse not being installed
        with patch.dict("sys.modules", {"langfuse": None}):
            executed = False
            with langfuse_user_context("user_123"):
                executed = True
            # Should not raise and should still execute the body
            assert executed

    @patch("seer.utilities.langfuse_tracing.config")
    @patch("seer.utilities.langfuse_tracing.logger")
    def test_handles_exceptions_gracefully(self, mock_logger, mock_config):
        """Should handle exceptions gracefully and still yield."""
        mock_config.langfuse_enabled = True

        mock_propagate = MagicMock(side_effect=RuntimeError("Test error"))
        mock_langfuse = MagicMock()
        mock_langfuse.propagate_attributes = mock_propagate

        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            from seer.utilities import langfuse_tracing
            import importlib
            importlib.reload(langfuse_tracing)

            with patch.object(langfuse_tracing, "config", mock_config):
                with patch.object(langfuse_tracing, "logger", mock_logger):
                    executed = False
                    with langfuse_tracing.langfuse_user_context("user_123"):
                        executed = True
                    # Should still execute the body even if propagate_attributes fails
                    assert executed
                    mock_logger.debug.assert_called()

    @patch("seer.utilities.langfuse_tracing.config")
    def test_workflow_exceptions_propagate_correctly(self, mock_config):
        """
        Workflow exceptions thrown through the context manager should propagate.

        This tests the fix for the bug where exceptions were suppressed with a yield,
        causing 'RuntimeError: generator didn't stop after throw()'.
        """
        mock_config.langfuse_enabled = True

        # Create a proper mock context manager for propagate_attributes
        @contextmanager
        def mock_propagate_attributes(**_kwargs):
            yield

        mock_langfuse = MagicMock()
        mock_langfuse.propagate_attributes = mock_propagate_attributes

        with patch.dict("sys.modules", {"langfuse": mock_langfuse}):
            from seer.utilities import langfuse_tracing
            import importlib
            importlib.reload(langfuse_tracing)

            with patch.object(langfuse_tracing, "config", mock_config):
                # Exception raised inside the context should propagate
                with pytest.raises(ValueError, match="workflow error"):
                    with langfuse_tracing.langfuse_user_context("user_123"):
                        raise ValueError("workflow error")
