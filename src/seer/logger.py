# pylint: disable=too-complex,wrong-import-position,wrong-import-order,invalid-name,global-statement
# pylint: disable=import-outside-toplevel,broad-exception-caught,broad-exception-raised,unused-variable
# pylint: disable=unused-argument,unused-import,no-else-break,logging-fstring-interpolation,protected-access
# Reason: Logger requires complex formatting, lazy config, global state; raises general Exception for critical errors
"""
Simple logging module for Seer project.

All services log to console (stdout) with colored, structured output.
run.py captures console output from subprocesses to log files automatically.

Usage:
    from seer.shared.logger import get_logger

    logger = get_logger(__name__)  # Use module name
    # or
    logger = get_logger('my_component')  # Use custom name

    logger.info("Message here")
"""

import asyncio
import logging
import os
import socket
import sys
import threading
import traceback
from typing import Optional

import httpx

# Global cache of loggers
_loggers = {}

import json

# Lazy import to avoid circular dependencies
_config = None


def _get_config():
    """Lazy import of config to avoid circular dependencies."""
    global _config
    if _config is None:
        from seer.config import config as _config_instance
        _config = _config_instance
    return _config

class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors and dynamically includes 'extra' fields"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    # Standard LogRecord attributes to ignore when looking for 'extra' fields
    RESERVED_ATTRS = {
        'args', 'asctime', 'created', 'exc_info', 'filename', 'funcName',
        'levelname', 'levelno', 'lineno', 'module', 'msecs', 'msg',
        'name', 'pathname', 'process', 'processName', 'relativeCreated',
        'stack_info', 'thread', 'threadName', 'correlation_id'
    }

    def format(self, record):
        # 1. Handle Correlation ID (logic from your original code)
        correlation_id = getattr(record, 'correlation_id', None)
        if correlation_id:
            record.msg = f"[{correlation_id[:8]}] {record.msg}"

        # 2. Capture arbitrary "extra" fields
        # We look for anything in record.__dict__ that isn't a standard attribute
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self.RESERVED_ATTRS and not k.startswith('_')
        }

        # 3. Append extras to the message
        if extras:
            # Format as key=value pairs
            extra_str = " ".join([f"{k}={v}" for k, v in extras.items()])
            # We modify a temporary attribute to avoid polluting the record for other handlers
            original_msg = record.msg
            record.msg = f"{original_msg} | {extra_str}"

        # 4. Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # 5. Call the standard formatter
        result = super().format(record)

        # Clean up: restore original message so if this record is reused,
        # it doesn't keep appending extras
        if extras:
            record.msg = original_msg

        return result


class SlackHandler(logging.Handler):
    """
    Custom logging handler that sends ERROR and CRITICAL messages to Slack
    asynchronously in a non-blocking fashion.
    """

    def __init__(self, bot_token: str, channel_id: str):
        super().__init__()
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.setLevel(logging.ERROR)  # Only handle ERROR and CRITICAL
        self._http_client: Optional[httpx.AsyncClient] = None

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to Slack asynchronously.
        This method is called by the logging system and should not block.
        """
        try:
            # Only process ERROR and CRITICAL levels
            if record.levelno < logging.ERROR:
                return

            # Try to get the running event loop (Python 3.7+)
            try:
                loop = asyncio.get_running_loop()
                # Event loop is running, create task (non-blocking)
                asyncio.create_task(self._send_slack_notification(record))
            except RuntimeError:
                # No running event loop - use background thread to avoid blocking
                def run_in_thread():
                    try:
                        # Create new event loop in this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            new_loop.run_until_complete(self._send_slack_notification(record))
                        finally:
                            new_loop.close()
                    except Exception as e:
                        # Fallback: log to console if async setup fails
                        self._log_fallback(f"Failed to setup async context for Slack notification: {e}", record)

                # Start background thread (daemon=True so it doesn't prevent shutdown)
                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()
        except Exception as e:
            # Silent failure - log to console only
            self._log_fallback(f"SlackHandler.emit failed: {e}", record)

    async def _send_slack_notification(self, record: logging.LogRecord) -> None:
        """
        Send a formatted log record to Slack via the Bot API.
        Main message is sent first, then stack trace (if any) is sent as a thread reply.
        This method handles all errors silently and logs to console on failure.
        Uses simple plain text format for maximum fault tolerance.
        """
        try:
            # Format the main message (without stack trace)
            main_message = self._format_slack_main_message(record)
            stack_trace = self._extract_stack_trace(record)

            # Prepare the API request - use simple text format
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "channel": self.channel_id,
                "text": main_message,  # Simple text format - most reliable
            }

            # Create HTTP client if needed
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=5.0)

            # Send main message with retry logic (1 retry)
            max_retries = 2
            last_error = None
            thread_ts = None

            for attempt in range(max_retries):
                try:
                    response = await self._http_client.post(url, json=payload, headers=headers)
                    response.raise_for_status()

                    result = response.json()
                    if result.get("ok"):
                        # Get thread timestamp for potential thread reply
                        thread_ts = result.get("ts")
                        break  # Success
                    else:
                        error_msg = result.get("error", "Unknown error")
                        # If blocks failed, try with even simpler format
                        if attempt == 0 and "invalid_blocks" in str(error_msg).lower():
                            # Fallback to absolute simplest format
                            payload = {
                                "channel": self.channel_id,
                                "text": f"{record.levelname} in {record.name}: {record.getMessage()}",
                            }
                            continue
                        raise Exception(f"Slack API error: {error_msg}")

                except (httpx.HTTPError, httpx.TimeoutException) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        # Wait a bit before retry
                        await asyncio.sleep(0.5)
                    continue

            # If main message failed, don't try to send thread
            if thread_ts is None:
                raise last_error or Exception("Failed to send Slack notification")

            # If we have a stack trace, send it as a thread reply
            if stack_trace:
                try:
                    thread_payload = {
                        "channel": self.channel_id,
                        "thread_ts": thread_ts,  # Reply in thread
                        "text": f"Stack Trace:\n{stack_trace}",
                    }
                    # Send thread reply (don't retry - if it fails, main message is already sent)
                    try:
                        response = await self._http_client.post(url, json=thread_payload, headers=headers)
                        response.raise_for_status()
                        result = response.json()
                        if not result.get("ok"):
                            # Silent failure for thread - main message was already sent
                            self._log_fallback(f"Failed to send stack trace thread: {result.get('error')}", record)
                    except Exception as e:
                        # Silent failure for thread - main message was already sent
                        self._log_fallback(f"Failed to send stack trace thread: {e}", record)
                except Exception as e:
                    # Silent failure for thread - main message was already sent
                    self._log_fallback(f"Failed to format stack trace thread: {e}", record)

        except Exception as e:
            # Silent failure - log to console only
            self._log_fallback(f"Failed to send Slack notification: {e}", record)

    def _format_slack_main_message(self, record: logging.LogRecord) -> str:
        """
        Format the main log message (without stack trace) for Slack.
        Stack trace will be sent separately as a thread reply.
        This is fault-tolerant and avoids complex Block Kit formatting issues.
        """
        try:
            # Build message parts
            parts = []

            # Level and logger name
            level_emoji = "🔴" if record.levelno == logging.CRITICAL else "⚠️"
            parts.append(f"{level_emoji} {record.levelname} in {record.name}")
            parts.append("=" * 60)

            # Main message
            message = record.getMessage()
            correlation_id = getattr(record, 'correlation_id', None)
            if correlation_id:
                message = f"[{correlation_id[:8]}] {message}"
            parts.append(f"Message: {message}")

            # Format timestamp
            try:
                if not record.asctime:
                    record.asctime = logging.Formatter().formatTime(record)
                timestamp = record.asctime
            except Exception:
                timestamp = "unknown"

            parts.append(f"Timestamp: {timestamp}")
            parts.append(f"Module: {record.module}")
            if record.filename:
                parts.append(f"File: {record.filename}:{record.lineno}")

            # If there's exception info, mention it briefly (full trace goes in thread)
            if record.exc_info:
                try:
                    if record.exc_info and len(record.exc_info) >= 2:
                        exc_type = record.exc_info[0]
                        exc_value = record.exc_info[1]
                        exc_name = getattr(exc_type, '__name__', str(exc_type))
                        parts.append(f"Exception: {exc_name}: {exc_value}")
                        parts.append("(Full stack trace in thread)")
                except Exception:
                    pass

            # Join all parts
            message_text = "\n".join(parts)

            # Slack has a message limit of ~4000 characters, truncate if needed
            if len(message_text) > 3900:
                message_text = message_text[:3900] + "\n... (message truncated)"

            return message_text

        except Exception as e:
            # Ultimate fallback - if even formatting fails, send minimal message
            try:
                return f"{record.levelname} in {record.name}: {record.getMessage()}"
            except Exception:
                return f"Error log from {record.name}"

    def _extract_stack_trace(self, record: logging.LogRecord) -> Optional[str]:
        """
        Extract and format the stack trace from a log record.
        Returns None if no stack trace is available.
        """
        if not record.exc_info:
            return None

        try:
            exc_type, exc_value, exc_traceback = record.exc_info
            if exc_traceback:
                stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                # Slack thread messages can be longer, but still truncate if extremely long
                if len(stack_trace) > 8000:
                    stack_trace = stack_trace[:8000] + "\n... (truncated)"
                return stack_trace
        except Exception:
            # If stack trace formatting fails, return None
            pass

        return None

    def _log_fallback(self, message: str, original_record: Optional[logging.LogRecord] = None) -> None:
        """
        Fallback logging to console when Slack notification fails.
        Uses a simple console logger to avoid recursion.
        """
        try:
            # Use a simple formatter to avoid circular dependencies
            console_logger = logging.getLogger("seer.logger.slack_fallback")
            console_logger.setLevel(logging.WARNING)
            if not console_logger.handlers:
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(logging.Formatter('%(asctime)s | SLACK_HANDLER | ERROR | %(message)s'))
                console_logger.addHandler(handler)
            console_logger.error(message)
        except Exception:
            # Last resort: print to stderr
            print(f"SLACK_HANDLER ERROR: {message}", file=sys.stderr)

    def close(self) -> None:
        """Clean up resources when handler is closed."""
        super().close()
        if self._http_client:
            try:
                # Close the HTTP client
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._http_client.aclose())
                else:
                    loop.run_until_complete(self._http_client.aclose())
            except Exception:
                pass
            finally:
                self._http_client = None


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger that outputs colored, structured logs to console.

    Args:
        name: Logger name (typically __name__ or component name)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Return cached logger if it exists
    if name in _loggers:
        return _loggers[name]

    # Create new logger
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False

    # Clear any existing handlers
    log.handlers.clear()

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    log.addHandler(console_handler)

    # Add Slack handler if configured and enabled
    config = _get_config()
    if (
        config.slack_notifications_enabled
        and config.is_slack_configured
        and config.slack_bot_token
        and config.slack_error_channel_id
    ):
        try:
            slack_handler = SlackHandler(
                bot_token=config.slack_bot_token,
                channel_id=config.slack_error_channel_id
            )
            log.addHandler(slack_handler)
        except Exception as e:
            # Silent failure - log to console only
            fallback_logger = logging.getLogger("seer.logger.slack_setup")
            fallback_logger.warning(f"Failed to add Slack handler: {e}")

    # Cache and return
    _loggers[name] = log
    return log


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = get_logger('test')
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
