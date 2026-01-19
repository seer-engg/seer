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

import logging
import sys

# Global cache of loggers
_loggers = {}

import json

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
        if hasattr(record, 'correlation_id') and record.correlation_id:
            record.msg = f"[{record.correlation_id[:8]}] {record.msg}"

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
