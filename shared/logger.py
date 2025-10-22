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


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_level = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = colored_level
        
        return super().format(record)


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
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # Cache and return
    _loggers[name] = logger
    return logger


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = get_logger('test')
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
