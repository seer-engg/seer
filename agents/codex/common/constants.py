import re


# Target agent constants

TARGET_AGENT_COMMAND = "langgraph dev --host 0.0.0.0"
TARGET_AGENT_PORT = 2024

SUCCESS_PAT = re.compile(r"Server started in \d+(\.\d+)?s")
# Detect failures early by catching the actual error types that appear first
FAIL_PATTERNS = [
    re.compile(r"GraphLoadError"),
    re.compile(r"Failed to load graph"),
    re.compile(r"ModuleNotFoundError"),
    re.compile(r"ImportError"),
    re.compile(r"Application startup failed"),
]
