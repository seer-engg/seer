"""Sandbox constants - imports from shared.config for consistency"""
import re
import textwrap

async def _build_git_shell_script() -> str:
    # Sandbox shell script: try unauthenticated Git first (public repos), then fallback to Basic auth header if needed.
    # Disable interactive prompts to avoid hangs in headless sandbox.
    script = """
set -euo pipefail
export GIT_TERMINAL_PROMPT=0

git config --global user.email 'lokeshdanu9@gmail.com'
git config --global user.name 'lokesh-danu'

: "${REPO_URL:?REPO_URL is required}"
BRANCH="${BRANCH:-main}"
TOKEN="${TOKEN:-}"

REPO_DIR="$(basename "$REPO_URL")"
REPO_DIR="${REPO_DIR%.git}"

# Build Basic auth header if token present (username: x-access-token, password: token)
AUTH_HEADER=""
if [ -n "$TOKEN" ]; then
  AUTH_B64=$(printf "x-access-token:%s" "$TOKEN" | base64 -w 0 2>/dev/null || printf "x-access-token:%s" "$TOKEN" | base64)
  AUTH_HEADER="Authorization: Basic $AUTH_B64"
fi

# Clone unauthenticated first; fallback to header if it fails
if [ ! -d "$REPO_DIR/.git" ]; then
  if ! git clone "$REPO_URL"; then
    if [ -n "$AUTH_HEADER" ]; then
      git -c "http.extraHeader=$AUTH_HEADER" clone "$REPO_URL"
    else
      echo "Clone failed and no token available" >&2
      exit 1
    fi
  fi
fi

cd "$REPO_DIR"

# Fetch branch; ignore failure if branch doesn't exist yet. Fallback with header if needed.
git fetch origin "$BRANCH" || {
  if [ -n "$AUTH_HEADER" ]; then
    git -c "http.extraHeader=$AUTH_HEADER" fetch origin "$BRANCH" || true
  else
    true
  fi
}

if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout "$BRANCH"
elif git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  git checkout -B "$BRANCH" "origin/$BRANCH"
else
  git checkout -b "$BRANCH"
fi

# Pull latest; ignore if nothing to pull. Fallback with header if needed.
git pull --ff-only origin "$BRANCH" || {
  if [ -n "$AUTH_HEADER" ]; then
    git -c "http.extraHeader=$AUTH_HEADER" pull --ff-only origin "$BRANCH" || true
  else
    true
  fi
}

echo "SANDBOX_REPO_DIR=$(pwd)"
echo "SANDBOX_BRANCH=$BRANCH"
"""
    return textwrap.dedent(script)


SUCCESS_PAT = re.compile(r"Server started in \d+(\.\d+)?s")
# Detect failures early by catching the actual error types that appear first
FAIL_PATTERNS = [
    re.compile(r"GraphLoadError"),
    re.compile(r"Failed to load graph"),
    re.compile(r"ModuleNotFoundError"),
    re.compile(r"ImportError"),
    re.compile(r"Application startup failed"),
]
TARGET_AGENT_PORT = 2024