from __future__ import annotations

import os
import re
import time
import shlex
import uuid
from typing import Optional, Tuple

import httpx
import base64
from urllib.parse import quote as urlquote

from shared.logger import get_logger
from e2b_code_interpreter import AsyncSandbox, CommandResult
from agents.codex.state import CodexState
from shared.config import config

logger = get_logger("codex.nodes.raise_pr")


def _parse_github_owner_repo(repo_url: str) -> Optional[Tuple[str, str]]:
    """Extract (owner, repo) from common GitHub URL formats.

    Supports:
    - https://github.com/owner/repo.git
    - https://github.com/owner/repo
    - git@github.com:owner/repo.git
    - ssh://git@github.com/owner/repo.git
    Returns None if host is not github.com or cannot be parsed.
    """
    try:
        # Normalize
        url = repo_url.strip()

        # SSH form: git@github.com:owner/repo.git
        m = re.match(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", url)
        if m:
            return m.group("owner"), m.group("repo")

        # SSH with scheme: ssh://git@github.com/owner/repo.git
        m = re.match(r"^ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", url)
        if m:
            return m.group("owner"), m.group("repo")

        # HTTPS form: https://github.com/owner/repo(.git)?
        m = re.match(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?(?:/)?$", url)
        if m:
            return m.group("owner"), m.group("repo")
    except Exception:
        pass
    return None


async def _run_in_sandbox(sbx: AsyncSandbox, command: str, repo_dir: str) -> CommandResult:
    """Run a command in the sandbox within the repo directory and return the result."""
    res: CommandResult = await sbx.commands.run(command, cwd=repo_dir)
    return res


def _masked(s: str) -> str:
    token = config.github_token or ""
    if not token:
        return s
    return s.replace(token, "***")


async def raise_pr(state: CodexState) -> CodexState:
    """Commit local changes in sandbox, push a branch, and open a GitHub PR.

    Requires in state:
    - sandbox_session_id: E2B sandbox id
    - repo_path: path to repo inside sandbox
    - repo_url: remote repository URL (GitHub)
    - branch_name: base branch name (defaults to main)
    """

    github_token = config.github_token
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN not configured in environment")

    sandbox_id = state.context.sandbox_context.sandbox_id
    repo_dir = state.context.sandbox_context.working_directory
    repo_url = state.context.github_context.repo_url
    base_branch = state.context.github_context.branch_name or "main"
    new_version = state.context.target_agent_version + 1

    # Generate branch name and commit message
    ts = time.strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:7]
    new_branch = f"seer/codex/{ts}-{short_id}/v{new_version}"
    req_snippet = (state.context.user_context.raw_request or "Automated update").strip().replace("\n", " ")
    if len(req_snippet) > 72:
        req_snippet = req_snippet[:69] + "..."
    commit_msg = f"chore(seer): {req_snippet}"

    logger.info(
        _masked(
            f"Preparing to commit and push from sandbox {sandbox_id} at {repo_dir} to branch {new_branch} (base {base_branch})"
        )
    )

    # Connect to sandbox
    sbx: AsyncSandbox = await AsyncSandbox.connect(sandbox_id)

    # 1) Check for changes
    status_res = await _run_in_sandbox(sbx, "git status --porcelain", repo_dir)
    if status_res.exit_code != 0:
        logger.error(f"git status failed: {status_res.stderr or status_res.stdout}")
        raise RuntimeError("Failed to check repository status in sandbox")

    if not (status_res.stdout or "").strip():
        # No changes: add a message and return
        msgs = list(state.messages)
        msgs.append({
            "role": "system",
            "content": "No changes detected in working directory; skipping PR creation.",
        })
        return {"messages": msgs}

    # 2) Ensure we are on a new branch
    res_checkout = await _run_in_sandbox(sbx, f"git checkout -B {shlex.quote(new_branch)}", repo_dir)
    if res_checkout.exit_code != 0:
        logger.error(f"git checkout failed: {res_checkout.stderr or res_checkout.stdout}")
        raise RuntimeError("Failed to create/switch to PR branch in sandbox")

    # 3) Configure author (local)
    await _run_in_sandbox(sbx, "git config user.email 'lokeshdanu9@gmail.com'", repo_dir)
    await _run_in_sandbox(sbx, "git config user.name 'lokesh-danu'", repo_dir)

    # 4) Add & commit
    res_add = await _run_in_sandbox(sbx, "git add -A", repo_dir)
    if res_add.exit_code != 0:
        logger.error(f"git add failed: {res_add.stderr or res_add.stdout}")
        raise RuntimeError("Failed to stage changes in sandbox")

    # If nothing staged after add, skip
    res_diff_cached = await _run_in_sandbox(sbx, "git diff --cached --quiet || echo CHANGES_STAGED", repo_dir)
    if "CHANGES_STAGED" not in (res_diff_cached.stdout or ""):
        msgs = list(state.messages)
        msgs.append({
            "role": "system",
            "content": "No staged changes after add; skipping PR creation.",
        })
        return {"messages": msgs}

    res_commit = await _run_in_sandbox(sbx, f"git commit -m {shlex.quote(commit_msg)}", repo_dir)
    if res_commit.exit_code != 0:
        # If commit fails due to nothing to commit, skip
        out = (res_commit.stderr or res_commit.stdout or "").lower()
        if "nothing to commit" not in out:
            logger.error(f"git commit failed: {res_commit.stderr or res_commit.stdout}")
            raise RuntimeError("Failed to commit changes in sandbox")

    # 5) Push branch non-interactively with temporary Basic auth header; fallback to tokenized URL
    branch_q = shlex.quote(new_branch)
    auth_b64 = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
    auth_b64_q = shlex.quote(auth_b64)
    primary_push_cmd = (
        "AUTH_B64="
        + auth_b64_q
        + " bash -lc 'set -e; export GIT_TERMINAL_PROMPT=0; "
        + "git config --local http.extraHeader \"Authorization: Basic $AUTH_B64\"; "
        + f"git push -u origin {branch_q}; "
        + "git config --local --unset-all http.extraHeader || true'"
    )
    res_push = await _run_in_sandbox(sbx, primary_push_cmd, repo_dir)
    if res_push.exit_code != 0:
        # Fallback: push using tokenized remote URL (do not persist credentials)
        owner_repo = _parse_github_owner_repo(repo_url)
        if owner_repo is None:
            logger.error("git push failed and remote is not GitHub-compatible")
            raise RuntimeError("Failed to push branch to origin from sandbox")
        owner, repo = owner_repo
        token_enc = urlquote(github_token, safe="")
        tokenized_url = f"https://x-access-token:{token_enc}@github.com/{owner}/{repo}.git"
        res_push2 = await _run_in_sandbox(
            sbx,
            f"GIT_TERMINAL_PROMPT=0 git push -u {shlex.quote(tokenized_url)} {branch_q}",
            repo_dir,
        )
        if res_push2.exit_code != 0:
            logger.error("Fallback push with tokenized URL failed")
            raise RuntimeError("Failed to push branch to origin from sandbox")

    # 6) Create GitHub PR from host using API
    owner_repo = _parse_github_owner_repo(repo_url)
    pr_url: Optional[str] = None
    if owner_repo is not None:
        owner, repo = owner_repo
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        title = f"Seer: {req_snippet}"
        body = (
            "Automated PR created by Seer Codex.\n\n"
            f"- Base: `{base_branch}`\n"
            f"- Head: `{new_branch}`\n"
            "- Changes were implemented in an isolated sandbox and pushed automatically."
            f"- PR Summary: {state.pr_summary}"
        )

        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
        }
        payload = {"title": title, "head": new_branch, "base": base_branch, "body": body}

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(api_url, headers=headers, json=payload)
            if resp.status_code == 201:
                pr_url = resp.json().get("html_url")
            elif resp.status_code == 422:
                # Possibly already exists. Try to find existing PR
                list_url = (
                    f"https://api.github.com/repos/{owner}/{repo}/pulls?head={owner}:{new_branch}&state=open"
                )
                list_resp = await client.get(list_url, headers=headers)
                if list_resp.status_code == 200 and isinstance(list_resp.json(), list):
                    items = list_resp.json()
                    if items:
                        pr_url = items[0].get("html_url")
                # If still no URL, fall back to compare link
                if not pr_url:
                    pr_url = f"https://github.com/{owner}/{repo}/compare/{base_branch}...{new_branch}?expand=1"
            else:
                logger.warning(
                    _masked(f"PR creation failed ({resp.status_code}): {resp.text}")
                )
                # Fallback to compare link
                pr_url = f"https://github.com/{owner}/{repo}/compare/{base_branch}...{new_branch}?expand=1"
    else:
        logger.warning(f"Non-GitHub repo URL; PR creation via API skipped: {repo_url}")

    # Update messages
    msgs = list(state.messages)
    if pr_url:
        msgs.append({
            "role": "system",
            "content": f"Opened PR: {pr_url}",
        })
    else:
        msgs.append({
            "role": "system",
            "content": f"Pushed branch '{new_branch}'. Please open a PR manually.",
        })
    
    state.context.sandbox_context.working_branch = new_branch

    return {
        "messages": msgs,
        "new_branch_name": new_branch,
        "context": state.context,
        'agent_updated': True,
        'target_agent_version': new_version,
    }


