"""
Extract Nexus chat sessions from prod DB for evaluation.

Pulls the last N weeks of real user sessions with messages, proposals,
and metadata. Outputs one JSON per session for eval ground truth.

Supports anonymization to remove PII before committing to version control.

Run:
    uv run python scripts/extract_nexus_threads.py              # from prod
    uv run python scripts/extract_nexus_threads.py --anonymize  # strip PII
    DATABASE_URL="..." uv run python scripts/extract_nexus_threads.py  # custom DB
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tortoise import Tortoise

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "eval" / "nexus"
OUTPUT_DIR.mkdir(exist_ok=True)

# Skip obviously spam/test sessions by ID
SKIP_SESSION_IDS = set()  # populated after data inspection

# Look-back window
WEEKS = 12


def _anonymize_email(email: str) -> str:
    """Replace email with user_{hash[:8]}."""
    h = hashlib.sha256(email.encode()).hexdigest()[:8]
    return f"user_{h}"


# Regex for email addresses in free-text content
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def _strip_emails(text: str) -> str:
    """Replace all email addresses in text with [email]."""
    return _EMAIL_RE.sub("[email]", text)


async def init_db() -> None:
    """Init Tortoise with prod DB using the app's full model config."""
    from seer.database.config import TORTOISE_ORM, _parse_postgres_credentials

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    # Override the default connection with the provided DATABASE_URL
    import copy
    config = copy.deepcopy(TORTOISE_ORM)
    config["connections"]["default"]["credentials"] = _parse_postgres_credentials(db_url)

    await Tortoise.init(config=config)


async def extract_sessions(anonymize: bool = False) -> List[Dict[str, Any]]:
    """Extract sessions from the last N weeks."""
    from seer.database.workflow_models import (
        WorkflowChatMessage,
        WorkflowChatSession,
        WorkflowProposal,
    )
    from seer.database.models import User

    cutoff = datetime.now(timezone.utc) - timedelta(weeks=WEEKS)

    sessions = await WorkflowChatSession.filter(
        created_at__gte=cutoff,
    ).order_by("id")

    results = []
    for session in sessions:
        # Get user email
        user = await User.filter(id=session.user_id).first()
        user_email = user.email if user else f"user_{session.user_id}"

        # Get messages
        messages = await WorkflowChatMessage.filter(
            session_id=session.id,
        ).order_by("created_at")

        # Get proposals for this session
        proposals = await WorkflowProposal.filter(
            session_id=session.id,
        ).order_by("created_at")

        # Skip spam/test sessions
        first_msg = next((m for m in messages if m.role == "user"), None)
        if first_msg and len(first_msg.content.strip()) < 5:
            print(f"  Skipping session {session.id}: spam/empty message")
            continue

        # Determine final outcome
        proposal_statuses = [p.status for p in proposals]
        if "accepted" in proposal_statuses:
            outcome = "accepted"
        elif "rejected" in proposal_statuses and "accepted" not in proposal_statuses:
            outcome = "rejected"
        elif proposal_statuses:
            outcome = "pending"
        else:
            outcome = "no_proposal"

        # Anonymize user email if requested
        safe_email = _anonymize_email(user_email) if anonymize else user_email

        session_data = {
            "session_id": session.id,
            "workflow_id": session.workflow_id,
            "user_email": safe_email,
            "title": session.title,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "messages": [
                {
                    "role": m.role,
                    "content": _strip_emails(m.content) if anonymize else m.content,
                    "ts": m.created_at.isoformat() if m.created_at else None,
                    "has_proposal": m.proposal_id is not None,
                }
                for m in messages
            ],
            "proposals": [
                {
                    "id": p.id,
                    "status": p.status,
                    "summary": _strip_emails(p.summary) if anonymize and p.summary else p.summary,
                    "spec": p.spec,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "decided_at": p.decided_at.isoformat() if p.decided_at else None,
                }
                for p in proposals
            ],
            "outcome": outcome,
            "execution_status": session.current_execution_status,
            "message_count": len(messages),
            "proposal_count": len(proposals),
        }
        results.append(session_data)

    return results


def _has_tool_signal(name: str, message: str) -> bool:
    """Check if message contains enough signal to find a tool/trigger by name.

    Requires at least 2 distinct parts to match for tool names with 3+ parts,
    avoiding false positives from common words like "create" covering all *_create tools.
    """
    msg_lower = message.lower()
    parts = [p for p in name.replace(".", "_").replace("-", "_").split("_") if len(p) > 2]
    if not parts:
        return False
    matching = sum(1 for p in parts if p in msg_lower)
    # For names with 3+ parts, need at least 2 matches to avoid single-common-word matches
    min_required = 2 if len(parts) >= 3 else 1
    return matching >= min(min_required, len(parts))


def auto_label(session: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-seed ground truth labels from session data."""
    messages = session["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    first_user_msg = user_msgs[0]["content"] if user_msgs else ""

    # Classify intent from message patterns
    intent = "unknown"
    error_category = None
    if first_user_msg:
        lower = first_user_msg.lower()
        if any(kw in lower for kw in ["fix", "error", "broken", "400", "500", "fail"]):
            intent = "fix_broken_workflow"
            error_category = "runtime_error"
        elif any(kw in lower for kw in ["edit", "update", "change", "add", "also", "remove"]):
            intent = "edit_existing"
        elif any(kw in lower for kw in ["debug", "why", "not working", "shows", "should be"]):
            intent = "debug_behavior"
        elif any(kw in lower for kw in ["format", "formatted", "looks wrong", "template"]):
            intent = "format_quality"
        elif any(kw in lower for kw in ["browser", "browse", "scan", "linkedin", "scrape"]):
            intent = "browser_agent"
        else:
            intent = "new_workflow"

    # Extract tools/triggers from accepted proposals
    expected_tools = []
    expected_triggers = []
    for p in session["proposals"]:
        if p["status"] == "accepted" and p["spec"]:
            spec = p["spec"]
            for node in spec.get("nodes", []):
                if node.get("type") == "tool" and node.get("tool"):
                    tool_name = node["tool"]
                    if tool_name not in expected_tools:
                        expected_tools.append(tool_name)
            for trigger in spec.get("triggers", []):
                if trigger.get("key") and trigger["key"] not in expected_triggers:
                    expected_triggers.append(trigger["key"])

    # Determine expected success
    expected_success = session["outcome"] == "accepted"

    # Classify context_type: standalone vs context_dependent
    # Sessions where the first message doesn't clearly mention the expected tools/triggers
    # are context-dependent — the real agent reads the existing workflow first,
    # so search-only eval is unfair for these.
    context_type = "standalone"
    if expected_tools or expected_triggers:
        tools_covered = sum(1 for t in expected_tools if _has_tool_signal(t, first_user_msg))
        tools_total = len(expected_tools) or 1
        triggers_covered = sum(1 for t in expected_triggers if _has_tool_signal(t, first_user_msg))
        triggers_total = len(expected_triggers) or 1

        tool_alignment = tools_covered / tools_total
        trigger_alignment = triggers_covered / triggers_total

        # If less than half the expected tools/triggers have signal in the message,
        # this is context-dependent — the agent is working off existing workflow state
        if tool_alignment <= 0.5 or trigger_alignment <= 0.5:
            context_type = "context_dependent"

    return {
        "session_id": session["session_id"],
        "user_email": session["user_email"],
        "intent": intent,
        "first_user_message": first_user_msg[:500] if first_user_msg else "",
        "expected_tools": expected_tools,
        "expected_triggers": expected_triggers,
        "expected_success": expected_success,
        "error_category": error_category,
        "outcome": session["outcome"],
        "context_type": context_type,
        "needs_review": True,  # always needs human review
    }


def _print_coverage_report(ground_truth: List[Dict[str, Any]]) -> None:
    """Print tool/trigger coverage report for eval sessions."""
    tool_counts: Counter = Counter()
    trigger_counts: Counter = Counter()
    sessions_with_tools = 0
    sessions_with_triggers = 0
    accepted_sessions = [gt for gt in ground_truth if gt["outcome"] == "accepted"]

    for gt in accepted_sessions:
        if gt["expected_tools"]:
            sessions_with_tools += 1
            for t in gt["expected_tools"]:
                tool_counts[t] += 1
        if gt["expected_triggers"]:
            sessions_with_triggers += 1
            for t in gt["expected_triggers"]:
                trigger_counts[t] += 1

    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    print(f"Accepted sessions with ground truth: {len(accepted_sessions)}")
    print(f"  With tools:   {sessions_with_tools}")
    print(f"  With triggers: {sessions_with_triggers}")
    print()

    print("Top tools by frequency:")
    for tool, count in tool_counts.most_common(20):
        print(f"  {tool:<40} {count:>3} uses")

    print()
    print("Top triggers by frequency:")
    for trigger, count in trigger_counts.most_common(15):
        print(f"  {trigger:<40} {count:>3} uses")

    # Tools with >=3 uses that have eval coverage
    high_use_tools = {t for t, c in tool_counts.items() if c >= 3}
    print(f"\nTool coverage: {len(high_use_tools)} tools with >=3 uses")
    print(f"Trigger coverage: {len(trigger_counts)} distinct triggers")
    print("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Nexus sessions for eval")
    parser.add_argument("--anonymize", action="store_true", help="Strip PII from extracted data")
    args = parser.parse_args()

    print(f"Extracting Nexus sessions from last {WEEKS} weeks...")
    if args.anonymize:
        print("  Anonymization: ON (emails stripped)")
    await init_db()

    sessions = await extract_sessions(anonymize=args.anonymize)
    print(f"Found {len(sessions)} sessions")

    # Save raw session data
    raw_path = OUTPUT_DIR / "raw_sessions.json"
    raw_path.write_text(json.dumps(sessions, indent=2, default=str))
    print(f"  Raw data: {raw_path}")

    # Auto-seed ground truth
    ground_truth = [auto_label(s) for s in sessions]
    gt_path = OUTPUT_DIR / "ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2, default=str))
    print(f"  Ground truth: {gt_path}")

    # Print summary
    by_outcome = {}
    by_intent = {}
    for gt in ground_truth:
        by_outcome[gt["outcome"]] = by_outcome.get(gt["outcome"], 0) + 1
        by_intent[gt["intent"]] = by_intent.get(gt["intent"], 0) + 1

    print(f"\n  Outcomes: {json.dumps(by_outcome, indent=4)}")
    print(f"  Intents:  {json.dumps(by_intent, indent=4)}")

    # Print sessions needing review
    print(f"\n  {len(ground_truth)} sessions need review in ground_truth.json")

    # Coverage report
    _print_coverage_report(ground_truth)

    await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(main())
