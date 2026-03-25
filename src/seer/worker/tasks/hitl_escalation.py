"""
HITL escalation task — checks for interrupted workflow runs past their
escalation thresholds and sends reminders via Slack or phone calls via Twilio.

Currently scoped to workflows named "PULIS" but designed to be extensible.
"""
from __future__ import annotations

import datetime as dt

from seer.database.workflow_models import WorkflowRun, WorkflowRunStatus
from seer.logger import get_logger
from seer.worker.broker_instance import broker

logger = get_logger(__name__)

# Escalation thresholds (seconds after interrupt started)
REMINDER_AFTER_SECONDS = 900   # 15 min — second Slack DM
CALL_AFTER_SECONDS = 1800      # 30 min — phone call

# Workflow name filter
PULIS_WORKFLOW_NAME = "PULIS"


@broker.task
async def check_hitl_escalations() -> dict:
    """
    Check for PULIS workflow runs in INTERRUPTED state that need escalation.

    Escalation levels:
    - 15min without response → send a reminder Slack DM
    - 30min without response → make a Twilio phone call

    Returns summary dict of actions taken.
    """
    now = dt.datetime.now(dt.timezone.utc)
    actions = {"reminders_sent": 0, "calls_made": 0, "errors": []}

    # Find PULIS runs that are INTERRUPTED with an expiry
    interrupted_runs = await WorkflowRun.filter(
        status=WorkflowRunStatus.INTERRUPTED,
        interrupt_expires_at__isnull=False,
        workflow__name=PULIS_WORKFLOW_NAME,
    ).select_related("workflow", "user").all()

    for run in interrupted_runs:
        if not run.interrupt_expires_at:
            continue

        # Calculate how long the interrupt has been active
        # interrupt_expires_at = created_at + timeout_seconds
        # We need to figure out when the interrupt started
        # timeout is stored in pending_interrupt_data
        timeout_seconds = CALL_AFTER_SECONDS  # default assumption
        if run.pending_interrupt_data and isinstance(run.pending_interrupt_data, dict):
            timeout_seconds = run.pending_interrupt_data.get("timeout_seconds", CALL_AFTER_SECONDS)

        interrupt_started_at = run.interrupt_expires_at - dt.timedelta(seconds=timeout_seconds)
        seconds_since_interrupt = (now - interrupt_started_at).total_seconds()

        # Track escalation state in node_traces to avoid duplicate escalations
        escalation_state = _get_escalation_state(run)

        try:
            if seconds_since_interrupt >= CALL_AFTER_SECONDS and not escalation_state.get("call_sent"):
                await _send_twilio_call(run)
                _set_escalation_state(run, "call_sent", True)
                await WorkflowRun.filter(id=run.id).update(node_traces=run.node_traces)
                actions["calls_made"] += 1
                logger.info("PULIS escalation: phone call for run %s", run.id)

            elif seconds_since_interrupt >= REMINDER_AFTER_SECONDS and not escalation_state.get("reminder_sent"):
                await _send_slack_reminder(run)
                _set_escalation_state(run, "reminder_sent", True)
                await WorkflowRun.filter(id=run.id).update(node_traces=run.node_traces)
                actions["reminders_sent"] += 1
                logger.info("PULIS escalation: Slack reminder for run %s", run.id)

        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: must process remaining runs even if one fails
            logger.exception("PULIS escalation error for run %s: %s", run.id, exc)
            actions["errors"].append({"run_id": run.id, "error": str(exc)})

    return actions


def _get_escalation_state(run: "WorkflowRun") -> dict:
    """Get escalation tracking state from run's node_traces."""
    if not run.node_traces or not isinstance(run.node_traces, dict):
        return {}
    return run.node_traces.get("_pulis_escalation", {})


def _set_escalation_state(run: "WorkflowRun", key: str, value: bool) -> None:
    """Set escalation tracking state in run's node_traces."""
    if not run.node_traces or not isinstance(run.node_traces, dict):
        run.node_traces = {}
    if "_pulis_escalation" not in run.node_traces:
        run.node_traces["_pulis_escalation"] = {}
    run.node_traces["_pulis_escalation"][key] = value


async def _send_slack_reminder(run: "WorkflowRun") -> None:
    """Send a reminder Slack DM for a missed check-in."""
    from seer.tools.slack.messages import SlackSendDirectMessageTool  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.tools.credential_resolver import CredentialResolver  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    tool = SlackSendDirectMessageTool()
    slack_config = _extract_slack_config(run)
    arguments = {
        "workspace_id": slack_config["workspace_id"],
        "user_id": slack_config["user_id"],
        "text": "⏰ You missed your PULIS check-in. How are you doing? Reply with craving level, mood, and what you're working on.",
    }

    resolver = CredentialResolver(tool=tool, user=run.user)
    credentials = await resolver.resolve(arguments=arguments)

    await tool.execute(
        access_token=None,
        arguments=arguments,
        credentials=credentials,
    )


async def _send_twilio_call(run: "WorkflowRun") -> None:
    """Make a phone call for an escalated missed check-in."""
    from seer.tools.twilio.call import TwilioMakeCallTool  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
    from seer.tools.credential_resolver import CredentialResolver  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    tool = TwilioMakeCallTool()
    twilio_config = _extract_twilio_config(run)
    arguments = {
        "to": twilio_config["to_number"],
        "message": "Hey, this is your PULIS check-in. You've been unreachable for 30 minutes. Please open Slack and log your check-in.",
    }

    resolver = CredentialResolver(tool=tool, user=run.user)
    credentials = await resolver.resolve(arguments=arguments)

    await tool.execute(
        access_token=None,
        arguments=arguments,
        credentials=credentials,
    )


def _extract_slack_config(run: "WorkflowRun") -> dict:
    """Extract Slack workspace_id and user_id from the workflow spec."""
    spec = run.spec or {}
    nodes = spec.get("nodes", [])
    for node in nodes:
        if node.get("type") == "tool" and "slack" in (node.get("tool") or ""):
            inputs = node.get("inputs", {})
            workspace_id = inputs.get("workspace_id")
            user_id = inputs.get("user_id")
            if workspace_id and user_id:
                return {"workspace_id": workspace_id, "user_id": user_id}
    raise ValueError("Could not find Slack config in PULIS workflow spec")


def _extract_twilio_config(run: "WorkflowRun") -> dict:
    """Extract Twilio phone number from the workflow spec variables."""
    spec = run.spec or {}
    variables = spec.get("variables", {})
    to_number = variables.get("escalation_phone_number")
    if not to_number:
        raise ValueError("Missing 'escalation_phone_number' in workflow variables")
    return {"to_number": to_number}


__all__ = ["check_hitl_escalations"]
