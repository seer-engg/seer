"""
EABotNode — meetingsEA node that joins a meeting, records, and waits for it to end.

Uses LangGraph's interrupt() mechanism to pause the workflow while the bot is
in the meeting. The webhook handler auto-resumes the workflow when the meeting ends,
passing transcript and recording data back through the interrupt.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

from seer.config import config
from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.core.nodes.registry import register_node_type
from seer.core.schema.models import EABotNode
from seer.logger import get_logger

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import NodeBase

logger = get_logger("core.nodes.ea_bot")

# Maps user-friendly recording_mode to Attendee recording_settings
# Attendee uses: format (mp4/mp3/none), view (speaker_view/gallery_view/speaker_view_no_sidebar)
_RECORDING_MODE_MAP = {
    "speaker_view": {"format": "mp4", "view": "speaker_view"},
    "gallery_view": {"format": "mp4", "view": "gallery_view"},
    "audio_only": {"format": "mp3"},
}

_REQUEST_TIMEOUT_SECONDS = 30

_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "bot_id": {"type": "string", "description": "Attendee bot identifier"},
        "state": {"type": "string", "description": "Final bot state (e.g., ended)"},
        "transcript": {
            "type": "array",
            "description": "Meeting transcript entries",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "start_ms": {"type": "integer"},
                    "duration_ms": {"type": "integer"},
                },
            },
        },
        "recording_url": {
            "type": ["string", "null"],
            "description": "Short-lived URL to download the recording",
        },
    },
}


async def _attendee_request(
    method: str,
    path: str,
    *,
    json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make an authenticated request to the Attendee API."""
    api_key = config.meetings_ea_api_key
    base_url = config.meetings_ea_base_url
    if not api_key or not base_url:
        raise RuntimeError("meetingsEA is not configured (MEETINGS_EA_API_KEY / MEETINGS_EA_BASE_URL)")

    url = f"{base_url.rstrip('/')}{path}"  # pylint: disable=no-member  # Reason: Pydantic Field resolves to str at runtime
    headers = {"Authorization": f"Token {api_key}"}

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.request(method, url, headers=headers, json=json)
        if response.status_code >= 400:
            # Log the full error body for debugging
            try:
                error_body = response.json()
            except (ValueError, KeyError):
                error_body = response.text[:500]
            logger.error(
                "Attendee API error: %s %s -> %d: %s",
                method, path, response.status_code, error_body,
            )
            response.raise_for_status()
        return response.json()


class EABotNodeType(BaseNodeType):
    """meetingsEA node — join meeting, record, wait for end, return transcript."""

    @property
    def type_literal(self) -> str:
        return "ea_bot"

    @property
    def model_class(self) -> type["NodeBase"]:
        return EABotNode

    @staticmethod
    def _build_bot_body(
        meeting_url: str,
        bot_name: Any,
        recording_mode: Any,
        *,
        node_id: str,
        run_id: Optional[str],
        org_id: Optional[str],
    ) -> Dict[str, Any]:
        """Build the Attendee API request body for bot creation."""
        body: Dict[str, Any] = {
            "meeting_url": meeting_url,
            "bot_name": bot_name if isinstance(bot_name, str) else "meetingsEA",
            "metadata": {
                "seer_org_id": org_id or "",
                "seer_run_id": run_id or "",
                "seer_node_id": node_id,
            },
        }

        # Recording settings — Attendee expects {format, view}
        mode_key = recording_mode if isinstance(recording_mode, str) else "speaker_view"
        if mode_key in _RECORDING_MODE_MAP:
            body["recording_settings"] = _RECORDING_MODE_MAP[mode_key]

        # Auto-leave when all humans leave (60s after becoming only participant)
        body["automatic_leave_settings"] = {
            "only_participant_in_meeting_timeout_seconds": 60,
        }

        # Register webhook for state change notifications (must be HTTPS)
        webhook_base = config.webhook_base_url
        # pylint: disable=no-member  # Reason: Pydantic Field resolves to str at runtime
        if webhook_base and webhook_base.startswith("https://"):
            body["webhooks"] = [{
                "url": f"{webhook_base.rstrip('/')}/api/v1/webhooks/meetings_ea",
                "triggers": ["bot.state_change"],
            }]

        return body

    async def execute_async(
        self,
        node: EABotNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """
        Execute the meetingsEA node:
        1. Create a bot via Attendee API
        2. Interrupt (pause) workflow until meeting ends
        3. Return transcript + recording from resume payload
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from langgraph.types import interrupt
        from seer.core.expr.evaluator import EvaluationContext, evaluate_value
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX

        # Build eval context for resolving ${...} expressions in inputs
        eval_ctx = EvaluationContext(
            state={k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)},
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
            vars=ctx.vars,
        )

        # Evaluate inputs
        inputs = node.inputs or {}
        meeting_url = evaluate_value(eval_ctx, inputs.get("meeting_url", ""))
        bot_name = evaluate_value(eval_ctx, inputs.get("bot_name", "meetingsEA"))
        recording_mode = evaluate_value(eval_ctx, inputs.get("recording_mode", "speaker_view"))

        if not meeting_url or not isinstance(meeting_url, str):
            raise ValueError("meetingsEA node requires a valid meeting_url input")

        run_id = ctx.runtime_context.workflow_run_id if ctx.runtime_context else None
        org_id = str(ctx.runtime_context.organization_id) if ctx.runtime_context and ctx.runtime_context.organization_id else None

        body = self._build_bot_body(meeting_url, bot_name, recording_mode, node_id=node.id, run_id=run_id, org_id=org_id)

        # Create bot
        logger.info(
            "meetingsEA node creating bot",
            extra={"node_id": node.id, "meeting_url": meeting_url, "run_id": run_id},
        )
        bot_id = (await _attendee_request("POST", "/api/v1/bots", json=body)).get("id", "")

        logger.info(
            "meetingsEA bot created, interrupting workflow",
            extra={"node_id": node.id, "bot_id": bot_id, "run_id": run_id},
        )

        # Pause workflow — resumes when webhook handler calls resume_workflow_run
        resume_data = interrupt({
            "type": "ea_bot",
            "node_id": node.id,
            "bot_id": bot_id,
            "run_id": run_id,
            "meeting_url": meeting_url,
            "timeout_seconds": node.timeout_seconds,
        })

        # When resumed, resume_data contains transcript + recording from webhook handler
        resume_data = resume_data or {}

        output_data = {
            "bot_id": resume_data.get("bot_id", bot_id),
            "state": resume_data.get("state", "ended"),
            "transcript": resume_data.get("transcript", []),
            "recording_url": resume_data.get("recording_url"),
        }

        output: Dict[str, Any] = {node.id: output_data}

        # Store trace data
        output[get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})] = {
            "node_id": node.id,
            "node_type": "ea_bot",
            "title": f"meetingsEA: {meeting_url}",
            "bot_id": bot_id,
            "output": output_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return output

    def register_type_sync(
        self,
        node: EABotNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """Register the meetingsEA node's output schema."""
        if node.id:
            env.register(node.id, _OUTPUT_SCHEMA)


# Auto-register on module import
register_node_type(EABotNodeType())
