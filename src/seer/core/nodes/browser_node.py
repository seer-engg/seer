"""
BrowserNode - Browser automation using natural language task descriptions.

Executes browser automation tasks via BrowserUse Agent. Users manage
browser profiles separately (with saved login sessions), and workflows
reference profiles by ID for authenticated automation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict

from seer.core.errors import ExecutionError
from seer.core.expr.typecheck import schema_from_output_contract
from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.core.nodes.registry import register_node_type
# Import model from schema/models.py (canonical location)
from seer.core.schema.models import BrowserNode, OutputMode

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import NodeBase

logger = logging.getLogger(__name__)


# =============================================================================
# Node Type Implementation
# =============================================================================

class BrowserNodeType(BaseNodeType):
    """Implementation of the browser automation node type."""

    @property
    def type_literal(self) -> str:
        return "browser"

    @property
    def model_class(self) -> type["NodeBase"]:
        return BrowserNode

    async def _check_credit_limit(self, context: Any) -> None:
        """Check credit limit before browser execution."""
        if not context or not context.user:
            return

        from seer.observability.credit_gate import check_credit_limit  # pylint: disable=import-outside-toplevel  # Reason: Late import for optional feature

        try:
            await check_credit_limit(context.user)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: Log and continue if credit check fails (except CreditLimitExceeded)
            if exc.__class__.__name__ == "CreditLimitExceeded":
                raise
            logger.error("Credit limit check failed: %s", exc)

    async def _track_usage_async(self, usage_metadata: Dict[str, Any], context: Any, node_id: str) -> None:
        """Track browser LLM usage via centralized CostTracker."""
        if not context or not context.user:
            logger.warning("Cannot track browser LLM usage: no user context")
            return

        from seer.observability.cost_tracking import CostTracker  # pylint: disable=import-outside-toplevel  # Reason: Late import for optional feature

        try:
            await CostTracker.track_and_enforce_cap(
                usage_metadata=usage_metadata,
                context=context,
                operation="browser_execution",
                extra_metadata={
                    "node_id": node_id,
                    "steps_taken": usage_metadata.get("steps_taken", 0),
                    "aggregated": True,
                },
            )
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Log error without crashing workflow
            if e.__class__.__name__ == "RunCostCapExceeded":
                raise
            logger.error("Failed to track browser LLM usage: %s", str(e), exc_info=True)

    @staticmethod
    def _get_extraction_schema(node: BrowserNode, type_schemas: Any) -> Any:
        """Get extraction schema if expect_outputs is specified with JSON mode."""
        if not (node.expect_outputs and node.expect_outputs.mode == OutputMode.json):
            return None
        full_schema = type_schemas.get(node.id)
        if not full_schema:
            return None
        return full_schema.get("properties", {}).get("extracted_data")

    @staticmethod
    def _get_screenshot_context(node: BrowserNode, runtime_context: Any) -> tuple:
        """Get file system and workflow_run_id for screenshots and recording.

        Note: workflow_run_id is returned regardless of save_screenshots setting
        because session recordings also need it for associating with workflow runs.
        """
        if not runtime_context:
            return None, None
        file_system = None
        if node.save_screenshots and runtime_context.has_file_system:
            file_system = runtime_context.file_system
        workflow_run_id = runtime_context.workflow_run_id  # Always return for recordings
        return file_system, workflow_run_id

    @staticmethod
    def _validate_extracted_data(node: BrowserNode, result: Dict[str, Any], type_schemas: Any) -> None:
        """Validate structured output against expect_outputs schema if applicable."""
        if not (node.expect_outputs and node.expect_outputs.mode == OutputMode.json):
            return
        from seer.core.runtime.validate_output import validate_against_schema  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
        full_schema = type_schemas.get(node.id)
        if not full_schema:
            return
        extracted_schema = full_schema.get("properties", {}).get("extracted_data")
        if extracted_schema:
            validate_against_schema(extracted_schema, result.get("extracted_data", {}), schema_id=f"{node.id}.extracted_data")

    async def execute_async(  # pylint: disable=too-many-locals  # Reason: Browser automation requires many context variables
        self,
        node: BrowserNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """
        Execute browser automation node with credit checking and usage tracking.

        Supports:
        - Pre-execution credit limit check
        - Structured output via expect_outputs with extraction_schema
        - Screenshot saving to S3 when save_screenshots=True
        - LLM usage tracking via CostTracker
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.services.browser import BrowserService
        from seer.core.expr.evaluator import EvaluationContext, evaluate_value
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX

        # Check credit limit before execution
        await self._check_credit_limit(ctx.runtime_context)

        # Build eval context
        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        eval_ctx = EvaluationContext(
            state=visible_state,
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
        )

        # Capture inputs and evaluate task expression
        inputs = self._evaluate_inputs(node, eval_ctx)
        task = evaluate_value(eval_ctx, node.task) if "${" in node.task else node.task

        # Resolve pre-execution config via helpers
        type_schemas = services.type_env.as_dict()
        extraction_schema = self._get_extraction_schema(node, type_schemas)
        file_system, workflow_run_id = self._get_screenshot_context(node, ctx.runtime_context)

        # Execute browser task
        try:
            logger.info("model=%s, task=%s, inputs=%s, profile_id=%s, max_steps=%s, timeout=%s",
                node.model or "default",
                task,
                inputs,
                node.browser_profile_id,
                node.max_steps,
                node.timeout_seconds,
            )
            result = await BrowserService.instance().execute_task(
                user=ctx.runtime_context.user if ctx.runtime_context else None,
                task=task,
                inputs=inputs,
                browser_profile_id=node.browser_profile_id,
                max_steps=node.max_steps,
                timeout_seconds=node.timeout_seconds,
                extraction_schema=extraction_schema,
                save_screenshots=node.save_screenshots,
                file_system=file_system,
                workflow_run_id=workflow_run_id,
                model=node.model,
            )
        except Exception as exc:
            trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
            error_trace = {
                trace_key: {
                    "node_id": node.id,
                    "node_type": "browser",
                    "inputs": inputs,
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                }
            }
            ctx.state.update(error_trace)  # type: ignore[arg-type]
            raise ExecutionError(f"Browser task failed: {exc}", trace_data=error_trace) from exc

        # Track LLM usage if available
        usage_metadata = result.pop("usage", None)
        if usage_metadata and ctx.runtime_context:
            await self._track_usage_async(usage_metadata, ctx.runtime_context, node.id)

        # Validate structured output only on success (skip validation on timeout/error)
        if result.get("success", False):
            self._validate_extracted_data(node, result, type_schemas)

        # Build usage trace
        usage_trace = {}
        if usage_metadata:
            usage_trace = {
                "model": usage_metadata.get("model", "moonshotai/kimi-k2.5"),
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "reasoning_tokens": usage_metadata.get("reasoning_tokens", 0),
                "total_tokens": usage_metadata.get("total_tokens", 0),
            }

        # Assemble output with trace data
        output: Dict[str, Any] = {node.id: result}
        trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
        trace_data: Dict[str, Any] = {
            "node_id": node.id,
            "node_type": "browser",
            "task": task,
            "inputs": inputs,
            "output": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "succeeded" if result.get("success") else "failed",
        }
        if usage_trace:
            trace_data["usage"] = usage_trace
        output[trace_key] = trace_data
        return output

    def _evaluate_inputs(self, node: BrowserNode, ctx: Any) -> Dict[str, Any]:
        """Evaluate input expressions."""
        from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

        inputs = {}
        for key, expr in node.inputs.items():
            try:
                inputs[key] = evaluate_value(ctx, expr)
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Capture eval error in trace
                inputs[key] = {"__error__": str(e), "__expression__": expr}
        return inputs

    def register_type_sync(
        self,
        node: BrowserNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """
        Register browser node's output schema.

        Browser nodes ALWAYS output {success, result, extracted_data, final_url, screenshots}.
        If expect_outputs is specified, its schema applies to extracted_data, not the root.
        """
        # Determine the extracted_data schema
        if node.expect_outputs:
            extracted_data_schema = schema_from_output_contract(node.expect_outputs, ctx.schema_registry)
        else:
            # Default: permissive object
            extracted_data_schema = {"type": "object", "additionalProperties": {}}

        # Browser always produces the same envelope structure
        # additionalProperties=False ensures strict type checking catches invalid
        # references like ${node.shops} when it should be ${node.extracted_data.shops}
        schema = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "result": {"type": "object"},  # format_browser_history returns object with steps/completed/success
                "extracted_data": extracted_data_schema,
                "final_url": {"type": ["string", "null"]},
                "urls": {"type": "array", "items": {"type": "string"}},
                "duration_seconds": {"type": ["number", "null"]},
                "steps_count": {"type": ["integer", "null"]},
                "extracted_content": {"type": "array", "items": {"type": "string"}},
                "model_thoughts": {"type": "array", "items": {"type": "object"}},
                "model_actions": {"type": "array", "items": {"type": "object"}},
                "screenshots": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        }

        if node.id:
            env.register(node.id, schema)


# Auto-register on module import
register_node_type(BrowserNodeType())
