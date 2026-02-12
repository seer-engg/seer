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

    async def execute_async(  # pylint: disable=too-many-locals  # Reason: Browser automation requires many context variables
        self,
        node: BrowserNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """
        Execute browser automation node using BrowserUse Agent.

        Supports:
        - Structured output via expect_outputs with extraction_schema
        - Screenshot saving to S3 when save_screenshots=True
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.services.browser import BrowserService
        from seer.core.expr.evaluator import EvaluationContext, evaluate_value
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX
        from seer.core.runtime.validate_output import validate_against_schema

        # Build eval context
        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        eval_ctx = EvaluationContext(
            state=visible_state,
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
        )

        # Capture inputs
        inputs = self._evaluate_inputs(node, eval_ctx)

        # Evaluate task expression
        task = evaluate_value(eval_ctx, node.task) if "${" in node.task else node.task

        # Get extraction schema if expect_outputs is specified with JSON mode
        type_schemas = services.type_env.as_dict()
        extraction_schema = None
        if node.expect_outputs and node.expect_outputs.mode == OutputMode.json:
            full_schema = type_schemas.get(node.id)
            if full_schema:
                extraction_schema = full_schema.get("properties", {}).get("extracted_data")

        # Get file system context for screenshot saving
        file_system = None
        workflow_run_id = None
        user_id = None
        if node.save_screenshots and ctx.runtime_context:
            if ctx.runtime_context.has_file_system:
                file_system = ctx.runtime_context.file_system
            workflow_run_id = ctx.runtime_context.workflow_run_id
            user_id = str(ctx.runtime_context.user.id) if ctx.runtime_context.user else None

        # Execute browser task
        service = BrowserService.instance()

        try:
            result = await service.execute_task(
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
                user_id=user_id,
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

        # Validate structured output
        if node.expect_outputs and node.expect_outputs.mode == OutputMode.json:
            full_schema = type_schemas.get(node.id)
            if full_schema:
                extracted_schema = full_schema.get("properties", {}).get("extracted_data")
                if extracted_schema:
                    validate_against_schema(extracted_schema, result.get("extracted_data", {}), schema_id=f"{node.id}.extracted_data")

        # Prepare output
        output = {node.id: result}

        # Store trace data (loop-aware key for nested loop support)
        trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
        output[trace_key] = {
            "node_id": node.id,
            "node_type": "browser",
            "task": task,
            "inputs": inputs,
            "output": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "succeeded" if result.get("success") else "failed",
        }

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
        schema = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "result": {"type": "string"},
                "extracted_data": extracted_data_schema,
                "final_url": {"type": ["string", "null"]},
                "screenshots": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": {},
        }

        if node.id:
            env.register(node.id, schema)


# Auto-register on module import
register_node_type(BrowserNodeType())
