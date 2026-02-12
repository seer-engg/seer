"""
LLMNode - Execute language model inference.

Supports text and structured (JSON) output modes with credit checking
and usage tracking.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict

from seer.core.errors import ExecutionError
from seer.core.expr.typecheck import schema_from_output_contract
from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.core.nodes.registry import register_node_type
# Import model from schema/models.py (canonical location)
from seer.core.schema.models import LLMNode, OutputMode

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import NodeBase

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

async def _resolve_llm_file_inputs(
    auxiliary: Dict[str, Any],
    context: Any,  # WorkflowRuntimeContext
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """
    Scan auxiliary inputs for file references and resolve to content.

    File references (WorkflowFileRef) in the inputs are detected using the
    is_file_ref() function and resolved via the WorkflowFileSystem.
    """
    from seer.core.files.models import is_file_ref, parse_file_ref  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    if not context or not context.has_file_system:
        return auxiliary, []

    file_contents: list[Dict[str, Any]] = []
    resolved: Dict[str, Any] = {}

    for key, value in auxiliary.items():
        if is_file_ref(value):
            file_ref = parse_file_ref(value)
            content = await context.file_system.get_file_content(file_ref)
            file_contents.append({
                "key": key,
                "mime_type": file_ref.mime_type,
                "filename": file_ref.filename,
                "content": content,
            })
            resolved[key] = {
                "_resolved_file": file_ref.filename,
                "mime_type": file_ref.mime_type,
                "size_bytes": file_ref.size_bytes,
            }
        elif isinstance(value, list):
            # Handle list of file refs
            resolved_list = []
            for item in value:
                if is_file_ref(item):
                    file_ref = parse_file_ref(item)
                    content = await context.file_system.get_file_content(file_ref)
                    file_contents.append({
                        "key": key,
                        "mime_type": file_ref.mime_type,
                        "filename": file_ref.filename,
                        "content": content,
                    })
                    resolved_list.append({
                        "_resolved_file": file_ref.filename,
                        "mime_type": file_ref.mime_type,
                    })
                else:
                    resolved_list.append(item)
            resolved[key] = resolved_list
        else:
            resolved[key] = value

    return resolved, file_contents


# =============================================================================
# Node Type Implementation
# =============================================================================

class LLMNodeType(BaseNodeType):
    """Implementation of the LLM node type."""

    @property
    def type_literal(self) -> str:
        return "llm"

    @property
    def model_class(self) -> type["NodeBase"]:
        return LLMNode

    async def _check_credit_limit(self, context: Any) -> None:
        """Check credit limit before LLM call."""
        if not context or not context.user:
            return

        from seer.observability.credit_gate import check_credit_limit  # pylint: disable=import-outside-toplevel  # Reason: Late import for optional feature

        try:
            await check_credit_limit(context.user)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: Log and continue if credit check fails (except CreditLimitExceeded)
            if exc.__class__.__name__ == "CreditLimitExceeded":
                raise
            logger.error("Credit limit check failed: %s", exc)

    def _track_usage_async(self, usage_metadata: Dict[str, Any], context: Any) -> None:
        """Track LLM usage asynchronously (fire and forget)."""
        if not context or not context.user:
            logger.warning("Cannot track LLM usage: no user context")
            return

        # pylint: disable=import-outside-toplevel  # Reason: Late import for optional feature
        from seer.observability.cost_tracking import CostTracker
        from seer.observability.exceptions import RunCostCapExceeded

        async def do_track():
            try:
                await CostTracker.track_and_enforce_cap(
                    usage_metadata=usage_metadata,
                    context=context,
                    operation="workflow_execution",
                )
            except RunCostCapExceeded:
                raise
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Log error without crashing workflow
                logger.error("Failed to track LLM usage: %s", str(e), exc_info=True)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(do_track())
            else:
                loop.run_until_complete(do_track())
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Log scheduling error without crashing workflow
            logger.error("Failed to schedule LLM usage tracking: %s", e)

    # pylint: disable=too-many-locals,too-complex  # Reason: LLM execution requires many context variables and conditional logic
    async def execute_async(
        self,
        node: LLMNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """
        Execute LLM node with credit checking and usage tracking.
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.expr.evaluator import EvaluationContext, evaluate_value, render_template
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX
        from seer.core.runtime.validate_output import validate_against_schema

        # Check credit limit
        await self._check_credit_limit(ctx.runtime_context)

        # Build eval context
        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        eval_ctx = EvaluationContext(
            state=visible_state,
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
        )

        # Capture inputs for trace
        inputs = self._capture_inputs(node, eval_ctx)

        # Extract LLM configuration
        model = node.inputs.get("model")
        if not isinstance(model, str):
            raise ExecutionError(f"LLMNode {node.id}: 'model' must be a string in inputs")

        prompt_template = node.inputs.get("prompt")
        if not isinstance(prompt_template, str):
            raise ExecutionError(f"LLMNode {node.id}: 'prompt' must be a string in inputs")

        temperature = node.inputs.get("temperature")
        max_tokens = node.inputs.get("max_tokens")

        # Evaluate auxiliary inputs
        reserved_keys = {"model", "prompt", "temperature", "max_tokens"}
        auxiliary = {
            key: evaluate_value(eval_ctx, value)
            for key, value in node.inputs.items()
            if key not in reserved_keys
        }

        # Resolve file references
        resolved_auxiliary, file_contents = await _resolve_llm_file_inputs(auxiliary, ctx.runtime_context)

        # Render prompt
        prompt = render_template(eval_ctx, prompt_template)
        model_def = services.model_registry.get(model)

        invocation = {
            "prompt": prompt,
            "inputs": resolved_auxiliary,
            "file_contents": file_contents,
            "config": dict(ctx.config),
            "parameters": {"temperature": temperature, "max_tokens": max_tokens},
            "ui": node.ui,
        }

        # Execute LLM
        usage_metadata = {}
        try:
            type_schemas = services.type_env.as_dict()
            if node.outputs.mode == OutputMode.text:
                if model_def.text_handler is None:
                    raise ExecutionError(f"Model '{model}' does not support text responses")
                result, usage_metadata = model_def.text_handler(invocation)
                if not isinstance(result, str):
                    raise ExecutionError(f"LLM node '{node.id}' expected text response")
            elif node.outputs.mode == OutputMode.json:
                schema = type_schemas.get(node.id)
                if schema is None:
                    raise ExecutionError(f"No schema recorded for '{node.id}'")
                if model_def.json_handler is None:
                    raise ExecutionError(f"Model '{model}' does not support structured responses")
                result, usage_metadata = model_def.json_handler(invocation, schema)
                if not isinstance(result, dict):
                    raise ExecutionError(f"LLM node '{node.id}' expected JSON response")
            else:
                raise ExecutionError(f"Unsupported output mode '{node.outputs.mode}' for node '{node.id}'")
        except ExecutionError:
            raise
        except Exception as exc:
            trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
            error_trace = {
                trace_key: {
                    "node_id": node.id,
                    "node_type": "llm",
                    "inputs": inputs,
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                }
            }
            ctx.state.update(error_trace)  # type: ignore[arg-type]
            raise ExecutionError(f"LLM node '{node.id}' failed: {exc}", trace_data=error_trace) from exc

        # Track usage
        if usage_metadata:
            self._track_usage_async(usage_metadata, ctx.runtime_context)

        # Validate and prepare output
        schema = services.type_env.get(node.id)
        if schema is not None:
            validate_against_schema(schema, result, schema_id=node.id)

        output = {node.id: result}

        # Add trace (loop-aware key for nested loop support)
        trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
        output[trace_key] = {
            "node_id": node.id,
            "node_type": "llm",
            "inputs": inputs,
            "output": result,
            "output_key": node.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "succeeded",
            "usage": {
                "model": usage_metadata.get("model", model),
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "reasoning_tokens": usage_metadata.get("reasoning_tokens", 0),
                "total_tokens": (
                    usage_metadata.get("input_tokens", 0) +
                    usage_metadata.get("output_tokens", 0) +
                    usage_metadata.get("reasoning_tokens", 0)
                ),
            },
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("LLM node '%s' output keys: %s", node.id, list(output.keys()))

        return output

    def _capture_inputs(self, node: LLMNode, ctx: Any) -> Dict[str, Any]:
        """Capture LLM node inputs for trace."""
        from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

        inputs = {
            "prompt_template": node.inputs.get("prompt"),
            "model": node.inputs.get("model"),
        }

        reserved_keys = {"model", "prompt", "temperature", "max_tokens"}
        auxiliary = {k: v for k, v in node.inputs.items() if k not in reserved_keys}

        if auxiliary:
            input_refs = {}
            for key, expr in auxiliary.items():
                try:
                    input_refs[key] = evaluate_value(ctx, expr)
                except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Capture eval error in trace
                    input_refs[key] = {"__error__": str(e), "__expression__": expr}
            inputs["input_refs"] = input_refs

        if "temperature" in node.inputs:
            inputs["temperature"] = node.inputs["temperature"]
        if "max_tokens" in node.inputs:
            inputs["max_tokens"] = node.inputs["max_tokens"]

        return inputs

    def register_type_sync(
        self,
        node: LLMNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """Register LLM node's output schema with JSON root type validation."""
        from seer.core.errors import TypeEnvironmentError  # pylint: disable=import-outside-toplevel  # Reason: Rare error path

        schema = schema_from_output_contract(node.outputs, ctx.schema_registry)

        # OpenAI structured outputs require root type to be "object"
        if node.outputs.mode == OutputMode.json:
            root_type = schema.get("type")
            if root_type == "array":
                raise TypeEnvironmentError(
                    f"LLM node '{node.id}': JSON output schema must have root type 'object', "
                    f"not 'array'. OpenAI structured outputs do not support array root types. "
                    f"Wrap your array in an object property, e.g.: "
                    f'{{"type": "object", "properties": {{"items": <your-array-schema>}}}}'
                )

        if node.id:
            env.register(node.id, schema)


# Auto-register on module import
register_node_type(LLMNodeType())
