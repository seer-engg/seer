# pylint: disable=too-many-lines  # Reason: Single-responsibility file; splitting would harm cohesion
"""
AgentNode - Multi-step autonomous task execution with tool access.

Uses LangChain's create_agent internally to enable reasoning, tool calling,
and iteration until the task is complete. Supports state resolution,
OAuth credential binding for tools, structured output via response_format,
and comprehensive tracing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model
from seer.core.errors import ExecutionError
from seer.core.expr.typecheck import schema_from_output_contract
from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, TypeRegistrationContext, get_trace_key
from seer.core.nodes.registry import register_node_type
from seer.core.schema.models import AgentNode, OutputMode

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import NodeBase

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

# Mapping from JSON Schema types to Python types for Pydantic model creation
_JSON_TYPE_TO_PYTHON: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _json_schema_to_pydantic_type(prop_schema: Dict[str, Any]) -> type:
    """Convert JSON schema type to Python type for Pydantic model creation."""
    prop_type = prop_schema.get("type", "string")

    # Union types (list of types) - default to Any
    if isinstance(prop_type, list):
        return Any  # type: ignore[return-value]

    # Look up in type map, default to str
    return _JSON_TYPE_TO_PYTHON.get(prop_type, str)


def _create_tool_input_model(tool_name: str, input_schema: Dict[str, Any]) -> type[BaseModel]:
    """Create a Pydantic model from tool's parameter schema."""
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    field_definitions: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        python_type = _json_schema_to_pydantic_type(prop_schema)

        if prop_name in required:
            field_definitions[prop_name] = (python_type, ...)
        else:
            field_definitions[prop_name] = (Optional[python_type], None)

    model_name = f"{tool_name.replace('.', '_').replace('-', '_').title()}Input"
    return create_model(model_name, **field_definitions)


def _create_output_model_from_schema(node_id: str, schema: Dict[str, Any]) -> type[BaseModel]:
    """
    Create a Pydantic model from output JSON schema for structured output.

    Args:
        node_id: The node identifier (used for model naming)
        schema: JSON schema with 'properties' field

    Returns:
        Dynamically created Pydantic model class
    """
    properties = schema.get("properties", {})

    field_definitions: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        python_type = _json_schema_to_pydantic_type(prop_schema)
        description = prop_schema.get("description", "")
        field_definitions[prop_name] = (python_type, Field(description=description))

    # Create a valid Python class name from node_id
    model_name = f"{node_id.replace('-', '_').replace('.', '_').title()}Output"
    return create_model(model_name, **field_definitions)


def _parse_tool_spec(spec: Any) -> tuple[str, Optional[int]]:
    """
    Parse a tool specification into (tool_name, connection_id).

    Args:
        spec: Tool name (str) or {name, connection_id} object

    Returns:
        Tuple of (tool_name, connection_id)

    Raises:
        ExecutionError: If spec is not a valid format
    """
    if isinstance(spec, str):
        return spec, None
    if isinstance(spec, dict):
        return spec.get("name", ""), spec.get("connection_id")
    raise ExecutionError(f"Invalid tool spec type: {type(spec)}")


def _make_tool_executor(
    base_tool: Any,
    connection_id: Optional[int],
    ctx: NodeExecutionContext,
) -> Any:
    """Create a tool executor closure with credential resolution."""
    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
    from seer.tools.credential_resolver import CredentialResolver

    async def execute(**kwargs: Any) -> str:
        """Execute tool with resolved credentials."""
        try:
            access_token = None
            credentials = None

            if ctx.runtime_context and ctx.runtime_context.user:
                resolver = CredentialResolver(
                    user=ctx.runtime_context.user,
                    tool=base_tool,
                    connection_id=str(connection_id) if connection_id else None,
                )
                credentials = await resolver.resolve(kwargs)
                access_token = credentials.access_token

            result = await base_tool.execute(
                access_token=access_token,
                arguments=kwargs,
                credentials=credentials,
                context=ctx.runtime_context,
            )

            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, default=str)
            return str(result)

        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Agent needs error as string
            logger.exception("Tool %s execution failed", base_tool.name)
            return f"Error executing {base_tool.name}: {str(e)}"

    return execute


async def _bind_tools_for_agent(
    tool_specs: List[Any],
    services: "RuntimeServices",  # pylint: disable=unused-argument  # Reason: Reserved for future use with ToolRegistry
    ctx: NodeExecutionContext,
) -> List[StructuredTool]:
    """Convert tool specifications to LangChain StructuredTools with credentials."""
    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
    from seer.tools.base import get_tool

    bound_tools: List[StructuredTool] = []

    for spec in tool_specs:
        tool_name, connection_id = _parse_tool_spec(spec)

        base_tool = get_tool(tool_name)
        if base_tool is None:
            raise ExecutionError(f"Agent tool '{tool_name}' not found in tool registry")

        tool_executor = _make_tool_executor(base_tool, connection_id, ctx)

        # Create input model from schema
        try:
            input_schema = base_tool.get_parameters_schema()
            input_model = _create_tool_input_model(tool_name, input_schema)
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Fallback to no schema
            logger.warning("Failed to create input model for %s, using None", tool_name)
            input_model = None  # type: ignore[assignment]

        structured_tool = StructuredTool.from_function(
            coroutine=tool_executor,
            name=tool_name.replace(".", "_"),  # LangGraph requires valid identifiers
            description=base_tool.description,
            args_schema=input_model,
        )

        bound_tools.append(structured_tool)

    return bound_tools


def _parse_agent_result(messages: List[Any]) -> tuple[str, List[Dict[str, Any]]]:
    """
    Extract final output and reasoning steps from agent messages.

    Args:
        messages: List of messages from agent execution

    Returns:
        Tuple of (final_output, steps) where steps is a list of reasoning/tool steps
    """
    steps: List[Dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            step: Dict[str, Any] = {
                "type": "reasoning",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            }

            # Extract tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                step["tool_calls"] = [
                    {"tool": tc.get("name", tc.get("id", "unknown")), "input": tc.get("args", {})}
                    for tc in msg.tool_calls
                ]

            steps.append(step)

        elif isinstance(msg, ToolMessage):
            steps.append({
                "type": "tool_response",
                "tool": msg.name or "unknown",
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })

    # Extract final output from last AI message
    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_output = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return final_output, steps


def _extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON content from markdown code blocks if present.

    Args:
        text: Raw output that may contain markdown-wrapped JSON

    Returns:
        Extracted JSON string or original text if no code blocks found
    """
    # Try ```json blocks first
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # Try generic ``` blocks
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    return text


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


def _extract_agent_config(node: "AgentNode") -> Dict[str, Any]:
    """Extract and validate agent configuration from node inputs."""
    model_id = node.inputs.get("model")
    if not isinstance(model_id, str):
        raise ExecutionError(f"AgentNode {node.id}: 'model' must be a string in inputs")

    prompt_template = node.inputs.get("prompt")
    if not isinstance(prompt_template, str):
        raise ExecutionError(f"AgentNode {node.id}: 'prompt' must be a string in inputs")

    tool_specs = node.inputs.get("tools", [])
    if not isinstance(tool_specs, list):
        tool_specs = []

    max_iterations = node.inputs.get("max_iterations", 10)
    temperature = node.inputs.get("temperature", 0.2)

    return {
        "model_id": model_id,
        "prompt_template": prompt_template,
        "tool_specs": cast(List[Any], tool_specs),
        "max_iterations": max_iterations,
        "temperature": temperature,
    }


def _build_inputs_for_trace(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the inputs dict for trace data from agent config."""
    tool_specs = config["tool_specs"]
    return {
        "model": config["model_id"],
        "prompt_template": config["prompt_template"],
        "tools": [
            spec if isinstance(spec, str) else spec.get("name", "unknown") if isinstance(spec, dict) else str(spec)
            for spec in tool_specs
        ],
        "max_iterations": config["max_iterations"],
    }


def _build_agent_trace(
    node_id: str,
    inputs: Dict[str, Any],
    status: str,
    *,
    success_data: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build trace data for agent execution."""
    trace: Dict[str, Any] = {
        "node_id": node_id,
        "node_type": "agent",
        "inputs": inputs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }

    if status == "succeeded" and success_data:
        steps = success_data.get("steps", [])
        trace.update({
            "prompt": success_data.get("prompt", ""),
            "tools": success_data.get("tool_names", []),
            "steps": steps,
            "iterations": len([s for s in steps if s.get("type") == "reasoning"]),
            "output": success_data.get("result_value"),
            "output_key": node_id,
        })
    elif error:
        trace["error"] = error

    return trace


# =============================================================================
# Node Type Implementation
# =============================================================================


class AgentNodeType(BaseNodeType):
    """Implementation of the agent node type using LangChain's create_agent."""

    @property
    def type_literal(self) -> str:
        return "agent"

    @property
    def model_class(self) -> type["NodeBase"]:
        return AgentNode

    async def _check_credit_limit(self, context: Any) -> None:
        """Check credit limit before agent execution."""
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
            logger.warning("Cannot track agent usage: no user context")
            return

        # pylint: disable=import-outside-toplevel  # Reason: Late import for optional feature
        from seer.observability.cost_tracking import CostTracker
        from seer.observability.exceptions import RunCostCapExceeded

        async def do_track() -> None:
            try:
                await CostTracker.track_and_enforce_cap(
                    usage_metadata=usage_metadata,
                    context=context,
                    operation="agent_execution",
                )
            except RunCostCapExceeded:
                raise
            except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Log error without crashing workflow
                logger.error("Failed to track agent usage: %s", str(e), exc_info=True)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(do_track())
            else:
                loop.run_until_complete(do_track())
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Log scheduling error without crashing workflow
            logger.error("Failed to schedule agent usage tracking: %s", e)

    def _resolve_response_format(
        self,
        node: AgentNode,
        services: "RuntimeServices",
    ) -> Optional[ToolStrategy]:
        """Return ToolStrategy for structured output, or None for text mode."""
        if node.outputs.mode != OutputMode.json:
            return None
        schema = services.type_env.get(node.id)
        if schema is None:
            return None
        return ToolStrategy(_create_output_model_from_schema(node.id, schema))

    async def _handle_json_output(
        self,
        node: AgentNode,
        services: "RuntimeServices",
        result: Dict[str, Any],
        final_output: str,
    ) -> Any:
        """Parse and validate JSON output from agent when in json output mode."""
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.runtime.validate_output import validate_against_schema

        if node.outputs.mode != OutputMode.json:
            return final_output
        schema = services.type_env.get(node.id)
        if schema is None:
            return final_output
        structured_response = result.get("structured_response")
        if structured_response is not None:
            result_value = structured_response.model_dump() if isinstance(structured_response, BaseModel) else structured_response
        else:
            try:
                result_value = json.loads(_extract_json_from_markdown(final_output))
            except json.JSONDecodeError as e:
                raise ExecutionError(
                    f"Agent node '{node.id}' expected JSON output but got invalid JSON: {e}"
                ) from e
        validate_against_schema(schema, result_value, schema_id=node.id)
        return result_value

    # pylint: disable=too-many-locals  # Reason: Agent execution requires many context variables
    async def execute_async(
        self,
        node: AgentNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """Execute agent node with credit checking and usage tracking."""
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.expr.evaluator import EvaluationContext, render_template
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX

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

        # Extract and validate agent configuration
        config = _extract_agent_config(node)
        model_id = config["model_id"]
        tool_specs_list = config["tool_specs"]
        max_iterations = config["max_iterations"]
        temperature = config["temperature"]

        # Evaluate auxiliary inputs (non-reserved keys) and resolve file references
        from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
        reserved_keys = {"model", "prompt", "tools", "max_iterations", "temperature"}
        auxiliary = {
            key: evaluate_value(eval_ctx, value)
            for key, value in node.inputs.items()
            if key not in reserved_keys
        }
        _, file_contents = await _resolve_llm_file_inputs(auxiliary, ctx.runtime_context)

        # Build inputs for trace
        inputs = _build_inputs_for_trace(config)
        if file_contents:
            inputs["file_inputs"] = [{"key": f["key"], "filename": f["filename"]} for f in file_contents]

        # Render prompt with state values
        prompt = render_template(eval_ctx, config["prompt_template"])

        # Build HumanMessage, incorporating file contents if present
        human_message: Any
        if file_contents:
            from seer.core.runtime.file_processor import FileContentProcessor  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
            processor = FileContentProcessor()
            processed = await processor.process_files(file_contents)
            image_blocks = processed.get("image_blocks", [])
            extracted_text = processed.get("extracted_text", "")

            if extracted_text:
                prompt = f"{prompt}\n\n{extracted_text}"

            if image_blocks:
                content: List[Any] = [{"type": "text", "text": prompt}] + image_blocks
                human_message = HumanMessage(content=content)
            else:
                human_message = HumanMessage(content=prompt)
        else:
            human_message = HumanMessage(content=prompt)

        try:
            # Bind tools with credentials
            bound_tools = await _bind_tools_for_agent(tool_specs_list, services, ctx)

            # Get LLM chat model
            model_def = services.model_registry.get(model_id)
            llm = model_def.get_chat_model(temperature=temperature if isinstance(temperature, (int, float)) else 0.2)

            # Determine response_format for structured output
            response_format = self._resolve_response_format(node, services)

            # Create agent with optional structured output
            agent = create_agent(
                model=llm,
                tools=bound_tools,
                response_format=response_format,
            )

            # Execute agent with recursion limit based on max_iterations
            # Each iteration can have multiple steps (reasoning + tool call + tool response)
            # So we multiply by 3 to allow for the full loop
            recursion_limit = max_iterations * 3 if isinstance(max_iterations, int) else 30

            result = await agent.ainvoke(
                {"messages": [human_message]},
                config={"recursion_limit": recursion_limit},
            )

            # Parse agent result
            messages = result.get("messages", [])
            final_output, steps = _parse_agent_result(messages)

        except Exception as exc:
            trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
            trace_data = _build_agent_trace(
                node.id, inputs, "failed",
                error={"type": exc.__class__.__name__, "message": str(exc)},
            )
            error_trace = {trace_key: trace_data}
            ctx.state.update(error_trace)  # type: ignore[arg-type]
            raise ExecutionError(f"Agent node '{node.id}' failed: {exc}", trace_data=error_trace) from exc

        # Handle JSON output mode if specified
        result_value: Any = await self._handle_json_output(node, services, result, final_output)

        # Build output with trace
        trace_key = get_trace_key(node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {})
        output: Dict[str, Any] = {
            node.id: result_value,
            trace_key: _build_agent_trace(
                node.id, inputs, "succeeded",
                success_data={
                    "prompt": prompt,
                    "tool_names": [t.name for t in bound_tools],
                    "steps": steps,
                    "result_value": result_value,
                },
            ),
        }

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Agent node '%s' completed with %d steps",
                node.id,
                len(steps),
            )

        return output

    def register_type_sync(
        self,
        node: AgentNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """Register agent node's output schema in the type environment."""
        from seer.core.errors import TypeEnvironmentError  # pylint: disable=import-outside-toplevel  # Reason: Rare error path

        schema = schema_from_output_contract(node.outputs, ctx.schema_registry)

        # Validate that tools exist in registry
        tool_specs_raw = node.inputs.get("tools", [])
        tool_specs_list: List[Any] = tool_specs_raw if isinstance(tool_specs_raw, list) else []
        for spec in tool_specs_list:
            tool_name = spec if isinstance(spec, str) else spec.get("name", "") if isinstance(spec, dict) else ""
            if tool_name:
                # Check in BaseTool registry
                from seer.tools.base import get_tool  # pylint: disable=import-outside-toplevel
                if get_tool(tool_name) is None:
                    # Also check in the ToolRegistry (for workflow tools)
                    tool_def = ctx.tool_registry.maybe_get(tool_name)
                    if tool_def is None:
                        raise TypeEnvironmentError(
                            f"Agent node '{node.id}': tool '{tool_name}' not found in registry"
                        )

        if node.id:
            env.register(node.id, schema)


# Auto-register on module import
register_node_type(AgentNodeType())
