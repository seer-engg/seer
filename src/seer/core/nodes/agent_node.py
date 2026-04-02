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
from fastapi import HTTPException
from pydantic import BaseModel, Field, create_model

from seer.core.errors import (  # pylint: disable=no-name-in-module  # Reason: ProviderAuthError/ProviderError defined in Task 6
    ExecutionError,
    ProviderAuthError,
    ProviderError,
)
from seer.core.registry.default_models import DEPRECATED_MODEL_MAP
from seer.logger import get_logger
from seer.core.expr.typecheck import schema_from_output_contract
from seer.core.nodes.base import (
    BaseNodeType,
    NodeExecutionContext,
    TypeRegistrationContext,
    get_trace_key,
)
from seer.core.nodes.registry import register_node_type
from seer.core.schema.models import (
    AgentNode,
    InlineSchema,
    OutputMode,
    enforce_all_properties_required,
)
from seer.runtime_credit_limits import check_runtime_credit_limit

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import NodeBase

logger = get_logger(__name__)

_ERROR_PREFIX = "Error executing "


# =============================================================================
# ToolWorker — Supervisor-style tool execution isolation
# =============================================================================


class ToolWorker:
    """Isolates tool execution from the supervisor agent's context.

    Wraps a tool executor so that on failure, a cheap worker LLM reasons
    about the error and generates corrected params for a retry. The
    supervisor only ever sees a clean result or concise failure summary —
    never raw error strings that would poison its conversation history.
    """

    def __init__(
        self,
        tool_name: str,
        executor: Any,
        *,
        input_schema: Dict[str, Any],
        worker_llm: Any,
        max_retries: int = 1,
    ) -> None:
        self.tool_name = tool_name
        self.executor = executor
        self.input_schema = input_schema
        self.worker_llm = worker_llm
        self.max_retries = max_retries
        self.failure_log: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> str:
        logger.debug(
            "ToolWorker executing '%s' (max_retries=%d)",
            self.tool_name,
            self.max_retries,
        )
        result = await self.executor(**kwargs)

        if not isinstance(result, str) or not result.startswith(_ERROR_PREFIX):
            return result

        logger.debug(
            "ToolWorker '%s' first attempt failed, entering retry loop", self.tool_name
        )
        self.failure_log.append({"params": kwargs, "error": result})

        for _attempt in range(self.max_retries):
            corrected = await self._reason_and_correct(kwargs, result)
            if corrected is None:
                break

            result = await self.executor(**corrected)
            if not isinstance(result, str) or not result.startswith(_ERROR_PREFIX):
                return result

            self.failure_log.append({"params": corrected, "error": result})

        original_error = self.failure_log[0]["error"]
        reason = original_error.removeprefix(
            f"{_ERROR_PREFIX}{self.tool_name}: "
        ).strip()
        if len(reason) > 200:
            reason = reason[:200] + "..."
        return (
            f"Tool '{self.tool_name}' failed: {reason}. "
            f"Retried {len(self.failure_log)} time(s) without success. "
            f"Try a different approach."
        )

    async def _reason_and_correct(
        self, original_params: Dict[str, Any], error_msg: str
    ) -> Optional[Dict[str, Any]]:
        """Use cheap LLM to analyze failure and suggest corrected params."""
        prompt = (
            f"A tool call to '{self.tool_name}' failed.\n"
            f"Parameters: {json.dumps(original_params, default=str)}\n"
            f"Error: {error_msg}\n"
            f"Parameter schema: {json.dumps(self.input_schema, default=str)}\n\n"
            f"Can this be fixed by changing the parameters? "
            f"If yes, respond with ONLY the corrected JSON parameters object. "
            f"If no (e.g. auth error, service down), respond with exactly: UNFIXABLE"
        )
        try:
            response = await self.worker_llm.ainvoke([HumanMessage(content=prompt)])
            text = (
                response.content.strip()
                if isinstance(response.content, str)
                else str(response.content).strip()
            )
            if "UNFIXABLE" in text:
                return None
            return json.loads(text)
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Worker LLM failure must not crash the agent
            logger.warning(
                "ToolWorker LLM failed for %s, skipping retry", self.tool_name
            )
            return None


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


class _RecallMemoriesInput(BaseModel):
    query: str = Field(description="Semantic search query for relevant memories")
    limit: int = Field(
        default=5, ge=1, le=20, description="Maximum number of memories to return"
    )


class _RememberFactInput(BaseModel):
    content: str = Field(description="Memory content to store")
    infer: bool = Field(
        default=True, description="Whether to let the memory system infer related facts"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata to attach to the memory"
    )


class _UpdateMemoryInput(BaseModel):
    memory_id: str = Field(description="Memory ID to update")
    content: str = Field(description="Replacement content for the memory")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata override"
    )


class _DeleteMemoryInput(BaseModel):
    memory_id: str = Field(description="Memory ID to delete")


def _json_schema_to_pydantic_type(
    prop_schema: Dict[str, Any], model_name_prefix: str = "nested"
) -> type:
    """Convert JSON schema type to Python type for Pydantic model creation."""
    prop_type = prop_schema.get("type", "string")

    # Union types (list of types) - default to Any
    if isinstance(prop_type, list):
        return Any  # type: ignore[return-value]

    # Recurse into nested object schemas to preserve field structure
    if prop_type == "object" and prop_schema.get("properties"):
        return _create_output_model_from_schema(model_name_prefix, prop_schema)

    # Recurse into array item schemas so the LLM sees typed list elements
    if prop_type == "array":
        items_schema = prop_schema.get("items", {})
        item_type = _json_schema_to_pydantic_type(
            items_schema, f"{model_name_prefix}_item"
        )
        return List[item_type]  # type: ignore[valid-type]

    # Look up in type map, default to str
    return _JSON_TYPE_TO_PYTHON.get(prop_type, str)


def _create_tool_input_model(
    tool_name: str, input_schema: Dict[str, Any]
) -> type[BaseModel]:
    """Create a Pydantic model from tool's parameter schema."""
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    field_definitions: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        python_type = _json_schema_to_pydantic_type(prop_schema)

        if prop_name in required:
            field_definitions[prop_name] = (python_type, ...)
        else:
            default = prop_schema.get("default")
            if default is not None:
                field_definitions[prop_name] = (Optional[python_type], default)
            else:
                field_definitions[prop_name] = (Optional[python_type], None)

    model_name = f"{tool_name.replace('.', '_').replace('-', '_').title()}Input"
    return create_model(model_name, **field_definitions)


def _strip_null_optional_fields(data: Any, schema: Dict[str, Any]) -> Any:
    """
    Recursively remove None values for non-required fields so jsonschema validation
    does not reject null entries that the LLM emits for optional fields.

    JSON Schema allows absent optional properties; it does NOT allow null unless
    the schema explicitly declares 'type': ['string', 'null']. Stripping the key
    is semantically equivalent to 'not provided'.
    """
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return data

    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    result: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None and key not in required:
            continue  # Omit null optional fields — absent ≡ not provided
        prop_schema = properties.get(key, {})
        if isinstance(value, dict):
            value = _strip_null_optional_fields(value, prop_schema)
        elif isinstance(value, list):
            items_schema = prop_schema.get("items", {})
            value = [
                _strip_null_optional_fields(item, items_schema)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        result[key] = value
    return result


def _create_output_model_from_schema(
    node_id: str, schema: Dict[str, Any]
) -> type[BaseModel]:
    """
    Create a Pydantic model from output JSON schema for structured output.

    Args:
        node_id: The node identifier (used for model naming)
        schema: JSON schema with 'properties' field

    Returns:
        Dynamically created Pydantic model class
    """
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    field_definitions: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        model_name_prefix = f"{node_id}_{prop_name}"
        python_type = _json_schema_to_pydantic_type(prop_schema, model_name_prefix)
        description = prop_schema.get("description", "")

        if prop_name in required_fields:
            field_definitions[prop_name] = (python_type, Field(description=description))
        else:
            field_definitions[prop_name] = (
                Optional[python_type],
                Field(default=None, description=description),
            )

    # Create a valid Python class name from node_id
    model_name = f"{node_id.replace('-', '_').replace('.', '_').title()}Output"
    return create_model(model_name, **field_definitions)


def _parse_tool_spec(spec: Any) -> tuple[str, Optional[int], Optional[int]]:
    """
    Parse a tool specification into (tool_name, connection_id, integration_resource_id).

    Args:
        spec: Tool name (str) or {name, connection_id, integration_resource_id} object

    Returns:
        Tuple of (tool_name, connection_id, integration_resource_id)

    Raises:
        ExecutionError: If spec is not a valid format
    """
    if isinstance(spec, str):
        return spec, None, None
    if isinstance(spec, dict):
        return (
            spec.get("name", ""),
            spec.get("connection_id"),
            spec.get("integration_resource_id"),
        )
    raise ExecutionError(f"Invalid tool spec type: {type(spec)}")


def _truncate_tool_output(text: str, max_chars: int) -> str:
    """Truncate tool output to prevent context overflow in agent conversations."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + (
        f"\n\n[Output truncated from {len(text)} to {max_chars} characters. "
        "Ask for more specific queries to get smaller results.]"
    )


def _make_tool_executor(
    base_tool: Any,
    connection_id: Optional[int],
    ctx: NodeExecutionContext,
    integration_resource_id: Optional[int] = None,
) -> Any:
    """Create a tool executor closure with credential resolution."""
    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
    from seer.tools.credential_resolver import CredentialResolver

    async def execute(**kwargs: Any) -> str:
        """Execute tool with resolved credentials."""
        # Strip None values so tools fall back to their own defaults
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # Override integration_resource_id with the value from the agent tool spec.
        # The LLM may hallucinate a wrong resource ID, so the user-configured value always wins.
        if integration_resource_id is not None:
            kwargs["integration_resource_id"] = integration_resource_id
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

            try:
                result = await base_tool.execute(
                    access_token=access_token,
                    arguments=kwargs,
                    credentials=credentials,
                    context=ctx.runtime_context,
                )
            except HTTPException as exc:
                if (
                    exc.status_code == 401
                    and credentials
                    and credentials.connection
                    and credentials.connection.refresh_token_enc
                ):
                    logger.info(
                        "Retrying tool %s after 401 with force-refreshed token",
                        base_tool.name,
                    )
                    credentials = await resolver.resolve(kwargs, force_refresh=True)
                    access_token = credentials.access_token
                    result = await base_tool.execute(
                        access_token=access_token,
                        arguments=kwargs,
                        credentials=credentials,
                        context=ctx.runtime_context,
                    )
                else:
                    raise

            from seer.config import config as seer_config  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time

            max_chars = seer_config.agent_tool_output_max_chars
            if isinstance(result, (dict, list)):
                return _truncate_tool_output(
                    json.dumps(result, indent=2, default=str), max_chars
                )
            return _truncate_tool_output(str(result), max_chars)

        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Agent needs error as string
            logger.exception("Tool %s execution failed", base_tool.name)
            return f"Error executing {base_tool.name}: {str(e)}"

    return execute


def _wrap_executor_with_worker(
    tool_name: str,
    tool_executor: Any,
    input_schema: Dict[str, Any],
    worker_llm: Any,
    max_retries: int,
) -> tuple[Any, "ToolWorker"]:
    """Wrap a tool executor in a ToolWorker for isolated failure handling."""
    worker = ToolWorker(
        tool_name=tool_name,
        executor=tool_executor,
        input_schema=input_schema,
        worker_llm=worker_llm,
        max_retries=max_retries,
    )
    _worker = worker

    async def _shielded_executor(_w: "ToolWorker" = _worker, **kwargs: Any) -> str:
        return await _w(**kwargs)

    return _shielded_executor, worker


async def _bind_tools_for_agent(
    tool_specs: List[Any],
    services: "RuntimeServices",  # pylint: disable=unused-argument  # Reason: Reserved for future use with ToolRegistry
    ctx: NodeExecutionContext,
    *,
    worker_llm: Any = None,
    max_retries: int = 1,
) -> tuple[List[StructuredTool], Dict[str, ToolWorker]]:
    """Convert tool specifications to LangChain StructuredTools with credentials.

    When worker_llm is provided, each tool executor is wrapped in a ToolWorker
    that isolates failures from the supervisor agent's conversation history.

    Returns (bound_tools, worker_map) where worker_map maps tool name → ToolWorker.
    """
    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
    from seer.tools.base import get_tool

    bound_tools: List[StructuredTool] = []
    worker_map: Dict[str, ToolWorker] = {}

    # Collect per-tool credential bindings from workflow spec so meta-tools
    # (e.g. fetch_to_scratch) can apply the same overrides as direct tool calls.
    tool_bindings: Dict[str, Dict[str, Any]] = {}

    for spec in tool_specs:
        tool_name, connection_id, resource_id = _parse_tool_spec(spec)
        if connection_id is not None or resource_id is not None:
            tool_bindings[tool_name] = {
                "connection_id": connection_id,
                "integration_resource_id": resource_id,
            }

        base_tool = get_tool(tool_name)
        if base_tool is None:
            raise ExecutionError(f"Agent tool '{tool_name}' not found in tool registry")

        tool_executor = _make_tool_executor(base_tool, connection_id, ctx, resource_id)

        # Create input model from schema
        input_schema: Dict[str, Any] = {}
        try:
            input_schema = base_tool.get_parameters_schema()
            input_model = _create_tool_input_model(tool_name, input_schema)
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Fallback to no schema
            logger.warning("Failed to create input model for %s, using None", tool_name)
            input_model = None  # type: ignore[assignment]

        worker: Optional[ToolWorker] = None
        if worker_llm is not None:
            # StructuredTool.from_function needs an actual async function, not a callable class.
            # Wrap in a closure so LangChain can inspect the signature via args_schema.
            tool_executor, worker = _wrap_executor_with_worker(
                tool_name, tool_executor, input_schema, worker_llm, max_retries
            )

        tool_id = tool_name.replace(".", "_")
        structured_tool = StructuredTool.from_function(
            coroutine=tool_executor,
            name=tool_id,
            description=base_tool.description,
            args_schema=input_model,
        )

        if worker is not None:
            worker_map[tool_id] = worker

        bound_tools.append(structured_tool)

    # Expose bindings to the runtime context so meta-tools can resolve credentials
    if ctx.runtime_context and tool_bindings:
        ctx.runtime_context.tool_bindings = tool_bindings

    return bound_tools, worker_map


def _format_memory_tool_response(memories: List[Dict[str, Any]]) -> str:
    """Format memory search results for autonomous agent use."""
    if not memories:
        return "No relevant memories found."

    lines = [f"Found {len(memories)} relevant memories:"]
    for index, memory in enumerate(memories, start=1):
        text = memory.get("memory", memory.get("text", str(memory)))
        score = memory.get("score")
        if isinstance(score, (int, float)):
            lines.append(f"{index}. {text} (relevance {score:.2f})")
        else:
            lines.append(f"{index}. {text}")
    return "\n".join(lines)


def _has_explicit_memory_write(steps: List[Dict[str, Any]]) -> bool:
    """Detect whether the agent explicitly invoked a memory write tool."""
    for step in steps:
        for tool_call in step.get("tool_calls") or []:
            if tool_call.get("tool") == "remember_fact":
                return True
    return False


def _stringify_memory_output(final_output: Any, result_value: Any) -> str:
    """Choose the most useful assistant output representation for memory extraction."""
    if isinstance(final_output, str) and final_output.strip():
        return final_output.strip()
    if isinstance(result_value, str) and result_value.strip():
        return result_value.strip()
    if result_value is None:
        return ""
    if isinstance(result_value, (dict, list)):
        return json.dumps(result_value, default=str)
    return str(result_value).strip()


async def _persist_attached_memory_conversation(
    ctx: NodeExecutionContext,
    *,
    node_id: str,
    memory_bank_id: Optional[str],
    user_prompt: str,
    assistant_output: str,
    steps: List[Dict[str, Any]],
) -> None:
    """Persist the latest agent exchange into the attached bank for future recall."""
    if (
        memory_bank_id is None
        or not user_prompt.strip()
        or not assistant_output.strip()
    ):
        return

    runtime_context = ctx.runtime_context
    if runtime_context is None or runtime_context.memory_access is None:
        return

    if _has_explicit_memory_write(steps):
        return

    transcript = f"User: {user_prompt.strip()}\n\nAssistant: {assistant_output.strip()}"
    metadata: Dict[str, Any] = {
        "source": "workflow_agent_run",
        "node_id": node_id,
    }
    if runtime_context.workflow_run_id:
        metadata["workflow_run_id"] = runtime_context.workflow_run_id
    if runtime_context.thread_id:
        metadata["thread_id"] = runtime_context.thread_id

    try:
        stored = await runtime_context.memory_access.add(
            memory_bank_id,
            transcript,
            metadata=metadata,
            infer=True,
        )
        if stored is None:
            logger.warning(
                "Automatic memory persistence returned no result for agent node '%s'",
                node_id,
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: memory persistence must not fail the workflow run
        logger.warning(
            "Automatic memory persistence failed for agent node '%s': %s", node_id, exc
        )


async def _build_attached_memory_tools(
    ctx: NodeExecutionContext, memory_bank_id: str
) -> List[StructuredTool]:
    """Create bank-bound convenience tools for an attached agent node."""
    runtime_context = ctx.runtime_context
    if runtime_context is None or runtime_context.memory_access is None:
        raise ExecutionError(
            "Agent node requires runtime memory access when memory_bank_id is configured"
        )

    memory_access = runtime_context.memory_access

    async def recall_memories(query: str, limit: int = 5) -> str:
        memories = await memory_access.search(memory_bank_id, query, limit=limit)
        return _format_memory_tool_response(memories)

    async def remember_fact(
        content: str, infer: bool = True, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        created = await memory_access.add(
            memory_bank_id, content, metadata=metadata, infer=infer
        )
        if not created:
            return "No memory was stored."
        return json.dumps(created, indent=2, default=str)

    async def update_memory(
        memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        updated = await memory_access.update(
            memory_bank_id, memory_id, content, metadata=metadata
        )
        if not updated:
            return f"Memory {memory_id} was not found."
        return json.dumps(updated, indent=2, default=str)

    async def delete_memory(memory_id: str) -> str:
        deleted = await memory_access.delete(memory_bank_id, memory_id)
        return (
            f"Deleted memory {memory_id}."
            if deleted
            else f"Memory {memory_id} was not found."
        )

    return [
        StructuredTool.from_function(
            coroutine=recall_memories,
            name="recall_memories",
            description="Search the attached memory bank for relevant prior facts and context.",
            args_schema=_RecallMemoriesInput,
        ),
        StructuredTool.from_function(
            coroutine=remember_fact,
            name="remember_fact",
            description="Store a new memory in the attached memory bank.",
            args_schema=_RememberFactInput,
        ),
        StructuredTool.from_function(
            coroutine=update_memory,
            name="update_memory",
            description="Update an existing memory in the attached memory bank.",
            args_schema=_UpdateMemoryInput,
        ),
        StructuredTool.from_function(
            coroutine=delete_memory,
            name="delete_memory",
            description="Delete an existing memory from the attached memory bank.",
            args_schema=_DeleteMemoryInput,
        ),
    ]


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
                "content": msg.content
                if isinstance(msg.content, str)
                else str(msg.content),
            }

            # Extract tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                step["tool_calls"] = [
                    {
                        "tool": tc.get("name", tc.get("id", "unknown")),
                        "input": tc.get("args", {}),
                    }
                    for tc in msg.tool_calls
                ]

            steps.append(step)

        elif isinstance(msg, ToolMessage):
            steps.append(
                {
                    "type": "tool_response",
                    "tool": msg.name or "unknown",
                    "content": msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content),
                }
            )

    # Extract final output from last AI message
    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_output = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            break

    return final_output, steps


def _collect_artifacts_from_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract WorkflowFileRef dicts from create_artifact ToolMessage responses.

    When the agent calls create_artifact, the tool returns a JSON-serialized
    WorkflowFileRef. This helper scans all ToolMessages and collects those refs
    into a list that gets emitted as the `{node.id}__artifacts` output key.

    Args:
        messages: List of messages from agent execution.

    Returns:
        List of WorkflowFileRef dicts (may be empty).
    """
    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
    from seer.core.files.models import is_file_ref
    from seer.core.nodes.artifacts.tool import ARTIFACT_TOOL_NAME

    artifacts: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == ARTIFACT_TOOL_NAME:
            try:
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                data = json.loads(content)
                if is_file_ref(data):
                    artifacts.append(data)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to parse artifact from create_artifact tool response"
                )
    return artifacts


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
            file_contents.append(
                {
                    "key": key,
                    "mime_type": file_ref.mime_type,
                    "filename": file_ref.filename,
                    "content": content,
                }
            )
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
                    file_contents.append(
                        {
                            "key": key,
                            "mime_type": file_ref.mime_type,
                            "filename": file_ref.filename,
                            "content": content,
                        }
                    )
                    resolved_list.append(
                        {
                            "_resolved_file": file_ref.filename,
                            "mime_type": file_ref.mime_type,
                        }
                    )
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
    model_id = DEPRECATED_MODEL_MAP.get(model_id, model_id)

    prompt_template = node.inputs.get("prompt")
    if not isinstance(prompt_template, str):
        raise ExecutionError(
            f"AgentNode {node.id}: 'prompt' must be a string in inputs"
        )

    tool_specs = node.inputs.get("tools", [])
    if not isinstance(tool_specs, list):
        tool_specs = []

    max_iterations = node.inputs.get("max_iterations", 10)
    temperature = node.inputs.get("temperature", 0.2)
    enable_artifacts = bool(node.inputs.get("enable_artifacts", False))
    memory_bank_id = node.inputs.get("memory_bank_id")

    return {
        "model_id": model_id,
        "prompt_template": prompt_template,
        "tool_specs": cast(List[Any], tool_specs),
        "max_iterations": max_iterations,
        "temperature": temperature,
        "enable_artifacts": enable_artifacts,
        "memory_bank_id": memory_bank_id if isinstance(memory_bank_id, str) else None,
    }


def _build_inputs_for_trace(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the inputs dict for trace data from agent config."""
    tool_specs = config["tool_specs"]
    return {
        "model": config["model_id"],
        "prompt_template": config["prompt_template"],
        "tools": [
            spec
            if isinstance(spec, str)
            else spec.get("name", "unknown")
            if isinstance(spec, dict)
            else str(spec)
            for spec in tool_specs
        ],
        "max_iterations": config["max_iterations"],
        "memory_bank_id": config.get("memory_bank_id"),
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
        trace.update(
            {
                "prompt": success_data.get("prompt", ""),
                "tools": success_data.get("tool_names", []),
                "steps": steps,
                "iterations": len([s for s in steps if s.get("type") == "reasoning"]),
                "output": success_data.get("result_value"),
                "output_key": node_id,
                "artifacts": success_data.get("artifacts", []),
                "tool_failures": success_data.get("tool_failures", {}),
            }
        )
    elif error:
        trace["error"] = error

    return trace


async def _build_human_message(
    prompt: str, file_contents: List[Any]
) -> tuple[str, Any]:
    """Build HumanMessage from prompt and optional file contents.

    Returns (possibly extended prompt, HumanMessage) where prompt may have
    extracted text appended when file contents include text-based files.
    """
    if not file_contents:
        return prompt, HumanMessage(content=prompt)

    from seer.core.runtime.file_processor import FileContentProcessor  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time

    processor = FileContentProcessor()
    processed = await processor.process_files(file_contents)
    image_blocks = processed.get("image_blocks", [])
    extracted_text = processed.get("extracted_text", "")

    if extracted_text:
        prompt = f"{prompt}\n\n{extracted_text}"

    if image_blocks:
        content: List[Any] = [{"type": "text", "text": prompt}] + image_blocks
        return prompt, HumanMessage(content=content)
    return prompt, HumanMessage(content=prompt)


def _build_agent_eval_context(ctx: NodeExecutionContext) -> Any:
    """Build the expression-evaluation context visible to the agent node."""
    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
    from seer.core.expr.evaluator import EvaluationContext
    from seer.core.runtime.state import INTERNAL_STATE_PREFIX

    visible_state = {
        key: value
        for key, value in ctx.state.items()
        if not key.startswith(INTERNAL_STATE_PREFIX)
    }
    return EvaluationContext(
        state=visible_state,
        locals=ctx.locals_ctx or {},
        config=ctx.config,
        trigger=ctx.trigger,
        vars=ctx.vars,
    )


async def _resolve_agent_file_contents(
    node: AgentNode, eval_ctx: Any, runtime_context: Any
) -> list[Dict[str, Any]]:
    """Evaluate auxiliary inputs and resolve any attached file references."""
    from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    reserved_keys = {
        "model",
        "prompt",
        "tools",
        "max_iterations",
        "temperature",
        "enable_artifacts",
        "memory_bank_id",
    }
    auxiliary = {
        key: evaluate_value(eval_ctx, value)
        for key, value in node.inputs.items()
        if key not in reserved_keys
    }
    _, file_contents = await _resolve_llm_file_inputs(auxiliary, runtime_context)
    return file_contents


async def _build_prompt_bundle(
    node: AgentNode,
    ctx: NodeExecutionContext,
    config: Dict[str, Any],
) -> tuple[Dict[str, Any], str, str, Any]:
    """Prepare trace inputs, rendered prompt, final prompt, and HumanMessage."""
    from seer.core.expr.evaluator import render_template  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time

    eval_ctx = _build_agent_eval_context(ctx)
    file_contents = await _resolve_agent_file_contents(
        node, eval_ctx, ctx.runtime_context
    )
    inputs = _build_inputs_for_trace(config)
    if file_contents:
        inputs["file_inputs"] = [
            {"key": item["key"], "filename": item["filename"]} for item in file_contents
        ]

    rendered_prompt = render_template(eval_ctx, config["prompt_template"])
    prompt = rendered_prompt
    memory_bank_id = config["memory_bank_id"]

    if memory_bank_id is not None:
        if ctx.runtime_context is None or ctx.runtime_context.memory_access is None:
            raise ExecutionError(
                f"Agent node '{node.id}' requires runtime memory access for memory_bank_id '{memory_bank_id}'"
            )
        memory_context = await ctx.runtime_context.memory_access.get_prompt_context(
            memory_bank_id,
            current_query=prompt,
        )
        if memory_context:
            prompt = f"{memory_context}\n\n{prompt}"

    prompt, human_message = await _build_human_message(prompt, file_contents)
    return inputs, rendered_prompt, prompt, human_message


def _build_recursion_limit(max_iterations: Any) -> int:
    """Compute the LangGraph recursion limit from max_iterations.

    Each agent iteration can produce up to 5 LangGraph steps (AIMessage,
    multiple ToolMessages, potential retries). Use 5x multiplier with
    a floor of 50 to avoid premature GraphRecursionError.
    """
    if isinstance(max_iterations, int):
        return max(max_iterations * 5, 50)
    return 50


def _collect_tool_worker_failures(
    worker_map: Dict[str, ToolWorker],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract failure logs from ToolWorker instances for trace data."""
    failures: Dict[str, List[Dict[str, Any]]] = {}
    for tool_name, worker in worker_map.items():
        if worker.failure_log:
            failures[tool_name] = worker.failure_log
    return failures


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
        await check_runtime_credit_limit(context, logger)

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

    async def _build_bound_tools(
        self,
        node: AgentNode,
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
        config: Dict[str, Any],
        *,
        worker_llm: Any = None,
        max_retries: int = 1,
    ) -> tuple[List[StructuredTool], Dict[str, ToolWorker]]:
        """Bind registry tools, attached memory tools, and artifact helpers."""
        bound_tools, worker_map = await _bind_tools_for_agent(
            config["tool_specs"],
            services,
            ctx,
            worker_llm=worker_llm,
            max_retries=max_retries,
        )

        if config["memory_bank_id"] is not None:
            bound_tools.extend(
                await _build_attached_memory_tools(ctx, config["memory_bank_id"])
            )

        if config["enable_artifacts"]:
            # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
            from seer.core.nodes.artifacts.tool import make_create_artifact_tool

            bound_tools.append(make_create_artifact_tool(ctx, node.id))

        return bound_tools, worker_map

    async def _invoke_agent(  # pylint: disable=too-many-locals  # Reason: Agent invocation requires tracking tools, middleware, callbacks, and BYOK creds
        self,
        *,
        node: AgentNode,
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
        config: Dict[str, Any],
        bound_tools: List[StructuredTool],
        human_message: Any,
    ) -> Dict[str, Any]:
        """Create and invoke the LangChain agent with usage tracking callbacks."""
        # Resolve BYOK credentials from runtime context
        byok_key = None
        byok_url = None
        if ctx.runtime_context:
            byok_key = ctx.runtime_context.byok_api_key
            byok_url = ctx.runtime_context.byok_base_url

        model_def = services.model_registry.get(config["model_id"])
        llm = model_def.get_chat_model(
            temperature=config["temperature"]
            if isinstance(config["temperature"], (int, float))
            else 0.2,
            byok_api_key=byok_key,
            byok_base_url=byok_url,
        )
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from langchain.agents.middleware import SummarizationMiddleware

        middleware = [
            SummarizationMiddleware(
                model=llm,
                trigger=("tokens", 100000),
            ),
        ]
        from seer.prompts import get_datetime_context  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import

        agent = create_agent(
            model=llm,
            tools=bound_tools,
            system_prompt=get_datetime_context(),
            response_format=self._resolve_response_format(node, services),
            middleware=middleware,
        )

        recursion_limit = _build_recursion_limit(config["max_iterations"])

        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.agents.nexus.cost_callback import (
            CostCapCallbackHandler,
            clear_chat_runtime_context,
            set_chat_runtime_context,
        )

        callbacks = []
        if ctx.runtime_context:
            set_chat_runtime_context(ctx.runtime_context)
            callbacks.append(CostCapCallbackHandler())

        try:
            try:
                result = await agent.ainvoke(
                    {"messages": [human_message]},
                    config={
                        "recursion_limit": recursion_limit,
                        "callbacks": callbacks,
                    },
                )
            except Exception as exc:
                exc_type_name = type(exc).__name__
                if "AuthenticationError" in exc_type_name or "401" in str(exc):
                    raise ProviderAuthError(f"Provider authentication failed — check your API key: {exc}") from exc
                if "APIError" in exc_type_name or "APIConnectionError" in exc_type_name:
                    raise ProviderError(f"Provider error (not a Seer issue): {exc}") from exc
                raise
            return result
        finally:
            if ctx.runtime_context:
                clear_chat_runtime_context()

    async def _parse_execution_result(
        self,
        *,
        node: AgentNode,
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
        config: Dict[str, Any],
        rendered_prompt: str,
        result: Dict[str, Any],
    ) -> tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse the agent result and persist any attached memory transcript."""
        messages = result.get("messages", [])
        final_output, steps = _parse_agent_result(messages)
        artifacts = (
            _collect_artifacts_from_messages(messages)
            if config["enable_artifacts"]
            else []
        )
        result_value: Any = await self._handle_json_output(
            node, services, result, final_output
        )
        await _persist_attached_memory_conversation(
            ctx,
            node_id=node.id,
            memory_bank_id=config["memory_bank_id"],
            user_prompt=rendered_prompt,
            assistant_output=_stringify_memory_output(final_output, result_value),
            steps=steps,
        )
        return result_value, steps, artifacts

    def _build_error_trace(
        self,
        node: AgentNode,
        ctx: NodeExecutionContext,
        inputs: Dict[str, Any],
        exc: Exception,
    ) -> Dict[str, Any]:
        """Build and store trace data for a failed execution."""
        trace_key = get_trace_key(
            node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {}
        )
        trace_data = _build_agent_trace(
            node.id,
            inputs,
            "failed",
            error={"type": exc.__class__.__name__, "message": str(exc)},
        )
        error_trace = {trace_key: trace_data}
        ctx.state.update(error_trace)  # type: ignore[arg-type]
        return error_trace

    def _build_success_output(
        self,
        *,
        node: AgentNode,
        ctx: NodeExecutionContext,
        inputs: Dict[str, Any],
        success_data: Dict[str, Any],
        enable_artifacts: bool,
    ) -> Dict[str, Any]:
        """Build the final workflow output payload, including traces."""
        trace_key = get_trace_key(
            node.id, ctx.state, ctx.loop_body_map or {}, ctx.nested_loop_parents or {}
        )
        artifacts = success_data["artifacts"]
        output: Dict[str, Any] = {
            node.id: success_data["result_value"],
            trace_key: _build_agent_trace(
                node.id,
                inputs,
                "succeeded",
                success_data=success_data,
            ),
        }
        if enable_artifacts:
            output[f"{node.id}__artifacts"] = artifacts
        return output

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
            result_value = (
                structured_response.model_dump()
                if isinstance(structured_response, BaseModel)
                else structured_response
            )
        else:
            try:
                result_value = json.loads(_extract_json_from_markdown(final_output))
            except json.JSONDecodeError as e:
                raise ExecutionError(
                    f"Agent node '{node.id}' expected JSON output but got invalid JSON: {e}"
                ) from e
        result_value = _strip_null_optional_fields(result_value, schema)
        validate_against_schema(schema, result_value, schema_id=node.id)
        return result_value

    @staticmethod
    def _init_worker_llm(node_id: str) -> tuple[Any, int]:
        """Initialize worker LLM for supervisor mode. Returns (llm, max_retries)."""
        from seer.config import config as seer_config  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time

        max_retries = seer_config.agent_tool_max_retries
        if not seer_config.agent_supervisor_mode:
            return None, max_retries
        try:
            from seer.llm import get_llm  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time

            worker_model = seer_config.default_llm_model
            worker_llm = get_llm(model=worker_model, temperature=0)
            logger.info(
                "Supervisor mode: worker LLM '%s' initialized for agent node '%s'",
                worker_model,
                node_id,
            )
            return worker_llm, max_retries
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Supervisor mode is optional; fall back to flat mode
            logger.warning("Failed to init worker LLM, falling back to flat agent mode")
            return None, max_retries

    async def execute_async(  # pylint: disable=too-many-locals  # Reason: Orchestration method, splitting would obscure control flow
        self,
        node: AgentNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """Execute agent node with credit checking and usage tracking."""
        await self._check_credit_limit(ctx.runtime_context)
        config = _extract_agent_config(node)
        inputs, rendered_prompt, prompt, human_message = await _build_prompt_bundle(
            node, ctx, config
        )
        worker_llm, max_retries = self._init_worker_llm(node.id)

        try:
            bound_tools, worker_map = await self._build_bound_tools(
                node,
                ctx,
                services,
                config,
                worker_llm=worker_llm,
                max_retries=max_retries,
            )
            result = await self._invoke_agent(
                node=node,
                ctx=ctx,
                services=services,
                config=config,
                bound_tools=bound_tools,
                human_message=human_message,
            )
            result_value, steps, artifacts = await self._parse_execution_result(
                node=node,
                ctx=ctx,
                services=services,
                config=config,
                rendered_prompt=rendered_prompt,
                result=result,
            )
        except Exception as exc:
            error_trace = self._build_error_trace(node, ctx, inputs, exc)
            raise ExecutionError(
                f"Agent node '{node.id}' failed: {exc}", trace_data=error_trace
            ) from exc

        tool_failures = _collect_tool_worker_failures(worker_map)
        output = self._build_success_output(
            node=node,
            ctx=ctx,
            inputs=inputs,
            success_data={
                "prompt": prompt,
                "tool_names": [tool.name for tool in bound_tools],
                "steps": steps,
                "result_value": result_value,
                "artifacts": artifacts,
                "tool_failures": tool_failures,
            },
            enable_artifacts=config["enable_artifacts"],
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Agent node '%s' completed with %d steps, %d artifacts, %d tool failures",
                node.id,
                len(steps),
                len(artifacts),
                sum(len(v) for v in tool_failures.values()),
            )

        return output

    def get_analytics_properties(
        self, node: AgentNode, ctx: "NodeExecutionContext"
    ) -> dict:  # type: ignore[override]
        return {"agent_type": node.inputs.get("model", "unknown")}

    def register_type_sync(
        self,
        node: AgentNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """Register agent node's output schema in the type environment."""
        from seer.core.errors import TypeEnvironmentError  # pylint: disable=import-outside-toplevel  # Reason: Rare error path

        schema = schema_from_output_contract(node.outputs, ctx.schema_registry)

        # Enforce strict schema for LLM structured output so the model cannot silently omit fields
        if node.outputs.mode == OutputMode.json and isinstance(
            node.outputs.schema, InlineSchema
        ):
            enforce_all_properties_required(node.outputs.schema.json_schema)

        # Validate that tools exist in registry
        tool_specs_raw = node.inputs.get("tools", [])
        tool_specs_list: List[Any] = (
            tool_specs_raw if isinstance(tool_specs_raw, list) else []
        )
        for spec in tool_specs_list:
            tool_name = (
                spec
                if isinstance(spec, str)
                else spec.get("name", "")
                if isinstance(spec, dict)
                else ""
            )
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
