from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Mapping, Sequence, Set

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from seer.core.runtime.file_processor import FileContentProcessor

import seer.tools  # noqa: F401  # pylint: disable=unused-import  # Reason: import triggers tool registration via decorators
from seer.database import User
from seer.llm import get_llm
from seer.tools.base import get_tool
from seer.tools.executor import execute_tool
from seer.core.compiler.compile_pipeline import compile_parsed_workflow
from seer.core.compiler.context import CompilerContext
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.errors import ExecutionError, WorkflowCompilerError
from seer.core.registry.mcp_client_registry import MCPClientRegistry
from seer.core.registry.model_registry import ModelDefinition, ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.schema.jsonschema_adapter import SchemaError, check_schema
from seer.core.schema.models import (
    ImageGenNode,
    LLMNode,
    MCPNode,
    Node,
    ToolNode,
    WorkflowSpec,
)
from seer.core.schema.schema_registry import SchemaRegistry, ensure_json_schema
from seer.observability.llm import extract_usage_metadata
from seer.utilities.llm_messages import message_to_text

logger = logging.getLogger(__name__)

__all__ = ["WorkflowCompilerSingleton", "UserBoundCompiledWorkflow"]


@dataclass(frozen=True)
class UserBoundCompiledWorkflow:
    """
    Convenience wrapper that keeps track of the DB user associated with a compiled workflow.
    """

    workflow: CompiledWorkflow
    user: User

    async def ainvoke(
        self,
        workflow_input: Any = None,
        config: Mapping[str, Any] | None = None,
        context: WorkflowRuntimeContext | None = None,
        trigger: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        merged_config = dict(config or {})
        runtime_context = context or WorkflowRuntimeContext(user=self.user)
        user_before = merged_config.get("user")
        merged_config.pop("user", None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "UserBoundCompiledWorkflow.ainvoke user_in_config_before=%s context_user=%s config_keys=%s",
                bool(user_before),
                getattr(runtime_context.user, "id", None),
                sorted(merged_config.keys()),
            )
        return await self.workflow.ainvoke(
            workflow_input=workflow_input, config=merged_config, context=runtime_context, trigger=trigger
        )


class WorkflowCompilerSingleton:
    """
    Process-wide singleton responsible for compiling workflow specs using shared registries.
    """

    _instance: WorkflowCompilerSingleton | None = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._schema_registry = SchemaRegistry()
        self._tool_registry = ToolRegistry()
        self._model_registry = ModelRegistry()
        self._mcp_client_registry = MCPClientRegistry()
        self._registry_lock = Lock()

    @property
    def schema_registry(self) -> SchemaRegistry:
        return self._schema_registry

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry

    @property
    def mcp_client_registry(self) -> MCPClientRegistry:
        return self._mcp_client_registry

    @classmethod
    def instance(cls) -> WorkflowCompilerSingleton:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def compile(
        self,
        user: User,
        workflow_spec: Any,
        *,
        checkpointer: AsyncPostgresSaver | None = None,
    ) -> UserBoundCompiledWorkflow:
        """
        Compile the provided workflow spec and bind it to the given DB user.
        """

        spec = parse_workflow_spec(workflow_spec)
        self._ensure_dependencies(spec)
        compiled = await self._compile_spec(spec, checkpointer=checkpointer)
        return UserBoundCompiledWorkflow(workflow=compiled, user=user)

    def ensure_tool(self, tool_name: str) -> ToolDefinition:
        self._ensure_tool_registered(tool_name)
        return self._tool_registry.get(tool_name)

    def ensure_model(self, model_id: str) -> ModelDefinition:
        self._ensure_model_registered(model_id)
        return self._model_registry.get(model_id)

    def list_registered_tools(self) -> list[ToolDefinition]:
        return self._tool_registry.all()

    def list_registered_models(self) -> list[ModelDefinition]:
        return self._model_registry.all()

    # -------------------------------------------------------------------------
    # Dependency management
    # -------------------------------------------------------------------------
    def _ensure_dependencies(self, spec: WorkflowSpec) -> None:
        tool_names: Set[str] = set()
        model_ids: Set[str] = set()
        mcp_servers: Set[tuple[str, str]] = set()
        self._collect_dependencies(spec.nodes, tool_names, model_ids, mcp_servers)

        for tool_name in tool_names:
            self._ensure_tool_registered(tool_name)
        for model_id in model_ids:
            self._ensure_model_registered(model_id)
        if mcp_servers:
            logger.info("Workflow uses %d MCP servers", len(mcp_servers))

    def _collect_dependencies(
        self,
        nodes: Sequence[Node],
        tool_acc: Set[str],
        model_acc: Set[str],
        mcp_acc: Set[tuple[str, str]],
    ) -> None:
        for node in nodes:
            if isinstance(node, ToolNode):
                tool_acc.add(node.tool)
            elif isinstance(node, LLMNode):
                model = node.inputs.get("model")
                if model and isinstance(model, str):
                    model_acc.add(model)
            elif isinstance(node, MCPNode):
                mcp_acc.add((node.server, node.server_type))
            elif isinstance(node, ImageGenNode):
                model = node.inputs.get("model")
                if model and isinstance(model, str):
                    model_acc.add(model)
            # No recursion needed - all nodes are at top level in spec.nodes

    def _ensure_tool_registered(self, tool_name: str) -> None:
        if self._tool_registry.maybe_get(tool_name):
            return

        with self._registry_lock:
            if self._tool_registry.maybe_get(tool_name):
                return

            tool = get_tool(tool_name)
            if tool is None:
                raise WorkflowCompilerError(f"Tool '{tool_name}' is not registered in shared.tools")

            input_schema = ensure_json_schema(
                tool.get_parameters_schema(),
                schema_id=f"tools.{tool_name}.input",
            )
            output_schema = ensure_json_schema(
                tool.get_output_schema(),
                schema_id=f"tools.{tool_name}.output",
            )

            try:
                check_schema(input_schema)
                check_schema(output_schema)
            except SchemaError as exc:
                raise WorkflowCompilerError(
                    f"Tool '{tool_name}' registered invalid schema: {exc.message}"
                ) from exc

            handler, async_handler = self._build_tool_handler(
                tool_name,
                provider=getattr(tool, "provider", None),
                integration_type=getattr(tool, "integration_type", None),
            )
            definition = ToolDefinition(
                name=tool.name,
                version=getattr(tool, "version", "v1"),
                input_schema=input_schema,
                output_schema=output_schema,
                handler=handler,
                async_handler=async_handler,
            )
            self._tool_registry.register(definition)
            # Expose the output schema via SchemaRegistry so SchemaRef consumers can resolve ids.
            schema_id = f"tools.{tool.name}.output@v1"
            self._schema_registry.register(schema_id, output_schema)

    def _ensure_model_registered(self, model_id: str) -> None:
        if self._model_registry.maybe_get(model_id):
            return

        with self._registry_lock:
            if self._model_registry.maybe_get(model_id):
                return

            definition = ModelDefinition(
                model_id=model_id,
                text_handler=self._build_text_handler(model_id),
                json_handler=self._build_json_handler(model_id),
            )
            self._model_registry.register(definition)

    # -------------------------------------------------------------------------
    # Handler factories
    # -------------------------------------------------------------------------
    def _resolve_connection_id(
        self,
        config: Dict[str, Any] | None,
        tool_name: str,
        provider: str | None,
        integration_type: str | None,
    ) -> str | None:
        """Extract connection_id from config, checking multiple fallback keys."""
        tool_auth_context = (config or {}).get("tool_auth_context") or {}
        connection_id = (config or {}).get("connection_id")
        if not connection_id:
            connection_id = tool_auth_context.get(tool_name)
        if not connection_id and provider:
            connection_id = tool_auth_context.get(provider)
        if not connection_id and integration_type:
            connection_id = tool_auth_context.get(integration_type)
        return connection_id

    def _build_tool_handler(
        self,
        tool_name: str,
        *,
        provider: str | None = None,
        integration_type: str | None = None,
    ):
        def _resolve_user_and_connection(
            config: Dict[str, Any] | None,
            context: WorkflowRuntimeContext | None,
        ) -> tuple[Any, str | None]:
            user = None
            if context is not None:
                user = context.user
            elif config and "user" in config:
                user = config["user"]
            if user is None:
                logger.warning(
                    "Tool '%s' invoked without user context; config_keys=%s",
                    tool_name,
                    sorted(config.keys()) if config else [],
                )
                raise ExecutionError(
                    f"Tool '{tool_name}' requires workflow runtime context with 'user'"
                )
            connection_id = self._resolve_connection_id(config, tool_name, provider, integration_type)
            return user, connection_id

        async def async_handler(
            inputs: Dict[str, Any],
            config: Dict[str, Any] | None,
            context: WorkflowRuntimeContext | None = None,
        ) -> Any:
            user, connection_id = _resolve_user_and_connection(config, context)
            return await execute_tool(
                tool_name=tool_name,
                user=user,
                connection_id=connection_id,
                arguments=inputs or {},
                context=context,
            )

        return None, async_handler

    @staticmethod
    def _inject_structured_inputs(prompt: str, inputs_block: Dict[str, Any] | None) -> str:
        if not inputs_block:
            return prompt
        serialized = json.dumps(inputs_block, indent=2, sort_keys=True)
        return (
            f"{prompt}\n\n"
            "Structured inputs:\n"
            f"{serialized}"
        )

    @staticmethod
    def _process_file_contents_sync(file_contents: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process file contents synchronously for LLM input.

        Runs the async FileContentProcessor in a sync context. This is necessary
        because LLM handlers are synchronous but file processing is async.

        Args:
            file_contents: List of file info dicts from _resolve_llm_file_inputs

        Returns:
            Dict with 'image_blocks' and 'extracted_text'
        """
        processor = FileContentProcessor()

        # Run async processing synchronously
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in an async context - use nest_asyncio pattern or run in executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, processor.process_files(file_contents))
                return future.result()
        else:
            # Not in async context - safe to use asyncio.run
            return asyncio.run(processor.process_files(file_contents))

    def _build_text_handler(self, model_id: str):
        def handler(invocation: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
            parameters = invocation.get("parameters") or {}
            file_contents = invocation.get("file_contents", [])

            llm = get_llm(
                model=model_id,
                temperature=parameters.get("temperature") or 0.2,
            )
            prompt = self._inject_structured_inputs(
                invocation["prompt"], invocation.get("inputs")
            )

            # Process file inputs if present
            if file_contents:
                processed = WorkflowCompilerSingleton._process_file_contents_sync(file_contents)

                # Append extracted document text to prompt
                if processed["extracted_text"]:
                    prompt = f"{prompt}\n\n{processed['extracted_text']}"

                # Build multimodal message if images are present
                if processed["image_blocks"]:
                    content: list[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                    content.extend(processed["image_blocks"])
                    message = HumanMessage(content=content)
                    response = llm.invoke([message])
                else:
                    response = llm.invoke(prompt)
            else:
                response = llm.invoke(prompt)

            # Extract usage metadata
            usage_metadata = extract_usage_metadata(response, model_id)

            text_result = message_to_text(response)
            return text_result, usage_metadata

        return handler

    def _build_json_handler(self, model_id: str):
        def handler(invocation: Dict[str, Any], schema: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
            parameters = invocation.get("parameters") or {}
            file_contents = invocation.get("file_contents", [])

            llm = get_llm(
                model=model_id,
                temperature=parameters.get("temperature") or 0.2,
            )

            prompt = self._inject_structured_inputs(
                invocation["prompt"], invocation.get("inputs")
            )
            logger.info("Schema: %s", schema)

            # Ensure schema has required top-level keys for LangChain
            enriched_schema = {
                "title": "Output",
                "description": "Structured output schema",
                **schema
            }

            structured_llm = llm.with_structured_output(enriched_schema, method="json_schema")

            # Process file inputs if present
            if file_contents:
                processed = WorkflowCompilerSingleton._process_file_contents_sync(file_contents)

                # Append extracted document text to prompt
                if processed["extracted_text"]:
                    prompt = f"{prompt}\n\n{processed['extracted_text']}"

                # Build multimodal message if images are present
                if processed["image_blocks"]:
                    content: list[Dict[str, Any]] = [{"type": "text", "text": prompt}]
                    content.extend(processed["image_blocks"])
                    message = HumanMessage(content=content)
                    response = structured_llm.invoke([message])
                else:
                    response = structured_llm.invoke(prompt)
            else:
                response = structured_llm.invoke(prompt)

            # Extract usage metadata
            # Note: structured output might return dict directly, not AIMessage
            # Need to check if response has metadata or if it's in underlying call
            usage_metadata = {}
            if hasattr(structured_llm, "_last_response"):
                # pylint: disable=protected-access  # Reason: accessing LangChain internal state for usage metadata
                usage_metadata = extract_usage_metadata(structured_llm._last_response, model_id)
            elif hasattr(response, "usage_metadata") or hasattr(response, "response_metadata"):
                usage_metadata = extract_usage_metadata(response, model_id)
            else:
                # No metadata available, use empty
                usage_metadata = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "model": model_id,
                }

            return response, usage_metadata

        return handler

    # -------------------------------------------------------------------------
    # Compilation pipeline
    # -------------------------------------------------------------------------
    async def _compile_spec(
        self,
        spec: WorkflowSpec,
        *,
        checkpointer: AsyncPostgresSaver | None = None,
    ) -> CompiledWorkflow:
        context = CompilerContext(
            schema_registry=self._schema_registry,
            tool_registry=self._tool_registry,
            model_registry=self._model_registry,
            mcp_client_registry=self._mcp_client_registry,
        )

        return await compile_parsed_workflow(
            spec,
            context,
            checkpointer=checkpointer,
        )
