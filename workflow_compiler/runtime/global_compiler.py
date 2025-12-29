from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Mapping, Sequence, Set

import shared.tools  # noqa: F401  # ensure default tool registration occurs
from shared.database.models import User
from shared.llm import get_llm
from shared.tools.base import get_tool
from shared.tools.executor import execute_tool
from workflow_compiler.compiler.context import CompilerContext
from workflow_compiler.compiler.lower_control_flow import build_execution_plan
from workflow_compiler.compiler.parse import parse_workflow_spec
from workflow_compiler.compiler.type_env import build_type_environment
from workflow_compiler.compiler.validate_refs import validate_references
from workflow_compiler.errors import ExecutionError, WorkflowCompilerError
from workflow_compiler.registry.model_registry import ModelDefinition, ModelRegistry
from workflow_compiler.registry.tool_registry import ToolDefinition, ToolRegistry
from workflow_compiler.runtime.context import WorkflowRuntimeContext
from workflow_compiler.runtime.execution import CompiledWorkflow
from workflow_compiler.runtime.nodes import NodeRuntime, RuntimeServices
from workflow_compiler.schema.models import (
    ForEachNode,
    IfNode,
    LLMNode,
    Node,
    ToolNode,
    WorkflowSpec,
)
from workflow_compiler.schema.schema_registry import SchemaRegistry, ensure_json_schema

logger = logging.getLogger(__name__)

__all__ = ["WorkflowCompilerSingleton", "UserBoundCompiledWorkflow"]


def _message_to_text(message: Any) -> str:
    """
    LangChain responses can return strings, AIMessage objects, or richer payloads.
    This helper normalizes them into a plain string for downstream processing.
    """

    if message is None:
        return ""

    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


@dataclass(frozen=True)
class UserBoundCompiledWorkflow:
    """
    Convenience wrapper that keeps track of the DB user associated with a compiled workflow.
    """

    workflow: CompiledWorkflow
    user: User

    def invoke(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        context: WorkflowRuntimeContext | None = None,
    ) -> Mapping[str, Any]:
        merged_config = dict(config or {})
        runtime_context = context or WorkflowRuntimeContext(user=self.user)
        user_before = merged_config.get("user")
        merged_config.pop("user", None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "UserBoundCompiledWorkflow.invoke user_in_config_before=%s context_user=%s config_keys=%s inputs_keys=%s",
                bool(user_before),
                getattr(runtime_context.user, "id", None),
                sorted(merged_config.keys()),
                sorted(inputs.keys()),
            )
        return self.workflow.invoke(inputs, config=merged_config, context=runtime_context)

    async def ainvoke(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        context: WorkflowRuntimeContext | None = None,
    ) -> Mapping[str, Any]:
        merged_config = dict(config or {})
        runtime_context = context or WorkflowRuntimeContext(user=self.user)
        user_before = merged_config.get("user")
        merged_config.pop("user", None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "UserBoundCompiledWorkflow.ainvoke user_in_config_before=%s context_user=%s config_keys=%s inputs_keys=%s",
                bool(user_before),
                getattr(runtime_context.user, "id", None),
                sorted(merged_config.keys()),
                sorted(inputs.keys()),
            )
        return await self.workflow.ainvoke(inputs, config=merged_config, context=runtime_context)


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

    @classmethod
    def instance(cls) -> WorkflowCompilerSingleton:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def compile(self, user: User, workflow_spec: Any) -> UserBoundCompiledWorkflow:
        """
        Compile the provided workflow spec and bind it to the given DB user.
        """

        spec = parse_workflow_spec(workflow_spec)
        self._ensure_dependencies(spec)
        compiled = self._compile_spec(spec)
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
        self._collect_dependencies(spec.nodes, tool_names, model_ids)

        for tool_name in tool_names:
            self._ensure_tool_registered(tool_name)
        for model_id in model_ids:
            self._ensure_model_registered(model_id)

    def _collect_dependencies(
        self,
        nodes: Sequence[Node],
        tool_acc: Set[str],
        model_acc: Set[str],
    ) -> None:
        for node in nodes:
            if isinstance(node, ToolNode):
                tool_acc.add(node.tool)
            elif isinstance(node, LLMNode):
                model_acc.add(node.model)

            if isinstance(node, IfNode):
                self._collect_dependencies(node.then, tool_acc, model_acc)
                self._collect_dependencies(node.else_, tool_acc, model_acc)
            elif isinstance(node, ForEachNode):
                self._collect_dependencies(node.body, tool_acc, model_acc)

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
    def _build_tool_handler(
        self,
        tool_name: str,
        *,
        provider: str | None = None,
        integration_type: str | None = None,
    ):
        def _resolve_user_and_connection(
            inputs: Dict[str, Any],
            config: Dict[str, Any] | None,
            context: WorkflowRuntimeContext | None,
        ) -> tuple[Any, Dict[str, Any], str | None]:
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
            config_copy = dict(config or {})
            tool_auth_context = config_copy.get("tool_auth_context") or {}
            connection_id = config_copy.get("connection_id")
            if not connection_id:
                connection_id = tool_auth_context.get(tool_name)
            if not connection_id and provider:
                connection_id = tool_auth_context.get(provider)
            if not connection_id and integration_type:
                connection_id = tool_auth_context.get(integration_type)
            return user, config_copy, connection_id

        async def async_handler(
            inputs: Dict[str, Any],
            config: Dict[str, Any] | None,
            context: WorkflowRuntimeContext | None = None,
        ) -> Any:
            user, config_copy, connection_id = _resolve_user_and_connection(
                inputs, config, context
            )
            return await execute_tool(
                tool_name=tool_name,
                user=user,
                connection_id=connection_id,
                arguments=inputs or {},
            )

        def sync_handler(
            inputs: Dict[str, Any],
            config: Dict[str, Any] | None,
            context: WorkflowRuntimeContext | None = None,
        ) -> Any:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(async_handler(inputs, config, context))
            raise ExecutionError(
                "Synchronous workflow execution cannot run tools from an active event loop. "
                "Use 'compiled.ainvoke(...)' instead."
            )

        return sync_handler, async_handler

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

    def _build_text_handler(self, model_id: str):
        def handler(invocation: Dict[str, Any]) -> str:
            parameters = invocation.get("parameters") or {}
            llm = get_llm(
                model=model_id,
                temperature=parameters.get("temperature") or 0.2,
            )
            prompt = self._inject_structured_inputs(
                invocation["prompt"], invocation.get("inputs")
            )
            response = llm.invoke(prompt)
            return _message_to_text(response)

        return handler

    def _build_json_handler(self, model_id: str):
        def handler(invocation: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
            parameters = invocation.get("parameters") or {}
            llm = get_llm(
                model=model_id,
                temperature=parameters.get("temperature") or 0.2,
            )

            prompt = self._inject_structured_inputs(
                invocation["prompt"], invocation.get("inputs")
            )
            schema_block = json.dumps(schema, indent=2)

            structured_prompt = (
                f"{prompt}\n\n"
                "Respond strictly with JSON that satisfies the following schema:\n"
                f"{schema_block}\n"
                "The response must be valid JSON without additional commentary."
            )

            response = llm.invoke(structured_prompt)
            text = _message_to_text(response).strip()

            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ExecutionError(
                    f"Model '{model_id}' returned invalid JSON: {text}"
                ) from exc

        return handler

    # -------------------------------------------------------------------------
    # Compilation pipeline
    # -------------------------------------------------------------------------
    def _compile_spec(self, spec: WorkflowSpec) -> CompiledWorkflow:
        context = CompilerContext(
            schema_registry=self._schema_registry,
            tool_registry=self._tool_registry,
            model_registry=self._model_registry,
        )

        type_env = build_type_environment(
            spec,
            schema_registry=context.schema_registry,
            tool_registry=context.tool_registry,
        )
        validate_references(spec, type_env)
        plan = build_execution_plan(spec)
        runtime = NodeRuntime(
            RuntimeServices(
                schema_registry=context.schema_registry,
                tool_registry=context.tool_registry,
                model_registry=context.model_registry,
                type_env=type_env,
            )
        )
        from workflow_compiler.compiler.emit_langgraph import emit_langgraph
        graph = emit_langgraph(plan, runtime)
        return CompiledWorkflow(
            spec=spec,
            type_env=type_env.as_dict(),
            graph=graph,
            runtime=runtime,
        )
