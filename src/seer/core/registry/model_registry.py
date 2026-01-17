"""
Simple registry for LLM model backends. The runtime uses this registry to
locate the callable responsible for executing a model request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, MutableMapping, Optional

from seer.core.schema.models import JsonSchema, OutputMode

ModelInvocation = Dict[str, Any]  # prompt, inputs, config
TextLLMCallable = Callable[[ModelInvocation], str]
StructuredLLMCallable = Callable[[ModelInvocation, JsonSchema], Any]


@dataclass
class ModelDefinition:
    model_id: str
    text_handler: Optional[TextLLMCallable] = None
    json_handler: Optional[StructuredLLMCallable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def supports_mode(self, mode: OutputMode) -> bool:
        if mode == OutputMode.text:
            return self.text_handler is not None
        if mode == OutputMode.json:
            return self.json_handler is not None
        return False


class ModelNotFoundError(KeyError):
    """Raised when a model id cannot be resolved."""


class ModelRegistry:
    def __init__(self, initial: MutableMapping[str, ModelDefinition] | None = None) -> None:
        self._models: Dict[str, ModelDefinition] = dict(initial or {})

    def register(self, definition: ModelDefinition) -> None:
        self._models[definition.model_id] = definition

    def get(self, model_id: str) -> ModelDefinition:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ModelNotFoundError(f"Model '{model_id}' is not registered") from exc

    def maybe_get(self, model_id: str) -> Optional[ModelDefinition]:
        return self._models.get(model_id)

    def all(self) -> List[ModelDefinition]:
        return list(self._models.values())
