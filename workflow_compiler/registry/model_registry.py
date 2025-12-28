"""
Simple registry for LLM model backends. The runtime uses this registry to
locate the callable responsible for executing a model request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, MutableMapping, Optional

LLMCallable = Callable[[str, Dict[str, Any]], Any]


@dataclass
class ModelDefinition:
    model_id: str
    handler: LLMCallable
    supports_structured_output: bool = False


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


