"""
Simple registry for LLM model backends. The runtime uses this registry to
locate the callable responsible for executing a model request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, MutableMapping, Optional

from seer.core.schema.models import JsonSchema, OutputMode

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

ModelInvocation = Dict[str, Any]  # prompt, inputs, config
TextLLMCallable = Callable[[ModelInvocation], str]
StructuredLLMCallable = Callable[[ModelInvocation, JsonSchema], Any]
ChatModelFactory = Callable[[], "BaseChatModel"]


@dataclass
class ModelDefinition:
    model_id: str
    text_handler: Optional[TextLLMCallable] = None
    json_handler: Optional[StructuredLLMCallable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional factory for creating LangChain chat model instances (for agent nodes)
    chat_model_factory: Optional[ChatModelFactory] = None

    def supports_mode(self, mode: OutputMode) -> bool:
        if mode == OutputMode.text:
            return self.text_handler is not None
        if mode == OutputMode.json:
            return self.json_handler is not None
        return False

    def get_chat_model(
        self,
        temperature: float = 0.2,
        byok_api_key: str | None = None,
        byok_base_url: str | None = None,
        byok_provider: str | None = None,
    ) -> "BaseChatModel":
        """
        Get a LangChain BaseChatModel instance for agent use cases.

        Uses BYOK credentials if provided, else chat_model_factory, else get_llm().
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
        from seer.llm import get_llm

        if byok_api_key:
            from seer.llm import get_llm_for_byok  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
            return get_llm_for_byok(
                model=self.model_id,
                temperature=temperature,
                api_key=byok_api_key,
                provider=byok_provider or "openrouter",
                base_url=byok_base_url or "https://openrouter.ai/api/v1",
            )

        if self.chat_model_factory is not None:
            return self.chat_model_factory()

        return get_llm(model=self.model_id, temperature=temperature)


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
