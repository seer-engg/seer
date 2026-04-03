# pylint: disable=too-many-arguments,too-many-positional-arguments
# Reason: browser_use integration requires matching their action handler signatures
"""
Custom browser-use tools with structured data submission and HITL actions.

Provides:
- submit_result: Bypasses complex StructuredOutputAction for reliable structured output.
- ask_human: Pauses the browser agent to collect human input via asyncio.Future,
  keeping the browser session alive with full page state.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

from browser_use.agent.views import ActionResult
from browser_use.tools.service import Tools
from pydantic import BaseModel, create_model

from seer.logger import get_logger

logger = get_logger(__name__)


# Type mapping from JSON schema types to Python types
_JSON_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _json_type_to_python(schema: Dict[str, Any]) -> type:
    """
    Map JSON schema type to Python type for Pydantic model generation.

    Args:
        schema: JSON schema definition for a single field

    Returns:
        Python type corresponding to the JSON schema type
    """
    json_type = schema.get("type", "string")

    # Handle array type recursively
    if json_type == "array":
        items_schema = schema.get("items", {})
        item_type = _json_type_to_python(items_schema)
        return List[item_type]  # type: ignore[valid-type]

    # Handle object type as Dict
    if json_type == "object":
        return Dict[str, Any]

    return _JSON_TYPE_MAP.get(json_type, str)


def create_submit_result_model(
    schema: Dict[str, Any],
    model_name: str = "SubmitResultParams",
) -> Type[BaseModel]:
    """
    Create a Pydantic model from JSON schema for submit_result action params.

    This creates a flat, simple model that LLMs can easily fill via tool-calling,
    avoiding the complex nested union types that cause parsing failures.

    Args:
        schema: JSON schema dict with "type", "properties", and optional "required"
        model_name: Name for the generated Pydantic model class

    Returns:
        Dynamically generated Pydantic model class

    Example:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"}
            },
            "required": ["name"]
        }
        Model = create_submit_result_model(schema)
        # Model has name (required str) and price (optional float)
    """
    if schema.get("type") != "object":
        # For non-object schemas, wrap in a "data" field
        return create_model(model_name, data=(Any, ...))

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    field_definitions: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = _json_type_to_python(field_schema)
        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (Optional[field_type], None)

    return create_model(model_name, **field_definitions)


# Callback type for HITL: receives (question, context, options) and persists interrupt to DB
HITLCallback = Callable[[str, Optional[str], Optional[List[str]]], Awaitable[None]]

# Factory type for HITL: returns an asyncio.Future to await for the human response
HITLFutureFactory = Callable[[], asyncio.Future]


class CustomBrowserTools(Tools):
    """
    Extended Tools with submit_result and ask_human actions.

    Actions:
    - submit_result: Lets the LLM submit structured data via tool-calling
      instead of the complex StructuredOutputAction wrapper.
    - ask_human: Pauses the agent to collect human input, keeping the
      browser session alive with full page/DOM/JS state.

    Usage:
        tools = CustomBrowserTools(
            extraction_schema=schema,
            enable_hitl=True,
            hitl_callback=on_hitl_request,
            hitl_future_factory=create_future,
            hitl_timeout_seconds=1800,
        )
    """

    def __init__(
        self,
        extraction_schema: Optional[Dict[str, Any]] = None,
        exclude_actions: Optional[List[str]] = None,
        enable_hitl: bool = False,
        hitl_callback: Optional[HITLCallback] = None,
        hitl_future_factory: Optional[HITLFutureFactory] = None,
        hitl_timeout_seconds: int = 1800,
    ):
        """
        Initialize custom tools with optional extraction schema and HITL support.

        Args:
            extraction_schema: JSON schema for structured data extraction.
                If provided, registers a submit_result action with matching params.
            exclude_actions: List of default action names to exclude.
            enable_hitl: When True, registers the ask_human action.
            hitl_callback: Async callback invoked when ask_human is called.
                Responsible for persisting the HITL request to the DB.
            hitl_future_factory: Callable that returns an asyncio.Future
                for the ask_human action to await.
            hitl_timeout_seconds: Max seconds to wait for human response.
        """
        # Don't pass output_model to parent - we handle structured output ourselves
        # This avoids the complex StructuredOutputAction wrapper
        super().__init__(output_model=None, exclude_actions=exclude_actions)

        self._extraction_schema = extraction_schema
        self._extracted_data: Dict[str, Any] = {}

        if extraction_schema:
            self._register_submit_result_action(extraction_schema)

        if enable_hitl:
            if hitl_callback is None or hitl_future_factory is None:
                raise ValueError("hitl_callback and hitl_future_factory are required when enable_hitl=True")  # pylint: disable=line-too-long  # Reason: Error message clarity
            self._register_ask_human_action(hitl_callback, hitl_future_factory, hitl_timeout_seconds)

    def _register_submit_result_action(self, schema: Dict[str, Any]) -> None:
        """
        Register submit_result action based on extraction schema.

        Creates a dynamically-typed action that accepts parameters matching
        the schema fields. When called, it stores the validated data and
        signals task completion.

        Args:
            schema: JSON schema defining expected output structure
        """
        param_model = create_submit_result_model(schema, "SubmitResultParams")
        field_names = list(schema.get("properties", {}).keys())
        fields_desc = ", ".join(field_names) if field_names else "the required data"

        # Store reference to self for closure
        tools_instance = self

        @self.registry.action(
            f"Submit the extracted structured data ({fields_desc}). "
            "Call this action when you have gathered all required information from the page.",
            param_model=param_model,
        )
        async def submit_result(params: param_model) -> ActionResult:  # type: ignore[valid-type]
            """Submit extracted data and complete the task."""
            # Store the validated data via public method
            data = params.model_dump()
            tools_instance.set_extracted_data(data)

            logger.info("submit_result called with data: %s", data)

            return ActionResult(
                is_done=True,
                success=True,
                extracted_content=json.dumps(data, ensure_ascii=False),
                long_term_memory="Structured data submitted successfully via submit_result.",
            )

    def _register_ask_human_action(
        self,
        hitl_callback: HITLCallback,
        hitl_future_factory: HITLFutureFactory,
        hitl_timeout_seconds: int,
    ) -> None:
        """
        Register ask_human action for human-in-the-loop interaction.

        When the LLM calls this action, the agent loop pauses while a human
        is prompted for input. The browser session stays alive in Browserless
        with full page state.

        Args:
            hitl_callback: Async callback to persist HITL request to DB.
            hitl_future_factory: Returns an asyncio.Future to await.
            hitl_timeout_seconds: Max wait time for human response.
        """

        class AskHumanParams(BaseModel):
            question: str
            context: Optional[str] = None
            options: Optional[List[str]] = None

        @self.registry.action(
            "Ask the human operator a question and wait for their response. "
            "Use this when you need clarification, encounter a CAPTCHA, "
            "need login credentials, or face an ambiguous choice. "
            "The browser will stay open while waiting for the human.",
            param_model=AskHumanParams,
        )
        async def ask_human(params: AskHumanParams) -> ActionResult:
            """Pause the agent and ask the human for input."""
            logger.info("ask_human called: question=%s", params.question)

            # Notify the outer system (persists interrupt to DB)
            await hitl_callback(params.question, params.context, params.options)

            # Get a Future to await for the human's response
            future = hitl_future_factory()

            try:
                response = await asyncio.wait_for(future, timeout=hitl_timeout_seconds)
                logger.info("ask_human received response for question=%s", params.question)

                # Format response for the agent
                if isinstance(response, dict):
                    response_text = json.dumps(response, ensure_ascii=False)
                else:
                    response_text = str(response)

                return ActionResult(
                    extracted_content=f"Human response: {response_text}",
                    long_term_memory=f"Asked human: {params.question}. Got response: {response_text}",
                )
            except asyncio.TimeoutError:
                logger.warning("ask_human timed out after %ds for question=%s", hitl_timeout_seconds, params.question)
                return ActionResult(
                    error=f"Human did not respond within {hitl_timeout_seconds} seconds.",
                    long_term_memory=f"Asked human: {params.question}. Timed out waiting for response.",
                )
            except asyncio.CancelledError:
                logger.info("ask_human cancelled for question=%s", params.question)
                return ActionResult(
                    error="The request for human input was cancelled.",
                    long_term_memory=f"Asked human: {params.question}. Request was cancelled.",
                )

    def set_extracted_data(self, data: Dict[str, Any]) -> None:
        """
        Set the extracted data (used by submit_result action).

        Args:
            data: The extracted data to store.
        """
        self._extracted_data = data

    def get_extracted_data(self) -> Dict[str, Any]:
        """
        Get the data submitted via submit_result action.

        Returns:
            Dict containing the extracted data, or empty dict if not yet submitted.
        """
        return self._extracted_data

    def has_extracted_data(self) -> bool:
        """
        Check if data has been submitted via submit_result.

        Returns:
            True if submit_result was called with data.
        """
        return bool(self._extracted_data)
