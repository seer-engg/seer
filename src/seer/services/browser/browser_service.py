# pylint: disable=broad-exception-caught,logging-fstring-interpolation,too-many-arguments,too-many-positional-arguments,too-many-lines
# Reason: Browser automation requires flexible exception handling, dynamic logging, complex configuration, and extensive session/recording logic
"""
Browser automation service using BrowserUse.

Executes browser tasks within workflows using pooled browser sessions.
This is the main entry point for the browser node executor.
"""
from __future__ import annotations

import asyncio
import base64
import json
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
from uuid import UUID

from browser_use import Agent, ChatOpenAI
from pydantic import BaseModel, create_model

from seer.logger import get_logger
from seer.config import config
from seer.database import User
from seer.services.browser.custom_tools import CustomBrowserTools
from seer.services.browser.exceptions import TargetDetachmentError
from seer.services.browser.pool_manager import BrowserPoolManager
from seer.services.browser.recording_service import RecordingService
from seer.services.browser.session_context_manager import SessionContextManager

# Target recovery configuration
TARGET_RECOVERY_MAX_ATTEMPTS = 3
TARGET_RECOVERY_DELAY_SECONDS = 2.0

if TYPE_CHECKING:
    from browser_use import BrowserSession
    from seer.core.files.service import WorkflowFileSystem
    from seer.services.browser.pool_manager import ManagedSession


logger = get_logger(__name__)


_JSON_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    # "object" is handled by recursive model creation in _json_type_to_python
}

# Counter for generating unique model names to avoid Pydantic model name collisions
# Using a mutable container to avoid global statement
_MODEL_COUNTER = [0]


def _strip_markdown_fencing(text: str) -> str:
    """
    Remove markdown code fences from LLM output.

    Some LLMs (e.g., kimi-k2.5 via OpenRouter) wrap JSON responses in markdown
    code fences like ```json ... ```. This breaks Pydantic's JSON parser which
    expects raw JSON. This helper strips those fences before parsing.

    Args:
        text: Raw LLM output that may contain markdown fencing

    Returns:
        Cleaned text with markdown fencing removed
    """
    text = text.strip()
    # Handle ```json or ```JSON (case-insensitive language tag)
    if text.lower().startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]  # Remove ``` (generic code fence)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def format_browser_history(history: Any) -> Dict[str, Any]:
    """
    Format AgentHistoryList into a human-readable summary.

    Transforms verbose browser_use history into a clean format showing:
    - Steps taken with action types and descriptions
    - Completion and success status
    - Any errors encountered

    Args:
        history: AgentHistoryList from browser_use agent.run()

    Returns:
        Dict with steps, completed, success, and optional errors
    """
    if history is None:
        return {"steps": [], "completed": False, "success": False}

    steps = []

    try:
        action_names = history.action_names() if hasattr(history, "action_names") else []
        action_results = history.action_results() if hasattr(history, "action_results") else []

        for i, action_result in enumerate(action_results):
            step: Dict[str, Any] = {"action": action_names[i] if i < len(action_names) else "unknown"}

            # Use extracted_content (human-readable) or fall back to long_term_memory
            if action_result.extracted_content:
                step["description"] = action_result.extracted_content
            elif action_result.long_term_memory:
                step["description"] = action_result.long_term_memory
            else:
                step["description"] = f"Performed {step['action']} action"

            if action_result.error:
                step["error"] = action_result.error

            steps.append(step)

        result: Dict[str, Any] = {
            "steps": steps,
            "completed": history.is_done() if hasattr(history, "is_done") else False,
            "success": history.is_successful() if hasattr(history, "is_successful") else None,
        }

        if hasattr(history, "has_errors") and history.has_errors():
            errors = history.errors() if hasattr(history, "errors") else []
            result["errors"] = [e for e in errors if e is not None]

        return result

    except Exception as e:
        logger.warning(f"Failed to format browser history: {e}")
        return {"steps": [], "completed": False, "success": False, "format_error": str(e)}


def _json_type_to_python(schema: Dict[str, Any], model_name_prefix: str = "Nested") -> type:
    """
    Map JSON schema type to Python type for Pydantic model generation.

    Recursively creates nested Pydantic models for object types with properties.
    This is required for browser-use's output_model_schema to work correctly with
    OpenAI's strict JSON Schema validation.

    Args:
        schema: JSON schema definition for a single field
        model_name_prefix: Prefix for generated model names (used for nested objects)

    Returns:
        Python type corresponding to the JSON schema type
    """
    json_type = schema.get("type", "any")

    # Handle array type - recursively resolve item type
    if json_type == "array":
        items_schema = schema.get("items", {})
        item_type = _json_type_to_python(items_schema, f"{model_name_prefix}Item")
        return List[item_type]  # type: ignore[valid-type]

    # Handle object type with properties - create a nested Pydantic model
    if json_type == "object" and "properties" in schema:
        return json_schema_to_pydantic(schema, model_name_prefix)

    # Handle object without properties as generic dict (fallback)
    if json_type == "object":
        return Dict[str, Any]  # type: ignore[misc]

    return _JSON_TYPE_MAP.get(json_type, Any)


def json_schema_to_pydantic(schema: Dict[str, Any], model_name: str = "DynamicModel") -> Type[BaseModel]:
    """
    Convert a JSON schema dict to a Pydantic model class.

    Recursively creates nested Pydantic models for object types with properties.
    This is required for browser-use's output_model_schema to work correctly with
    OpenAI's strict JSON Schema validation, which requires properly typed nested models
    instead of Dict[str, Any].

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
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}}
                    }
                }
            },
            "required": ["name"]
        }
        Model = json_schema_to_pydantic(schema)
        # Model has name: str, items: List[NestedModel] (not List[Dict])
    """
    _MODEL_COUNTER[0] += 1
    unique_name = f"{model_name}_{_MODEL_COUNTER[0]}"

    if schema.get("type") != "object":
        # For non-object schemas, wrap in a simple model with a "data" field
        return create_model(unique_name, data=(Any, ...))

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    field_definitions: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        # Pass unique prefix for nested model names
        field_type = _json_type_to_python(field_schema, f"{unique_name}_{field_name.title()}")
        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (Optional[field_type], None)

    return create_model(unique_name, **field_definitions)


class BrowserService:
    """
    Singleton service for browser automation task execution.

    Uses BrowserUse Agent for LLM-driven browser automation with
    pooled browser sessions for concurrency control and session persistence.
    """

    _instance: Optional["BrowserService"] = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._session_context = SessionContextManager()

    @classmethod
    def instance(cls) -> "BrowserService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def _load_storage_state(
        self, user: Optional[User], browser_profile_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Load browser profile storage state if available."""
        if not browser_profile_id or not user:
            return None
        try:
            storage_state = await self._session_context.load_session_state(
                user, UUID(browser_profile_id)
            )
            if storage_state:
                logger.info(f"Loaded session from profile {browser_profile_id}")
            return storage_state
        except Exception as e:
            logger.warning(f"Failed to load profile {browser_profile_id}: {e}")
            return None

    def _create_output_model(
        self, extraction_schema: Optional[Dict[str, Any]]
    ) -> Optional[Type[BaseModel]]:
        """Create Pydantic model from extraction schema if provided."""
        if not extraction_schema:
            return None
        try:
            model = json_schema_to_pydantic(extraction_schema, "BrowserOutputModel")
            logger.debug(f"Created dynamic Pydantic model from schema: {model}")
            return model
        except Exception as e:
            logger.warning(f"Failed to create Pydantic model from schema: {e}")
            return None

    @staticmethod
    def _build_error_result(message: str) -> Dict[str, Any]:
        """Build a standardized error result dict."""
        return {
            "success": False,
            "result": message,
            "extracted_data": {},
            "final_url": None,
            "urls": [],
            "duration_seconds": None,
            "steps_count": None,
            "extracted_content": [],
            "model_thoughts": [],
            "model_actions": [],
            "screenshots": [],
        }

    @staticmethod
    def _uses_custom_submit_tool(model_name: str) -> bool:
        """
        Check if model should use custom submit_result tool vs standard structured output.

        Models like Kimi-k2.5 excel at tool-calling but struggle with complex
        nested schemas like StructuredOutputAction[T]. For these models, we use
        a simpler submit_result tool approach.

        Args:
            model_name: The model identifier (e.g., "moonshotai/kimi-k2.5")

        Returns:
            True if model should use custom submit_result tool
        """
        kimi_patterns = ["kimi", "moonshot"]
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in kimi_patterns)

    async def _run_browser_agent(
        self,
        task: str,
        inputs: Dict[str, Any],
        extraction_schema: Optional[Dict[str, Any]],
        max_steps: int,
        timeout_seconds: int,
        browser_session: "BrowserSession",
        model: Optional[str] = None,
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """
        Run the BrowserUse agent with a managed browser session.

        Uses model-aware structured output:
        - For Kimi/Moonshot: Uses custom submit_result tool for reliable extraction
        - For OpenAI/Claude: Uses standard output_model_schema

        Args:
            task: The task description
            inputs: Additional context data
            extraction_schema: JSON schema for structured output
            max_steps: Maximum agent steps
            timeout_seconds: Task timeout
            browser_session: The browser session to use
            model: OpenRouter model identifier. If None, uses config.default_llm_model

        Returns:
            Tuple of (agent_history, tool_extracted_data).
            tool_extracted_data is set when using custom submit_result tool.
        """
        llm = self._get_agent_llm(model)
        model_name = getattr(llm, "model", "")

        # Decide approach based on model capabilities
        use_custom_tool = self._uses_custom_submit_tool(model_name) and extraction_schema

        if use_custom_tool:
            # Kimi/Moonshot: Use custom submit_result tool
            custom_tools = CustomBrowserTools(extraction_schema=extraction_schema)
            enhanced_task = self._enhance_task(task, inputs, extraction_schema, use_submit_tool=True)

            agent = Agent(
                task=enhanced_task,
                llm=llm,
                browser_session=browser_session,
                tools=custom_tools,
                output_model_schema=None,  # No standard structured output
                calculate_cost=True,
            )

            history = await asyncio.wait_for(
                agent.run(max_steps=max_steps),
                timeout=timeout_seconds
            )

            # Return data from custom tool if available
            return history, custom_tools.get_extracted_data() if custom_tools.has_extracted_data() else None

        # OpenAI/Claude: Use standard structured output
        output_model = self._create_output_model(extraction_schema)
        enhanced_task = self._enhance_task(task, inputs, extraction_schema, use_submit_tool=False)

        agent = Agent(
            task=enhanced_task,
            llm=llm,
            browser_session=browser_session,
            output_model_schema=output_model,
            calculate_cost=True,
        )

        history = await asyncio.wait_for(
            agent.run(max_steps=max_steps),
            timeout=timeout_seconds
        )

        return history, None  # Standard extraction handled by caller

    async def _start_recording_if_enabled(
        self,
        managed: "ManagedSession",
        start_url: Optional[str],
        *,
        user: Optional[User] = None,
        browser_profile_id: Optional[str] = None,
        workflow_run_id: Optional[str] = None,
    ) -> None:
        """Start rrweb recording for session replay if enabled.

        With chunked recording support, events are periodically flushed to the
        database during long-running sessions to prevent data loss on CDP target
        detachment.
        """
        if not config.browser_recording_enabled:
            return
        try:
            recorder = await RecordingService.get_instance()
            managed.recording_id = await recorder.start_recording(
                managed.id,
                managed.session,
                start_url=start_url,
                user=user,
                profile_id=browser_profile_id,
                workflow_run_id=workflow_run_id,
            )
            managed.start_url = start_url
        except Exception as e:
            logger.warning(f"Failed to start workflow recording: {e}")

    async def _save_recording_if_enabled(
        self,
        managed: "ManagedSession",
        user: Optional[User],
        browser_profile_id: Optional[str],
        workflow_run_id: Optional[str],
    ) -> None:
        """Save rrweb recording if enabled and recording was started."""
        if not (managed.recording_id and config.browser_recording_enabled and user):
            return
        try:
            recorder = await RecordingService.get_instance()
            await recorder.save_recording(
                managed.id,
                user,
                profile_id=browser_profile_id,
                workflow_run_id=workflow_run_id,
                session_type="workflow",
                start_url=managed.start_url,
            )
        except Exception as e:
            logger.warning(f"Failed to save workflow recording: {e}")

    async def _run_browser_agent_with_recovery(
        self,
        task: str,
        inputs: Dict[str, Any],
        extraction_schema: Optional[Dict[str, Any]],
        max_steps: int,
        timeout_seconds: int,
        browser_session: "BrowserSession",
        model: Optional[str] = None,
        managed_session: Optional["ManagedSession"] = None,
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """Run agent with automatic recovery from transient target detachment.

        Wraps _run_browser_agent with retry logic for "No valid agent focus"
        errors that can occur when CDP targets detach during long-running tasks.

        Args:
            Same as _run_browser_agent, plus:
            managed_session: Session reference for error context

        Returns:
            Tuple of (agent_history, tool_extracted_data)

        Raises:
            TargetDetachmentError: If recovery fails after all attempts
        """
        last_error: Optional[Exception] = None

        for attempt in range(TARGET_RECOVERY_MAX_ATTEMPTS):
            try:
                return await self._run_browser_agent(
                    task, inputs, extraction_schema, max_steps, timeout_seconds,
                    browser_session, model
                )
            except ValueError as e:
                error_str = str(e).lower()
                if "no valid agent focus" in error_str:
                    last_error = e
                    session_id = managed_session.id if managed_session else "unknown"

                    if attempt < TARGET_RECOVERY_MAX_ATTEMPTS - 1:
                        logger.warning(
                            f"Target focus lost for session {session_id}, "
                            f"attempting recovery (attempt {attempt + 1}/{TARGET_RECOVERY_MAX_ATTEMPTS})"
                        )
                        await asyncio.sleep(TARGET_RECOVERY_DELAY_SECONDS)

                        # Try to recover focus via browser-use's internal mechanism
                        recovered = await self._try_recover_focus(browser_session)
                        if not recovered:
                            raise TargetDetachmentError(session_id) from e
                        continue
                    # Max attempts reached
                    raise TargetDetachmentError(
                        session_id,
                        f"Failed to recover from target detachment after {TARGET_RECOVERY_MAX_ATTEMPTS} attempts"
                    ) from e
                # Re-raise non-focus errors
                raise

        # Should not reach here, but handle gracefully
        session_id = managed_session.id if managed_session else "unknown"
        raise TargetDetachmentError(session_id, "Recovery exhausted") from last_error

    async def _try_recover_focus(self, browser_session: "BrowserSession") -> bool:
        """Attempt to recover browser focus after target detachment.

        Waits briefly for browser-use's internal recovery, then validates
        that a valid page is accessible.

        Returns:
            True if recovery succeeded, False otherwise
        """
        try:
            # Give browser-use time to run its internal recovery
            await asyncio.sleep(1.0)

            # Try to get current page - this triggers browser-use's focus recovery
            page = await asyncio.wait_for(
                browser_session.must_get_current_page(),
                timeout=5.0
            )
            return page is not None
        except Exception as e:
            logger.warning(f"Focus recovery failed: {e}")
            return False

    def _categorize_error(self, error: Exception) -> str:
        """Categorize error type for metrics and handling.

        Returns:
            Error category string: "target_detachment", "cdp_error",
            "timeout", "connection_error", or "unknown"
        """
        error_str = str(error).lower()

        if "no valid agent focus" in error_str:
            return "target_detachment"
        if "cdp" in error_str or "chrome" in error_str:
            return "cdp_error"
        if "timeout" in error_str:
            return "timeout"
        if "connection" in error_str:
            return "connection_error"

        return "unknown"

    async def _emergency_save_recording(
        self,
        managed: "ManagedSession",
        user: Optional[User],
        browser_profile_id: Optional[str],
        workflow_run_id: Optional[str],
    ) -> bool:
        """Emergency save recording when session fails unexpectedly.

        Attempts to save whatever recording data exists (including any
        already-flushed chunks) when the browser session encounters
        an unrecoverable error.

        Returns:
            True if recording was saved, False otherwise
        """
        if not (managed.recording_id and config.browser_recording_enabled and user):
            return False

        try:
            recorder = await RecordingService.get_instance()
            # save_recording handles chunked recordings gracefully
            recording_id = await recorder.save_recording(
                managed.id,
                user,
                profile_id=browser_profile_id,
                workflow_run_id=workflow_run_id,
                session_type="workflow",
                start_url=managed.start_url,
            )
            if recording_id:
                logger.info(f"Emergency saved recording {recording_id} for failed session {managed.id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed emergency recording save for session {managed.id}: {e}")
            return False

    async def execute_task(  # pylint: disable=too-many-locals  # Reason: Task execution requires pool, session, agent history, extracted data, error handling, and final state variables
        self,
        user: Optional[User],
        task: str,
        inputs: Dict[str, Any],
        *,
        browser_profile_id: Optional[str] = None,
        max_steps: int = 25,
        timeout_seconds: int = 300,
        extraction_schema: Optional[Dict[str, Any]] = None,
        save_screenshots: bool = False,
        file_system: Optional["WorkflowFileSystem"] = None,
        workflow_run_id: Optional[str] = None,
        model: Optional[str] = None,
        organization_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a browser automation task using a pooled browser session.

        Uses BrowserUse Agent to interpret and execute natural language
        browser automation tasks, with pooled session management and
        automatic session state persistence.

        Args:
            user: User context for profile resolution
            task: Natural language task description
            inputs: Additional context data passed to the agent
            browser_profile_id: UUID of profile with saved login sessions
            max_steps: Maximum BrowserUse agent steps
            timeout_seconds: Task timeout in seconds
            extraction_schema: JSON schema for structured output extraction
            save_screenshots: When True, save screenshots to S3 as WorkflowFileRef
            file_system: WorkflowFileSystem for saving screenshots
            workflow_run_id: Run ID for organizing screenshot files
            model: OpenRouter model identifier. If None, uses config.default_llm_model
            organization_id: Organization ID for file ownership

        Returns:
            Result dict with success, result, extracted_data, final_url, screenshots
        """
        logger.info(f"Executing browser task: {task[:100]}...")

        pool = await BrowserPoolManager.get_instance()
        managed = await pool.create_session(
            user_id=str(user.user_id) if user else "anonymous",
            profile_id=browser_profile_id,
            session_type="workflow",
            storage_state=await self._load_storage_state(user, browser_profile_id),
            timeout=timeout_seconds,
        )

        await self._start_recording_if_enabled(
            managed,
            inputs.get("url") or inputs.get("start_url"),
            user=user,
            browser_profile_id=browser_profile_id,
            workflow_run_id=workflow_run_id,
        )

        try:
            history, tool_extracted_data = await self._run_browser_agent_with_recovery(
                task, inputs, extraction_schema, max_steps, timeout_seconds,
                browser_session=managed.session,
                model=model,
                managed_session=managed,
            )

            # Get extracted data - from custom tool (Kimi) or standard extraction
            if tool_extracted_data is not None:
                # Kimi: Data from submit_result tool
                extracted_data = tool_extracted_data
            elif extraction_schema:
                # OpenAI/Claude: Standard extraction from history
                extracted_data = await self._extract_structured_data(history, extraction_schema)
            else:
                # No schema: Basic extraction
                extracted_data = self._extract_data(history)

            # If extraction_schema was provided but extraction failed, return failure
            # This prevents downstream validation errors on empty extracted_data
            # Still include available metadata for debugging
            if extraction_schema and not extracted_data:
                logger.warning("Browser agent completed but failed to extract structured data")
                return {
                    **self._build_error_result("Failed to extract structured data from agent output"),
                    "final_url": self._extract_final_url(history),
                    "urls": self._extract_urls(history),
                    "duration_seconds": self._extract_duration(history),
                    "steps_count": self._extract_steps_count(history),
                    "extracted_content": self._extract_all_content(history),
                    "model_thoughts": self._extract_model_thoughts(history),
                    "model_actions": self._extract_model_actions(history),
                    "usage": self._extract_usage_metadata(history, model),
                }

            return {
                "success": True,
                "result": format_browser_history(history) if history else {},
                "extracted_data": extracted_data if extracted_data else {},
                "final_url": self._extract_final_url(history),
                "urls": self._extract_urls(history),
                "duration_seconds": self._extract_duration(history),
                "steps_count": self._extract_steps_count(history),
                "extracted_content": self._extract_all_content(history),
                "model_thoughts": self._extract_model_thoughts(history),
                "model_actions": self._extract_model_actions(history),
                "screenshots": await self._save_screenshots(
                    history=history,
                    save_screenshots=save_screenshots,
                    file_system=file_system,
                    workflow_run_id=workflow_run_id,
                    user=user,
                    organization_id=organization_id,
                ),
                "usage": self._extract_usage_metadata(history, model),
            }

        except TargetDetachmentError as e:
            # Target detachment with recovery failed - emergency save recording
            logger.error(f"Target detachment during task: {e}")
            await self._emergency_save_recording(managed, user, browser_profile_id, workflow_run_id)
            return {
                **self._build_error_result(f"Browser target detached: {str(e)}"),
                "error_type": "target_detachment",
                "partial_recording_saved": True,
                "usage": None,
            }
        except asyncio.TimeoutError:
            logger.warning(f"Browser task timed out after {timeout_seconds}s")
            return {
                **self._build_error_result(f"Task timed out after {timeout_seconds} seconds"),
                "error_type": "timeout",
                "usage": None,
            }
        except Exception as e:
            # Categorize and log the error
            error_type = self._categorize_error(e)
            logger.error(f"Browser task failed ({error_type}): {e}")

            # Emergency save for target-related errors
            if error_type == "target_detachment":
                await self._emergency_save_recording(managed, user, browser_profile_id, workflow_run_id)

            return {
                **self._build_error_result(f"Task failed: {str(e)}"),
                "error_type": error_type,
                "usage": None,
            }
        finally:
            await self._save_recording_if_enabled(managed, user, browser_profile_id, workflow_run_id)
            final_state = await pool.release_session(managed.id)
            if final_state and browser_profile_id and user:
                await self._session_context.save_session_state(
                    user, UUID(browser_profile_id), final_state
                )

    def _get_agent_llm(self, model: Optional[str] = None) -> Any:
        """
        Get the LLM instance for the BrowserUse agent.

        BrowserUse accesses llm.provider for telemetry and feature detection.
        We add this attribute dynamically to support OpenRouter-backed models.

        Args:
            model: OpenRouter model identifier. If None, uses config.default_llm_model.
        """
        api_key = config.openrouter_api_key
        if api_key is None or api_key == "":
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        # Use provided model or fall back to config default
        resolved_model = model if model else config.default_llm_model
        logger.info(f"Using OpenRouter API | Model: {resolved_model} | Base URL: https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model=resolved_model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )

    @staticmethod
    def _extract_usage_metadata(history: Any, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Extract LLM usage metadata from browser_use agent history.

        Converts browser_use's UsageSummary (aggregated across all agent steps)
        to the standard usage_metadata dict format for cost tracking.

        Args:
            history: AgentHistoryList from browser_use agent.run()
            model: The model that was used. If None, uses config.default_llm_model.

        Returns:
            Usage metadata dict, or None if usage unavailable.
        """
        try:
            if not hasattr(history, "usage") or history.usage is None:
                logger.debug("No usage data available from browser agent history")
                return None

            usage = history.usage
            if usage.total_tokens == 0:
                logger.debug("Browser agent reported zero token usage")
                return None

            # Use provided model or fall back to config default
            resolved_model = model if model else config.default_llm_model

            return {
                "model": resolved_model,
                "input_tokens": usage.total_prompt_tokens,
                "output_tokens": usage.total_completion_tokens,
                "reasoning_tokens": 0,
                "total_tokens": usage.total_tokens,
                "steps_taken": usage.entry_count,
            }
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Usage extraction must not break workflow
            logger.warning(f"Failed to extract usage metadata from browser history: {e}")
            return None

    @staticmethod
    def _extract_final_url(history: Any) -> Optional[str]:
        """Get the last visited URL from browser history."""
        try:
            urls = history.urls() if hasattr(history, "urls") else []
            return urls[-1] if urls else None
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: URL extraction must not break workflow
            return None

    @staticmethod
    def _extract_urls(history: Any) -> List[str]:
        """Get all visited URLs from browser history."""
        try:
            return history.urls() if hasattr(history, "urls") else []
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: URL extraction must not break workflow
            return []

    @staticmethod
    def _extract_duration(history: Any) -> Optional[float]:
        """Get total execution duration in seconds."""
        try:
            if hasattr(history, "total_duration_seconds"):
                return history.total_duration_seconds()
            return None
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Duration extraction must not break workflow
            return None

    @staticmethod
    def _extract_steps_count(history: Any) -> Optional[int]:
        """Get total number of steps executed."""
        try:
            if hasattr(history, "number_of_steps"):
                return history.number_of_steps()
            return None
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Steps extraction must not break workflow
            return None

    @staticmethod
    def _extract_all_content(history: Any) -> List[str]:
        """Get extracted content from all steps."""
        try:
            if hasattr(history, "extracted_content"):
                content = history.extracted_content()
                return [c for c in content if c] if content else []
            return []
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Content extraction must not break workflow
            return []

    @staticmethod
    def _extract_model_thoughts(history: Any) -> List[Dict[str, Any]]:
        """Get agent reasoning/brain from all steps."""
        try:
            if hasattr(history, "model_thoughts"):
                thoughts = history.model_thoughts()
                # Convert AgentBrain objects to dicts
                return [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in thoughts if t]
            return []
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Thoughts extraction must not break workflow
            return []

    @staticmethod
    def _extract_model_actions(history: Any) -> List[Dict[str, Any]]:
        """Get detailed action info from all steps."""
        try:
            if hasattr(history, "model_actions"):
                actions = history.model_actions()
                # Convert action objects to dicts
                return [a.model_dump() if hasattr(a, "model_dump") else dict(a) for a in actions if a]
            return []
        except Exception:  # pylint: disable=broad-exception-caught  # Reason: Actions extraction must not break workflow
            return []

    def _enhance_task(
        self,
        task: str,
        inputs: Dict[str, Any],
        extraction_schema: Optional[Dict[str, Any]] = None,
        use_submit_tool: bool = False,
    ) -> str:
        """
        Enhance task description with input context and extraction instructions.

        Appends structured context from workflow inputs and provides
        model-appropriate instructions for structured data extraction.

        Args:
            task: Original task description
            inputs: Additional context from workflow
            extraction_schema: JSON schema for expected output structure
            use_submit_tool: If True, instruct to use submit_result action (for Kimi)

        Returns:
            Enhanced task string with context and instructions
        """
        enhanced = task

        if inputs:
            inputs_str = json.dumps(inputs, indent=2)
            enhanced += f"\n\nAdditional context:\n{inputs_str}"

        if extraction_schema:
            fields = list(extraction_schema.get("properties", {}).keys())
            fields_str = ", ".join(fields) if fields else "the requested data"

            if use_submit_tool:
                # Kimi: Explicit instructions to use submit_result tool
                enhanced += (
                    f"\n\nIMPORTANT: When you have gathered all the required information, "
                    f"you MUST call the submit_result action with these fields: {fields_str}. "
                    f"Do NOT complete the task with the done action until you have called submit_result "
                    f"with the extracted data."
                )
            else:
                # OpenAI/Claude: Standard instruction (structured output handles it)
                enhanced += f"\n\nIMPORTANT: Make sure to collect and return: {fields_str}"

        return enhanced

    def _extract_data(self, result: Any) -> Dict[str, Any]:
        """
        Extract structured data from agent result.

        Attempts to parse the result as structured data for downstream
        workflow nodes to consume.

        Args:
            result: Raw result from BrowserUse agent

        Returns:
            Dict with extracted data, or raw result under "raw" key
        """
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                pass
        return {"raw": str(result)}

    async def _extract_structured_data(
        self,
        history: Any,
        extraction_schema: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from agent history.

        When output_model_schema is used (via extraction_schema), the agent's
        done action returns pure JSON that can be directly parsed. Some LLMs
        wrap JSON in markdown code fences, which we strip before parsing.

        Args:
            history: AgentHistory from BrowserUse agent.run()
            extraction_schema: JSON schema that was used to create the output model

        Returns:
            Dict with extracted data conforming to the schema, or None if
            extraction_schema was provided but parsing failed (signals failure)
        """
        if not extraction_schema:
            return self._extract_data(history)

        try:
            final_result = history.final_result()
            if final_result:
                if isinstance(final_result, str):
                    # Strip markdown code fences that some LLMs add around JSON
                    cleaned = _strip_markdown_fencing(final_result)
                    return json.loads(cleaned)
                if isinstance(final_result, dict):
                    return final_result
            logger.warning("No final result from agent history - extraction failed")
            return None  # Signal extraction failure when schema was expected
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse structured output as JSON: {e}")
            return None  # Signal extraction failure
        except Exception as e:
            logger.warning(f"Failed to extract structured data: {e}")
            return None  # Signal extraction failure

    async def _save_screenshots(
        self,
        history: Any,
        save_screenshots: bool,
        file_system: Optional["WorkflowFileSystem"],
        workflow_run_id: Optional[str],
        user: Optional[User],
        organization_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Save screenshots from browser history to S3.

        Screenshots are saved as WorkflowFileRef objects that can be
        accessed by downstream workflow nodes. Also creates WorkflowFile
        database records for file management API visibility.

        Args:
            history: AgentHistory from BrowserUse agent.run()
            save_screenshots: Whether screenshot saving is enabled
            file_system: WorkflowFileSystem for S3 uploads
            workflow_run_id: Run ID for organizing files
            user: User who owns the files
            organization_id: Organization ID for file ownership

        Returns:
            List of WorkflowFileRef dicts with file references
        """
        if not save_screenshots:
            return []

        if not file_system or not workflow_run_id or not user:
            logger.warning(
                "Screenshot saving requested but missing required context "
                f"(file_system={file_system is not None}, "
                f"run_id={workflow_run_id}, user={user is not None})"
            )
            return []

        screenshots_result: List[Dict[str, Any]] = []

        try:
            screenshots_b64 = history.screenshots()
            if not screenshots_b64:
                logger.debug("No screenshots captured during browser task")
                return []

            for idx, screenshot_b64 in enumerate(screenshots_b64):
                if not screenshot_b64:
                    continue

                filename = f"screenshot_{idx:03d}.png"
                try:
                    screenshot_data = base64.b64decode(screenshot_b64)
                    file_ref = await file_system.store_file_with_record(
                        user=user,
                        run_id=workflow_run_id,
                        filename=filename,
                        data=screenshot_data,
                        mime_type="image/png",
                        source_tool="browser_screenshot",
                        organization_id=organization_id,
                    )
                    screenshots_result.append(file_ref.to_dict())
                    logger.debug(f"Saved screenshot {filename} to S3")
                except Exception as e:
                    logger.warning(f"Failed to save screenshot {filename}: {e}")

            logger.info(f"Saved {len(screenshots_result)} screenshots to S3")

        except Exception as e:
            logger.warning(f"Failed to retrieve screenshots from history: {e}")

        return screenshots_result

    @property
    def profile_manager(self) -> Any:
        """Get the profile manager for direct profile operations."""
        from seer.services.browser.profile_manager import BrowserProfileManager  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular import
        return BrowserProfileManager()
