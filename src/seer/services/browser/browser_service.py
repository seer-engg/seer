# pylint: disable=broad-exception-caught,logging-fstring-interpolation,too-many-arguments,too-many-positional-arguments
# Reason: Browser automation requires flexible exception handling, dynamic logging, and complex configuration
"""
Browser automation service using BrowserUse.

Executes browser tasks within workflows using persisted profiles.
This is the main entry point for the browser node executor.
"""
from __future__ import annotations

import asyncio
import json
import logging
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
from uuid import UUID

from browser_use import Agent, Browser, ChatOpenAI
from playwright.async_api import async_playwright
from pydantic import BaseModel, create_model

from seer.config import config
from seer.database import User
from seer.services.browser.profile_manager import BrowserProfileManager

if TYPE_CHECKING:
    from seer.core.files.service import WorkflowFileSystem

logger = logging.getLogger(__name__)


_JSON_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": Dict[str, Any],  # type: ignore[misc]
}


def _json_type_to_python(schema: Dict[str, Any]) -> type:
    """
    Map JSON schema type to Python type for Pydantic model generation.

    Args:
        schema: JSON schema definition for a single field

    Returns:
        Python type corresponding to the JSON schema type
    """
    json_type = schema.get("type", "any")

    # Handle array type separately due to recursive item type resolution
    if json_type == "array":
        items_schema = schema.get("items", {})
        item_type = _json_type_to_python(items_schema)
        return List[item_type]  # type: ignore[valid-type]

    return _JSON_TYPE_MAP.get(json_type, Any)


def json_schema_to_pydantic(schema: Dict[str, Any], model_name: str = "DynamicModel") -> Type[BaseModel]:
    """
    Convert a JSON schema dict to a Pydantic model class.

    Supports basic JSON schema types: string, number, integer, boolean, array, object.
    This enables BrowserUse's output_model_schema feature which forces structured JSON output.

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
        Model = json_schema_to_pydantic(schema)
        # Model is now a Pydantic class with name (required str) and price (optional float)
    """
    if schema.get("type") != "object":
        # For non-object schemas, wrap in a simple model with a "data" field
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


class BrowserService:
    """
    Singleton service for browser automation task execution.

    Uses BrowserUse Agent for LLM-driven browser automation with
    Playwright for browser management and session persistence.
    """

    _instance: Optional["BrowserService"] = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._profile_manager = BrowserProfileManager()

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
            storage_state = await self._profile_manager.get_session_state(
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
            "screenshots": [],
        }

    async def _run_browser_agent(
        self,
        task: str,
        inputs: Dict[str, Any],
        extraction_schema: Optional[Dict[str, Any]],
        max_steps: int,
        timeout_seconds: int,
    ) -> Any:
        """Run the BrowserUse agent and return history."""
        llm = self._get_agent_llm()
        enhanced_task = self._enhance_task(task, inputs)
        output_model = self._create_output_model(extraction_schema)

        browser_use_browser = Browser()
        agent = Agent(
            task=enhanced_task,
            llm=llm,
            browser=browser_use_browser,
            output_model_schema=output_model,
        )

        return await asyncio.wait_for(
            agent.run(max_steps=max_steps),
            timeout=timeout_seconds
        )

    async def execute_task(
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
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a browser automation task.

        Uses BrowserUse Agent to interpret and execute natural language
        browser automation tasks, optionally with a persisted session.

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
            user_id: User ID for scoping screenshot files

        Returns:
            Result dict with success, result, extracted_data, final_url, screenshots
        """
        logger.info(f"Executing browser task: {task[:100]}...")

        storage_state = await self._load_storage_state(user, browser_profile_id)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            try:
                context = await browser.new_context(storage_state=storage_state)
                page = await context.new_page()

                history = await self._run_browser_agent(
                    task, inputs, extraction_schema, max_steps, timeout_seconds
                )

                return {
                    "success": True,
                    "result": str(history) if history else "",
                    "extracted_data": await self._extract_structured_data(history, extraction_schema),
                    "final_url": page.url if page else None,
                    "screenshots": await self._save_screenshots(
                        history=history,
                        save_screenshots=save_screenshots,
                        file_system=file_system,
                        workflow_run_id=workflow_run_id,
                        user_id=user_id,
                    ),
                }

            except asyncio.TimeoutError:
                logger.warning(f"Browser task timed out after {timeout_seconds}s")
                return self._build_error_result(f"Task timed out after {timeout_seconds} seconds")
            except Exception as e:
                logger.error(f"Browser task failed: {e}")
                return self._build_error_result(f"Task failed: {str(e)}")
            finally:
                await browser.close()

    def _get_agent_llm(self) -> Any:
        """
        Get the LLM instance for the BrowserUse agent.

        BrowserUse accesses llm.provider for telemetry and feature detection.
        We add this attribute dynamically to support OpenRouter-backed models.
        """
        # Use the standard get_llm which routes through OpenRouter
        # llm = get_llm(model="gpt-4o", temperature=0.1)
        api_key = config.openrouter_api_key
        if api_key is None or api_key == "":
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        model = 'moonshotai/kimi-k2.5'
        logger.info(f"🌐 Using OpenRouter API | Model: {model} | Base URL: https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )

    def _enhance_task(self, task: str, inputs: Dict[str, Any]) -> str:
        """
        Enhance task description with input context.

        Appends structured context from workflow inputs to help
        the agent understand the full task requirements.

        Args:
            task: Original task description
            inputs: Additional context from workflow

        Returns:
            Enhanced task string with context
        """
        if not inputs:
            return task
        inputs_str = json.dumps(inputs, indent=2)
        return f"{task}\n\nAdditional context:\n{inputs_str}"

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
            # Try to parse as JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                pass
        return {"raw": str(result)}

    async def _extract_structured_data(
        self,
        history: Any,
        extraction_schema: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extract structured data from agent history.

        When output_model_schema is used (via extraction_schema), the agent's
        done action returns pure JSON that can be directly parsed.

        Args:
            history: AgentHistory from BrowserUse agent.run()
            extraction_schema: JSON schema that was used to create the output model

        Returns:
            Dict with extracted data conforming to the schema
        """
        if not extraction_schema:
            # No structured output requested, use legacy extraction
            return self._extract_data(history)

        # Use BrowserUse's structured output - with output_model_schema,
        # final_result() returns pure JSON (not markdown)
        try:
            final_result = history.final_result()
            if final_result:
                if isinstance(final_result, str):
                    return json.loads(final_result)
                if isinstance(final_result, dict):
                    return final_result
            logger.debug("No final result from agent history")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse structured output as JSON: {e}")
            # Fall back to legacy extraction on parse failure
            return self._extract_data(history)
        except Exception as e:
            logger.warning(f"Failed to extract structured data: {e}")
            return {}

    async def _save_screenshots(
        self,
        history: Any,
        save_screenshots: bool,
        file_system: Optional["WorkflowFileSystem"],
        workflow_run_id: Optional[str],
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Save screenshots from browser history to S3.

        Screenshots are saved as WorkflowFileRef objects that can be
        accessed by downstream workflow nodes.

        Args:
            history: AgentHistory from BrowserUse agent.run()
            save_screenshots: Whether screenshot saving is enabled
            file_system: WorkflowFileSystem for S3 uploads
            workflow_run_id: Run ID for organizing files
            user_id: User ID for scoping files

        Returns:
            List of WorkflowFileRef dicts with file references
        """
        if not save_screenshots:
            return []

        if not file_system or not workflow_run_id or not user_id:
            logger.warning(
                "Screenshot saving requested but missing required context "
                f"(file_system={file_system is not None}, "
                f"run_id={workflow_run_id}, user_id={user_id})"
            )
            return []

        screenshots_result: List[Dict[str, Any]] = []

        try:
            # BrowserUse history.screenshots() returns list of base64 strings
            screenshots_b64 = history.screenshots()
            if not screenshots_b64:
                logger.debug("No screenshots captured during browser task")
                return []

            for idx, screenshot_b64 in enumerate(screenshots_b64):
                if not screenshot_b64:
                    continue

                filename = f"screenshot_{idx:03d}.png"
                try:
                    file_ref = await file_system.store_from_base64(
                        user_id=user_id,
                        run_id=workflow_run_id,
                        filename=filename,
                        base64_data=screenshot_b64,
                        mime_type="image/png",
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
    def profile_manager(self) -> BrowserProfileManager:
        """Get the profile manager for direct profile operations."""
        return self._profile_manager
