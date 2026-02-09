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
from typing import Any, Dict, Optional
from uuid import UUID

from browser_use import Agent, Browser, ChatOpenAI
# from langchain_core.language_models import BaseChatModel
from playwright.async_api import async_playwright

from seer.database import User
from seer.services.browser.profile_manager import BrowserProfileManager
from seer.config import config

logger = logging.getLogger(__name__)


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

    async def execute_task(
        self,
        user: Optional[User],
        task: str,
        inputs: Dict[str, Any],
        *,
        browser_profile_id: Optional[str] = None,
        max_steps: int = 25,
        timeout_seconds: int = 300,
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

        Returns:
            Result dict with success, result, extracted_data, final_url, screenshots
        """
        logger.info(f"Executing browser task: {task[:100]}...")

        # Load profile session if provided
        storage_state = None
        if browser_profile_id and user:
            try:
                storage_state = await self._profile_manager.get_session_state(
                    user, UUID(browser_profile_id)
                )
                if storage_state:
                    logger.info(f"Loaded session from profile {browser_profile_id}")
            except Exception as e:
                logger.warning(f"Failed to load profile {browser_profile_id}: {e}")

        # Execute with Playwright browser
        async with async_playwright() as p:
            # Launch headless browser for workflow execution
            browser = await p.chromium.launch(headless=True)

            try:
                context = await browser.new_context(storage_state=storage_state)
                page = await context.new_page()

                # Get LLM for the BrowserUse agent
                llm = self._get_agent_llm()

                # Enhance task with inputs context
                enhanced_task = self._enhance_task(task, inputs)

                # Create BrowserUse Browser wrapper and Agent
                browser_use_browser = Browser()
                agent = Agent(
                    task=enhanced_task,
                    llm=llm,
                    browser=browser_use_browser,
                )

                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.run(max_steps=max_steps),
                    timeout=timeout_seconds
                )

                # Get final URL from the Playwright page
                final_url = page.url if page else None

                logger.info("Browser task completed successfully")
                return {
                    "success": True,
                    "result": str(result) if result else "",
                    "extracted_data": self._extract_data(result),
                    "final_url": final_url,
                    "screenshots": [],
                }

            except asyncio.TimeoutError:
                logger.warning(f"Browser task timed out after {timeout_seconds}s")
                return {
                    "success": False,
                    "result": f"Task timed out after {timeout_seconds} seconds",
                    "extracted_data": {},
                    "final_url": None,
                    "screenshots": [],
                }
            except Exception as e:
                logger.error(f"Browser task failed: {e}")
                return {
                    "success": False,
                    "result": f"Task failed: {str(e)}",
                    "extracted_data": {},
                    "final_url": None,
                    "screenshots": [],
                }
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

        model = 'kimi-k2.5'
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

    @property
    def profile_manager(self) -> BrowserProfileManager:
        """Get the profile manager for direct profile operations."""
        return self._profile_manager
