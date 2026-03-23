"""Shared LLM utilities"""
import logging
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from seer.config import config

load_dotenv()

logger = logging.getLogger(__name__)


def _detect_provider(model: str) -> Literal["openai", "anthropic", "openrouter"]:
    """Detect provider from model name."""
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("moonshot/", "moonshotai/", "openrouter/", "z-ai/", "qwen/", "google/", "mistralai/")):
        return "openrouter"
    # Default to OpenRouter for other models
    return "openrouter"


def get_llm(
    model: str = config.default_llm_model,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """
    Get a configured LLM instance for OpenRouter or Anthropic models.

    Args:
        model: Model name (e.g., "moonshotai/kimi-k2.5", "claude-opus-4-5")
        temperature: Temperature setting
        api_key: Optional API key override (provider-specific)

    Returns:
        Configured ChatOpenAI, ChatAnthropic, or OpenRouter instance
    """
    provider = _detect_provider(model)
    logger.info("🤖 Initializing LLM | Model: %s | Provider: %s | Temperature: %s", model, provider, temperature)

    if provider == "openai":
        if api_key is None:
            api_key = config.openai_api_key
        if api_key is None or api_key == "":
            raise ValueError("OPENAI_API_KEY not found in environment")

        # Build kwargs for ChatOpenAI
        kwargs = {
            "model": model,
            "api_key": api_key,
            "use_responses_api": True,
            "temperature": temperature,
        }

        return ChatOpenAI(**kwargs)

    if provider == "openrouter":
        if api_key is None:
            api_key = config.openrouter_api_key
        if api_key is None or api_key == "":
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        logger.info("🌐 Using OpenRouter API | Model: %s | Base URL: https://openrouter.ai/api/v1", model)
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            use_responses_api=False,
        )

    if provider == "anthropic":
        if api_key is None:
            api_key = config.anthropic_api_key
        if api_key is None or api_key == "":
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        return ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider for model: {model}")


async def get_agent_final_respone(result: dict) -> str:
    """
    Get the final response from the agent. response is in the format of the responses API

    Args:
        result: dict - The result from the agent invoked via responses API

    Returns:
        str - The final response from the agent.
    """
    output = result.get("messages", [])[-1].content
    final_output = ""
    if isinstance(output, str):
        final_output = output
    elif isinstance(output, list):
        for content_block in output:
            if content_block.get("type") == "text":
                final_output += content_block.get("text")
    return final_output


def get_llm_without_responses_api(
    model: str = config.default_llm_model,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """
    Get a configured LLM instance without responses API.

    Args:
        model: Model name
        temperature: Temperature setting
        api_key: Optional API key override
    Returns:
        Configured ChatOpenAI instance without responses API
    """
    if api_key is None:
        api_key = config.openai_api_key
    if api_key is None or api_key == "":
        raise ValueError("OPENAI_API_KEY not found in environment")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        use_responses_api=False,
        temperature=temperature,
    )
