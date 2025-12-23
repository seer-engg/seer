"""Shared LLM utilities"""
from typing import Optional, Literal, Union
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv
from shared.config import config

load_dotenv()


def _detect_provider(model: str) -> Literal["openai", "anthropic"]:
    """Detect provider from model name."""
    if model.startswith(("gpt-", "o3-")):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    else:
        # Default to OpenAI for backward compatibility
        return "openai"


def get_llm(
    model: str = config.default_llm_model,
    temperature: float = 0.2,
    reasoning_effort: str = "minimal",
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """
    Get a configured LLM instance for OpenAI or Anthropic models.
    
    Args:
        model: Model name (e.g., "gpt-5.2", "claude-opus-4-5")
        temperature: Temperature setting
        reasoning_effort: Reasoning effort level (only used for models that support it)
        api_key: Optional API key override (provider-specific)
        
    Returns:
        Configured ChatOpenAI or ChatAnthropic instance
    """
    provider = _detect_provider(model)
    
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
        
        # Only include reasoning parameter for models that support it
        # o3 models support reasoning effort
        if model.startswith("o3-"):
            kwargs["reasoning"] = {"effort": reasoning_effort}
        # GPT-5.1 and GPT-5 (but not mini/nano variants) support reasoning effort
        elif model.startswith(("gpt-5.1", "gpt-5")) and not model.startswith("gpt-5-mini") and not model.startswith("gpt-5-nano"):
            kwargs["reasoning"] = {"effort": reasoning_effort}
        # Don't pass reasoning parameter for other models
        
        return ChatOpenAI(**kwargs)
    
    elif provider == "anthropic":
        if api_key is None:
            api_key = config.anthropic_api_key
        if api_key is None or api_key == "":
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        return ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=temperature,
        )
    
    else:
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
    final_output=""
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
