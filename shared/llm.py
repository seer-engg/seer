"""Shared LLM utilities"""
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from shared.config import config

load_dotenv()


def get_llm(
    model: str = config.default_llm_model,
    temperature: float = 0.2,
    reasoning_effort: str = "minimal",
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """
    Get a configured LLM instance.
    
    Args:
        model: Model name
        temperature: Temperature setting
        api_key: Optional API key override
        
    Returns:
        Configured ChatOpenAI instance
    """
    if api_key is None:
        api_key = config.openai_api_key
    if api_key is None or api_key == "":
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Always use responses API for consistent output
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        use_responses_api=True,                
        reasoning={"effort": reasoning_effort},
    )


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

    # Always use responses API for consistent output
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        use_responses_api=False,
    )
