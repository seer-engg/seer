"""Shared LLM utilities"""

from langchain_openai import ChatOpenAI
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.3,
    api_key: Optional[str] = None
) -> ChatOpenAI:
    """
    Get a configured LLM instance.
    
    Args:
        model: Model name (uses gpt-4o-mini if not specified)
        temperature: Temperature setting
        api_key: Optional API key override
        
    Returns:
        Configured ChatOpenAI instance
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
    
    return ChatOpenAI(
        model=model or "gpt-4o-mini",
        temperature=temperature,
        api_key=api_key
    )

