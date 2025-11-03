"""Shared LLM utilities"""
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None
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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
    
    model_name = model or "gpt-4.1-nano-2025-04-14"

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key
    )
