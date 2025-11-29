"""Shared LLM utilities"""
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from shared.config import config

load_dotenv()


def get_llm(
    model: str = config.default_llm_model,
    temperature: float = 0.2,
    reasoning_effort: str = "medium",
    api_key: Optional[str] = config.openai_api_key,
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
    if api_key is None or api_key == "":
        raise ValueError("OPENAI_API_KEY not found in environment")

    if 'codex' in model:
        return ChatOpenAI(
            model="gpt-5-codex",
            use_responses_api=True,            
            output_version="responses/v1",     
            reasoning={"effort": reasoning_effort},
        )
    else:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )

