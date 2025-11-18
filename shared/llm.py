"""Shared LLM utilities"""
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from shared.config import DEFAULT_LLM_MODEL
from dotenv import load_dotenv

load_dotenv()


def get_llm(
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
    reasoning_effort: str = "medium",
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
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


async def convert_response_v1_output_to_message_string(result: dict) -> str:
    message_string = ""
    last_message = result.get("messages", [])[-1].content
    if isinstance(last_message, list):
        for message in last_message:
            if message.get("type") == "text":
                message_string += message.get("text")
    return message_string

