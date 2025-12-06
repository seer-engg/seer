"""
Simple retry middleware for OpenAI API errors with exponential backoff.
No sanitization - just retry on any error.
"""
import logging
import asyncio
import time
from openai import OpenAIError

logger = logging.getLogger(__name__)


def wrap_model_with_retry(model, max_retries: int = 3, base_delay: float = 2.0):
    """
    Wrap a ChatOpenAI model with simple retry logic and exponential backoff.
    
    Usage:
        model = ChatOpenAI(model="gpt-5-mini")
        model = wrap_model_with_retry(model, max_retries=3)
        agent = create_agent(model, tools)
    
    Args:
        model: ChatOpenAI model instance
        max_retries: Maximum number of retry attempts (default: 3, total attempts = 4)
        base_delay: Base delay in seconds for exponential backoff (default: 2.0)
                    Delays: 2s, 4s, 8s, 16s
    
    Returns:
        Wrapped model using RunnableLambda
    """
    async def retry_ainvoke(input_data, config=None, **kwargs):
        """Wrapper that retries on any OpenAI error with exponential backoff."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await model.ainvoke(input_data, config=config, **kwargs)
                if attempt > 0:
                    logger.info(f"✅ Retry succeeded on attempt {attempt + 1}")
                return result
                
            except OpenAIError as e:
                last_error = e
                error_message = str(e)
                
                if attempt < max_retries:
                    # Exponential backoff: delay = base_delay * (2 ^ attempt)
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ OpenAI error (attempt {attempt + 1}/{max_retries + 1}): {error_message[:200]}")
                    logger.info(f"🔄 Retrying after {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ Max retries reached after {max_retries + 1} attempts.")
                    raise
    
    def retry_invoke(input_data, config=None, **kwargs):
        """Sync wrapper that retries on any OpenAI error."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = model.invoke(input_data, config=config, **kwargs)
                if attempt > 0:
                    logger.info(f"✅ Retry succeeded on attempt {attempt + 1}")
                return result
                
            except OpenAIError as e:
                last_error = e
                error_message = str(e)
                
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ OpenAI error (attempt {attempt + 1}/{max_retries + 1}): {error_message[:200]}")
                    logger.info(f"🔄 Retrying after {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ Max retries reached after {max_retries + 1} attempts.")
                    raise
        
        if last_error:
            raise last_error
    
    # Use composition: wrap the model's invoke methods
    # Since we can't modify Pydantic models, we'll create a wrapper class
    class RetryModel:
        def __init__(self, original_model):
            self._model = original_model
        
        async def ainvoke(self, input_data, config=None, **kwargs):
            return await retry_ainvoke(input_data, config=config, **kwargs)
        
        def invoke(self, input_data, config=None, **kwargs):
            return retry_invoke(input_data, config=config, **kwargs)
        
        def __getattr__(self, name):
            # Delegate all other attributes/methods to the original model
            return getattr(self._model, name)
    
    return RetryModel(model)
