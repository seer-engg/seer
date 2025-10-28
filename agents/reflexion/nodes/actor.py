from langchain_core.runnables import RunnableConfig
from agents.reflexion.models import ReflexionState
from shared.logger import get_logger
from shared.llm import get_llm
from agents.reflexion.pinecone_client import pinecone_search_memories
from langchain.agents import create_agent

logger = get_logger('reflexion_agent')

ACTOR_PROMPT = """You are a Coding Agent in a reflexion system - an expert software engineer who writes high-quality, production-ready code.

YOUR ROLE:
- Write clean, efficient, and well-documented code for user requests
- Learn from reflection feedback stored in your memory
- Improve your code with each iteration based on test failures and feedback


CODE GENERATION GUIDELINES:
- Read the user's coding request carefully
- Write production-quality code with proper error handling
- Include docstrings and inline comments for clarity
- Follow best practices and design patterns
- Consider edge cases and error scenarios
- Write testable, modular code
- If you have reflection feedback, incorporate those coding paradigms

CODE QUALITY STANDARDS:
- Clean, readable code following PEP 8 (for Python) or language-specific standards
- Proper naming conventions
- DRY (Don't Repeat Yourself) principle
- SOLID principles where applicable
- Error handling and input validation
- Efficient algorithms and data structures

AVAILABLE TOOLS:
1. **get_reflection_memory(code_context)**: Get relevant reflection memories for the given code context

#note 
 - always use the get_reflection_memory tool  first to get the reflection memories that are relevant to the code context before writing the code

"""

from langchain.tools import tool

def actor_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Actor generates a response to the user's message.
    Uses reflection memory from persistent store to improve responses.
    """
    logger.info(f"Actor node - Attempt {state.current_attempt}/{state.max_attempts}")

    @tool
    def get_reflection_memory(code_context: str) -> str:
        """
        Get relevant reflection memories for the given code context
        """
        pinecone_results = pinecone_search_memories(query=code_context, user_id=state.memory_key)
        reflection_memory = []
        for item in pinecone_results:
            reflection_memory.append(item.get('metadata', {}).get('reflection', ''))
        return reflection_memory

    actor_agent = create_agent(
        model=get_llm(temperature=0),
        tools=[get_reflection_memory],
        system_prompt=ACTOR_PROMPT,
    )

    actor_result = actor_agent.invoke({"messages": state.messages})
    logger.info(f"Actor generated response: {actor_result.keys()}")
    
    return {
        "trajectory": actor_result.get('messages', [])
    }
