import traceback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from agents.reflexion.models import ReflexionState
from shared.logger import get_logger
from shared.llm import get_llm
from agents.reflexion.pinecone_client import pinecone_search_memories

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

"""


def actor_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Actor generates a response to the user's message.
    Uses reflection memory from persistent store to improve responses.
    """
    logger.info(f"Actor node - Attempt {state.current_attempt}/{state.max_attempts}")
    
    # Extract user's latest message from message history
    user_message = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    if not user_message:
        logger.warning("No user message found in history")
        user_message = "Please provide a response."
    
    # Retrieve reflection memory from Pinecone using semantic search
    try:
        # Use user message as the semantic query to recall relevant reflections
        pinecone_results = pinecone_search_memories(query=user_message, user_id=state.memory_key)
        reflection_memory = []
        for item in pinecone_results:
            # Pinecone results are normalized dicts with 'content' field
            if isinstance(item, dict):
                text = item.get('content', '')
                if text:
                    reflection_memory.append(text)
        logger.info(f"Retrieved {len(reflection_memory)} reflections from Pinecone for user_id={state.memory_key}")
    except Exception as e:
        logger.error(f"Error retrieving memory from Pinecone: {e}")
        logger.error(traceback.format_exc())
        reflection_memory = []
    
    # Build memory context from reflection feedback
    memory_context = ""
    if reflection_memory:
        memory_context = "\n\n=== MEMORY_CONTEXT (Reflection Feedback from Previous Interactions) ===\n"
        for idx, reflection_text in enumerate(reflection_memory, 1):
            memory_context += f"\n--- Reflection {idx} ---\n"
            memory_context += f"{reflection_text}\n"
        memory_context += "\n=== END MEMORY_CONTEXT ===\n"
    
    # Build prompt with memory
    prompt_messages = [
        SystemMessage(content=ACTOR_PROMPT),
    ]
    
    if memory_context:
        prompt_messages.append(SystemMessage(content=memory_context))
    
    prompt_messages.append(HumanMessage(content=f"User Query: {user_message}\n\nGenerate your response now, incorporating any feedback from memory."))
    
    # Generate response
    llm = get_llm(temperature=0)
    response = llm.invoke(prompt_messages)
    actor_response = response.content
    
    logger.info(f"Actor generated response (length: {len(actor_response)})")
    
    return {
        "messages": [AIMessage(content=actor_response)]
    }
