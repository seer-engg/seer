from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from agents.reflexion.models import ReflexionState
from shared.logger import get_logger
from shared.llm import get_llm

logger = get_logger('reflexion_agent')

MEMORY_NAMESPACE = ("reflexion", "feedback")


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
    current_attempt = state.get("current_attempt", 1)
    max_attempts = state.get("max_attempts", 3)
    memory_key = state.get("memory_key", "default")
    messages = state.get("messages", [])
    
    logger.info(f"Actor node - Attempt {current_attempt}/{max_attempts}")
    
    # Extract user's latest message from message history
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    if not user_message:
        logger.warning("No user message found in history")
        user_message = "Please provide a response."
    
    # Retrieve reflection memory from persistent store
    try:
        store = config.get("configurable", {}).get("store")
        if store:
            items = list(store.search(MEMORY_NAMESPACE, filter={"memory_key": memory_key}))
            reflection_memory = [item.value for item in items if item.value]
            logger.info(f"Retrieved {len(reflection_memory)} reflections from persistent memory")
        else:
            reflection_memory = []
            logger.warning("No store available in config")
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")
        reflection_memory = []
    
    # Build memory context from reflection feedback
    memory_context = ""
    if reflection_memory:
        memory_context = "\n\n=== MEMORY_CONTEXT (Reflection Feedback from Previous Interactions) ===\n"
        for idx, reflection in enumerate(reflection_memory, 1):
            memory_context += f"\n--- Reflection {idx} ---\n"
            if isinstance(reflection, dict):
                memory_context += f"Key Issues: {', '.join(reflection.get('key_issues', []))}\n"
                memory_context += f"Suggestions: {', '.join(reflection.get('suggestions', []))}\n"
                memory_context += f"Focus Areas: {', '.join(reflection.get('focus_areas', []))}\n"
                examples = reflection.get('examples', [])
                if examples:
                    memory_context += f"Examples: {', '.join(examples)}\n"
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
