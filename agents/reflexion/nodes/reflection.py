import traceback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from agents.reflexion.models import ReflexionState, Reflection
from shared.logger import get_logger
from shared.llm import get_llm
from agents.reflexion.pinecone_client import pinecone_add_memory
from datetime import datetime
logger = get_logger('reflexion_agent')


REFLECTION_PROMPT = """You are a Coding Reflection Agent in a reflexion system - a senior software architect who provides expert guidance on code improvements and best practices.

YOUR ROLE:
- Analyze why the Actor's code failed evaluation
- Suggest coding paradigms, patterns, and best practices to prevent the issues in the future
- Provide constructive, actionable feedback.
- Help the Actor learn and improve for future iterations

REFLECTION PROCESS:
1. **Review Requirements**: What was the user asking for?
2. **Examine Code**: What did the Actor write?
3. **Study Test Failures**: Which test cases failed and why?
4. **Identify Root Causes**: What coding mistakes or misconceptions led to failures?
5. **Suggest Paradigms**: What coding patterns, principles, or approaches would fix this?

CODING PARADIGMS & PATTERNS TO CONSIDER:
- **Design Patterns**: Factory, Strategy, Observer, Decorator, etc.
- **SOLID Principles**: Single Responsibility, Open/Closed, Liskov Substitution, etc.
- **DRY**: Don't Repeat Yourself
- **Error Handling**: Try-except, validation, defensive programming
- **Data Structures**: Choose appropriate structures (dict vs list, set vs list, etc.)
- **Algorithms**: Time/space complexity, optimization techniques
- **Testing**: Write testable code, dependency injection
- **Security**: Input validation, sanitization, authentication
- **Concurrency**: Thread safety, locks, async patterns
- **Functional vs OOP**: When to use each paradigm

FEEDBACK GUIDELINES:
- Be specific and actionable with code-level suggestions
- Focus on teaching coding principles, not just fixes
- Prioritize critical bugs, then edge cases, then quality improvements
- Explain WHY a paradigm/pattern solves the problem
- Think about what the Actor should LEARN for future code

YOUR FEEDBACK FORMAT:
Return a  Reflection with following key information:
- coding_context : description of the coding task that failed the evaluation and this reflection is applicable for, Only include the discription of the coding task, not errors or issues or feedback or anything else.
- reflection : key reflection points to be considered to avoid the issues in the future

Your feedback will be stored in the Actor's memory and help improve ALL future code!
Focus on teaching reusable lessons and paradigms, not just quick fixes.
"""


def reflection_node(state: ReflexionState, config: RunnableConfig) -> dict:
    """
    Reflection agent provides feedback when evaluation fails.
    Stores actionable suggestions in persistent memory for the Actor.
    """
    logger.info("Reflection node - Generating feedback")
    
    # Extract user query and actor response from message history
    user_message = ""
    actor_response = ""
    
    # Get the last human message (user query)
    for msg in reversed(state.trajectory):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    # Get the last AI message (actor response)
    for msg in reversed(state.trajectory):
        if isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
            actor_response = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    # Build reflection prompt
    reflection_prompt = f"""
User Query:
{user_message}

Actor's Response:
{actor_response}

Evaluator's Verdict:
- Passed: {state.evaluator_verdict.passed}
- Score: {state.evaluator_verdict.score}
- Reasoning: {state.evaluator_verdict.reasoning}
- Issues: {', '.join(state.evaluator_verdict.issues)}

Analyze why the response failed and provide constructive feedback to help the Actor improve.
Focus on specific, actionable suggestions for the next attempt.
"""
    
    prompt_messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=reflection_prompt)
    ]
    
    # Get structured reflection
    llm = get_llm(temperature=0.3).with_structured_output(Reflection)
    reflection: Reflection = llm.invoke(prompt_messages)
    
    logger.info(f"Reflection generated - {reflection.coding_context} - {reflection.reflection}")
    
    # Store reflection in Pinecone as a memory for this user (memory_key)
    try:
        user_id = state.memory_key
        # Store reflection with context (simplified API)
        metadata = {
            "coding_context": reflection.coding_context,
            "reflection": reflection.reflection,
            "timestamp": datetime.now().isoformat(),
        }
        result = pinecone_add_memory(context=reflection.coding_context, user_id=user_id, metadata=metadata)
        if result.get("success"):
            logger.info(f"Stored reflection in Pinecone successfully: {result.get('memory_id')}")
        else:
            logger.error(f"Failed to store reflection in Pinecone: {result.get('error')}")
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Failed to store reflection in Pinecone: {e}")
    
    # Don't add reflection to message history - it's stored in persistent memory
    # User only sees actor's responses, not internal reflections
    return {
        "current_attempt": state.current_attempt + 1
    }

