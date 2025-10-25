import traceback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from agents.reflexion.models import ReflexionState, Reflection
from shared.logger import get_logger
from shared.llm import get_llm
from agents.reflexion.mem0_client import mem0_add_memory

logger = get_logger('reflexion_agent')

MEMORY_NAMESPACE = ("reflexion", "feedback")


REFLECTION_PROMPT = """You are a Coding Reflection Agent in a reflexion system - a senior software architect who provides expert guidance on code improvements and best practices.

YOUR ROLE:
- Analyze why the Actor's code failed evaluation
- Suggest coding paradigms, patterns, and best practices to fix issues
- Provide constructive, actionable feedback with code examples
- Help the Actor learn and improve for future iterations
- Generate insights that will be stored in the Actor's persistent memory

REFLECTION PROCESS:
1. **Review Requirements**: What was the user asking for?
2. **Examine Code**: What did the Actor write?
3. **Study Test Failures**: Which test cases failed and why?
4. **Identify Root Causes**: What coding mistakes or misconceptions led to failures?
5. **Suggest Paradigms**: What coding patterns, principles, or approaches would fix this?
6. **Provide Examples**: Give concrete code snippets or approaches

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
- Provide concrete code examples or pseudocode
- Explain WHY a paradigm/pattern solves the problem
- Think about what the Actor should LEARN for future code

YOUR FEEDBACK FORMAT:
Return a structured Reflection with:
- **key_issues**: List of main coding problems (bugs, missing edge cases, design flaws)
- **suggestions**: Specific coding improvements with paradigm/pattern recommendations
  * Example: "Use try-except with specific exceptions instead of bare except"
  * Example: "Apply Strategy pattern to handle multiple algorithms"
  * Example: "Add input validation at function entry point"
- **focus_areas**: What to prioritize in next attempt
  * Example: "Edge case handling", "Error recovery", "Input validation"
- **examples**: Concrete code snippets or pseudocode showing the fix
  * Example: "```python\nif not items: return []\n```"

REFLECTION STRATEGIES:
- If test failed: Explain what input/scenario broke it and how to fix
- If edge case missed: Show how to handle it with code example
- If design flaw: Suggest better architecture or pattern
- If performance issue: Recommend algorithm or data structure change
- If readability issue: Show cleaner code structure
- If security issue: Explain vulnerability and secure alternative

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
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    # Get the last AI message (actor response)
    for msg in reversed(state.messages):
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
    
    logger.info(f"Reflection generated - {len(reflection.key_issues)} issues, {len(reflection.suggestions)} suggestions")
    
    # Store reflection in Mem0 as a memory for this user (memory_key)
    try:
        user_id = state.memory_key
        # Represent reflection as assistant message text for semantic recall
        reflection_text_parts = []
        if reflection.key_issues:
            reflection_text_parts.append("Key Issues: " + ", ".join(reflection.key_issues))
        if reflection.suggestions:
            reflection_text_parts.append("Suggestions: " + ", ".join(reflection.suggestions))
        if reflection.focus_areas:
            reflection_text_parts.append("Focus Areas: " + ", ".join(reflection.focus_areas))
        if reflection.examples:
            reflection_text_parts.append("Examples: " + "; ".join(reflection.examples))
        reflection_text = "\n".join(reflection_text_parts) or "Reflection feedback"

        messages_payload = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reflection_text},
        ]

        mem0_add_memory(messages=messages_payload, user_id=user_id)
        logger.info("Stored reflection in Mem0 successfully")
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Failed to store reflection in Mem0: {e}")
    
    # Don't add reflection to message history - it's stored in persistent memory
    # User only sees actor's responses, not internal reflections
    return {
        "current_attempt": state.current_attempt + 1
    }

