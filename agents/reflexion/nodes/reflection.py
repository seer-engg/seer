from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from agents.reflexion.models import ReflexionState, Reflection
from shared.logger import get_logger
from shared.llm import get_llm

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
    current_attempt = state.get("current_attempt", 1)
    memory_key = state.get("memory_key", "default")
    messages = state.get("messages", [])
    verdict_dict = state.get("evaluator_verdict", {})
    
    logger.info("Reflection node - Generating feedback")
    
    # Extract user query and actor response from message history
    user_message = ""
    actor_response = ""
    
    # Get the last human message (user query)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
            user_message = msg.content if hasattr(msg, 'content') else str(msg)
            break
    
    # Get the last AI message (actor response)
    for msg in reversed(messages):
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
- Passed: {verdict_dict.get('passed', False)}
- Score: {verdict_dict.get('score', 0.0)}
- Reasoning: {verdict_dict.get('reasoning', '')}
- Issues: {', '.join(verdict_dict.get('issues', []))}

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
    
    # Store reflection in persistent memory
    try:
        store = config.get("configurable", {}).get("store")
        if store:
            import time
            reflection_id = f"{memory_key}_{int(time.time() * 1000)}"
            reflection_data = reflection.model_dump()
            reflection_data["memory_key"] = memory_key
            reflection_data["attempt"] = current_attempt
            
            store.put(
                MEMORY_NAMESPACE,
                reflection_id,
                reflection_data
            )
            logger.info(f"Stored reflection in persistent memory: {reflection_id}")
        else:
            logger.warning("No store available, reflection not persisted")
    except Exception as e:
        logger.error(f"Failed to store reflection: {e}")
    
    # Increment attempt counter
    new_attempt = current_attempt + 1
    
    # Don't add reflection to message history - it's stored in persistent memory
    # User only sees actor's responses, not internal reflections
    return {
        "current_attempt": new_attempt
    }

