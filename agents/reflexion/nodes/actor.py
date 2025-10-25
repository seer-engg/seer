import traceback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from agents.reflexion.models import ReflexionState
from shared.logger import get_logger
from shared.llm import get_llm
from agents.reflexion.mem0_client import mem0_search_memories
from agents.reflexion.memory_artifacts import parse_artifact_from_content

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
    
    # Retrieve artifacts from Mem0 with simple query expansion and reranking
    def _expand_queries(q: str) -> list[str]:
        base = q.strip()
        return [
            base,
            f"python {base}",
            f"edge cases {base}",
            f"errors {base}",
            f"performance {base}",
        ]

    def _guess_tags(q: str) -> set[str]:
        tokens = set(t.lower().strip(",.:;()[]{}") for t in q.split())
        guesses = set()
        for t in tokens:
            if t in {"python", "py"}:
                guesses.add("python")
            if t in {"edge", "edges", "edge-cases", "cases"}:
                guesses.add("edge")
            if t in {"error", "errors", "exception"}:
                guesses.add("error")
            if t in {"performance", "perf", "optimize"}:
                guesses.add("performance")
            if t in {"type", "types", "typing"}:
                guesses.add("type")
            if t in {"test", "tests", "testing"}:
                guesses.add("testing")
        return guesses

    lessons: list[dict] = []
    try:
        seen_rules: set[str] = set()
        agg: list[dict] = []
        for q in _expand_queries(user_message):
            try:
                results = mem0_search_memories(query=q, user_id=state.memory_key)
            except Exception:
                results = []
            for item in results:
                text = None
                if isinstance(item, dict):
                    text = item.get('content') or item.get('memory') or item.get('text')
                    if text is None:
                        data = item.get('data')
                        if isinstance(data, dict):
                            text = data.get('content') or data.get('text')
                if not text:
                    text = str(item)
                parsed = parse_artifact_from_content(text)
                rule = parsed.get('rule')
                if not rule or rule in seen_rules:
                    continue
                seen_rules.add(rule)
                agg.append(parsed)

        # Rerank by tag overlap vs guessed tags
        guessed = _guess_tags(user_message)
        def _score(a: dict) -> int:
            tags = set((a.get('tags') or []))
            return len(tags & guessed)
        agg.sort(key=_score, reverse=True)

        lessons = agg[:8]
        logger.info(f"Retrieved {len(lessons)} lessons from Mem0 after rerank")
    except Exception as e:
        logger.error(f"Error retrieving lessons from Mem0: {e}")
        logger.error(traceback.format_exc())
        lessons = []

    # Build memory context as concise lessons
    memory_context = ""
    if lessons:
        memory_context = "\n\n=== LESSONS (Prior Artifacts) ===\n"
        for i, art in enumerate(lessons, 1):
            rule = art.get('rule') or ''
            tags = art.get('tags') or []
            tag_str = ", ".join(tags)
            memory_context += f"- {rule}"
            if tag_str:
                memory_context += f" [tags: {tag_str}]"
            memory_context += "\n"
        memory_context += "=== END LESSONS ===\n"
    
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
