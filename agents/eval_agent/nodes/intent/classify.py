"""Classify user intent from their message."""
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from agents.eval_agent.models import EvalAgentState
from shared.schema import UserIntent
from shared.logger import get_logger

logger = get_logger("eval_agent.intent")


async def classify_user_intent(state: EvalAgentState) -> dict:
    """Classify user's intent from their latest message."""
    # Get last human message
    last_human = None
    for msg in reversed(state.messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            last_human = msg
            break
    
    if not last_human:
        # Default to evaluation_request if no message
        logger.info("No user message found, defaulting to evaluation_request")
        return {
            "user_intent": UserIntent(
                intent_type="evaluation_request",
                confidence=1.0,
                reasoning="No user message found, defaulting to evaluation"
            )
        }
    
    # Use LLM to classify intent
    classifier = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.0
    ).with_structured_output(UserIntent)
    
    prompt = """Classify the user's intent from their message.

INTENT TYPES:
- "informational": User is asking for information, capabilities, or help (e.g., "What can you do?", "How does this work?", "List features", "Help me understand")
- "evaluation_request": User wants to evaluate an agent, create test plans, or generate agent specs (e.g., "Evaluate my agent", "Create test plan", "Generate agent spec", "Test my agent")

USER MESSAGE:
{message}

Classify the intent and provide reasoning.""".format(message=last_human.content)
    
    intent: UserIntent = await classifier.ainvoke(prompt)
    logger.info(f"Intent classified: {intent.intent_type} (confidence: {intent.confidence:.2f}) - {intent.reasoning}")
    
    return {
        "user_intent": intent
    }

