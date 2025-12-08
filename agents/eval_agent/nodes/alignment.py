"""
Alignment node for handling user responses to alignment questions.
Processes partial answers and refines the agent spec accordingly.
"""
import json
from langchain_core.messages import AIMessage, HumanMessage
from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from shared.llm import get_llm
from shared.config import config
from shared.schema import AgentSpec

logger = get_logger("eval_agent.alignment")


async def alignment_node(state: EvalAgentState) -> dict:
    """
    Present questions to user, collect responses, refine plan.
    
    Behavior:
    - If questions exist but no answers: return questions for user
    - If answers received (partial or complete): refine plan based on answers
    - Support partial answers (user can skip some questions)
    """
    alignment_state = state.alignment_state
    
    if not alignment_state:
        logger.warning("No alignment_state found, skipping alignment")
        return {}
    
    # Check if user has provided answers in the latest message
    latest_message = state.messages[-1] if state.messages else None
    answers_provided = False
    user_answers = {}
    
    if latest_message and isinstance(latest_message, HumanMessage):
        content = latest_message.content
        # Try to parse alignment answers from message
        # Format: JSON with alignment_answers key, or structured text
        try:
            # Try parsing as JSON first
            if content.strip().startswith("{"):
                parsed = json.loads(content)
                if "alignment_answers" in parsed:
                    user_answers = parsed["alignment_answers"]
                    answers_provided = True
            # Check for structured text format
            elif "alignment_answers" in content.lower() or "question_id" in content.lower():
                # Try to extract answers from text
                # This is a simple heuristic - could be enhanced
                for question in alignment_state.questions:
                    if question.question_id in content or question.question.lower()[:20] in content.lower():
                        # Extract answer (simple heuristic)
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if question.question_id in line or question.question.lower()[:20] in line.lower():
                                # Next line might be the answer
                                if i + 1 < len(lines):
                                    answer = lines[i + 1].strip()
                                    if answer and not answer.startswith("Question"):
                                        user_answers[question.question_id] = answer
                                        answers_provided = True
                                        break
        except Exception as e:
            logger.debug(f"Could not parse answers from message: {e}")
    
    # If no answers yet, return questions for user
    if not answers_provided or not user_answers:
        logger.info("No answers provided yet, returning questions for user")
        questions_text = format_questions_for_user(alignment_state.questions)
        return {
            "messages": [AIMessage(content=questions_text)]
        }
    
    # Update alignment_state with answers
    alignment_state.answers.update(user_answers)
    
    # Mark questions as answered
    for question in alignment_state.questions:
        if question.question_id in user_answers:
            question.answer = user_answers[question.question_id]
    
    # Refine agent spec based on answers
    refined_spec = await refine_agent_spec(
        state.agent_spec,
        alignment_state.answers,
        alignment_state.questions
    )
    
    # Mark alignment as complete (even if partial answers)
    alignment_state.is_complete = True
    
    logger.info(f"Refined agent spec based on {len(user_answers)} answers")
    
    return {
        "agent_spec": refined_spec,
        "alignment_state": alignment_state,
        "messages": [AIMessage(content="✅ Plan refined based on your answers. The agent specification has been updated.")]
    }


def format_questions_for_user(questions: list) -> str:
    """Format alignment questions for display to user."""
    lines = [
        "📋 **Alignment Questions**",
        "",
        "To ensure we're aligned, please answer the following questions:",
        "",
    ]
    
    for i, question in enumerate(questions, 1):
        lines.append(f"**Question {i}:** {question.question}")
        lines.append(f"*Context: {question.context}*")
        lines.append(f"*Question ID: {question.question_id}*")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("You can answer by replying with:")
    lines.append("```json")
    lines.append('{"alignment_answers": {')
    for i, question in enumerate(questions):
        lines.append(f'  "{question.question_id}": "your answer here"{"," if i < len(questions) - 1 else ""}')
    lines.append("}}")
    lines.append("```")
    lines.append("")
    lines.append("Or simply reply with your answers in natural language.")
    lines.append("You can skip questions you don't want to answer.")
    
    return "\n".join(lines)


async def refine_agent_spec(
    agent_spec: AgentSpec,
    answers: dict[str, str],
    questions: list
) -> AgentSpec:
    """
    Refine agent spec based on user answers to alignment questions.
    
    Uses LLM to incorporate answers into the spec.
    """
    if not answers:
        logger.info("No answers provided, returning original spec")
        return agent_spec
    
    # Build prompt for refinement
    system_prompt = """You are an expert at refining agent specifications based on user feedback.

Your task is to update the AgentSpec based on the user's answers to alignment questions.
Incorporate the answers into the spec, updating assumptions, capabilities, and goals as needed.
"""

    user_prompt = f"""Original Agent Spec:
- Name: {agent_spec.agent_name}
- Primary Goal: {agent_spec.primary_goal}
- Key Capabilities: {', '.join(agent_spec.key_capabilities)}
- Required Integrations: {', '.join(agent_spec.required_integrations)}
- Assumptions: {', '.join(agent_spec.assumptions)}
- Confidence Score: {agent_spec.confidence_score}

User Answers:
{json.dumps([
    {
        "question": q.question,
        "answer": answers.get(q.question_id, "Not answered")
    }
    for q in questions if q.question_id in answers
], indent=2)}

Please refine the AgentSpec based on these answers. Update:
1. Primary goal if the answer clarifies the objective
2. Key capabilities if new capabilities are mentioned
3. Assumptions if answers clarify or correct assumptions
4. Confidence score (increase if answers clarify ambiguities)

Return the refined spec in the same format as the original.
"""

    llm = get_llm(reasoning_effort=config.eval_reasoning_effort)
    structured_llm = llm.with_structured_output(AgentSpec, method="json_schema", strict=True)
    
    try:
        refined_spec: AgentSpec = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Preserve test_scenarios from original (don't regenerate)
        refined_spec.test_scenarios = agent_spec.test_scenarios
        
        logger.info("Successfully refined agent spec")
        return refined_spec
        
    except Exception as e:
        logger.error(f"Error refining spec: {e}", exc_info=True)
        # Return original spec if refinement fails
        return agent_spec

