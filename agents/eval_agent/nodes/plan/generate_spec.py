"""
Generate AgentSpec and AlignmentState from dataset_examples and user request.
This is used in plan-only mode to provide structured output for user alignment.
"""
import json
from uuid import uuid4
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ConfigDict

from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from shared.schema import (
    AgentSpec,
    AlignmentState,
    AlignmentQuestion,
    TestCaseIntent,
)
from shared.llm import get_llm
from shared.config import config

logger = get_logger("eval_agent.plan.generate_spec")


def _normalize_raw_request(raw_request) -> str:
    """
    Ensure the raw request is a plain string.
    Handles cases where the request is stored as a list of message chunks.
    """
    if not raw_request:
        return ""
    if isinstance(raw_request, str):
        return raw_request
    if isinstance(raw_request, list):
        parts = []
        for chunk in raw_request:
            if isinstance(chunk, dict):
                if "text" in chunk:
                    parts.append(chunk["text"])
                elif "content" in chunk and isinstance(chunk["content"], str):
                    parts.append(chunk["content"])
                else:
                    parts.append(str(chunk))
            else:
                parts.append(str(chunk))
        return "\n".join(filter(None, parts))
    return str(raw_request)


class SpecGenerationOutput(BaseModel):
    """Structured output for agent spec and alignment questions."""
    model_config = ConfigDict(extra="forbid")
    
    agent_spec: AgentSpec = Field(..., description="Agent specification")
    alignment_questions: list[dict] = Field(..., description="Exactly 3 alignment questions with question_id, question, and context")


async def generate_agent_spec_and_alignment(state: EvalAgentState) -> dict:
    """
    Generate AgentSpec and exactly 3 alignment questions from dataset_examples and user request.
    
    This function:
    1. Analyzes the dataset_examples to extract test case intents
    2. Generates an AgentSpec summarizing the agent's understanding
    3. Generates exactly 3 alignment questions to clarify ambiguities
    """
    raw_request = _normalize_raw_request(state.context.user_context.raw_request)
    agent_name = state.context.agent_name
    dataset_examples = state.dataset_examples or []
    mcp_services = state.context.mcp_services or []
    
    logger.info(f"Generating agent spec and alignment questions for {agent_name}")
    
    # Convert dataset_examples to test case intents
    test_case_intents = []
    for example in dataset_examples:
        intent = TestCaseIntent(
            intent_id=example.example_id,
            description=f"Test: {example.input_message[:100]}...",
            expected_behavior=example.expected_output.expected_action,
            validation_criteria=[
                f"Verify {inst.service_name}: {', '.join(inst.instructions)}"
                for inst in example.expected_output.assert_final_state
            ],
            complexity="moderate",  # Could be enhanced to analyze actual complexity
            estimated_duration=None,
        )
        test_case_intents.append(intent)
    
    # Create prompt for LLM to generate spec and questions
    system_prompt = """You are an expert at analyzing AI agent requirements and generating clear specifications.

Your task is to:
1. Analyze the user's request and generated test cases to create an AgentSpec
2. Identify exactly 3 key ambiguities or areas that need clarification
3. Generate exactly 3 alignment questions (one per ambiguity)

CRITICAL RULES:
- Generate EXACTLY 3 questions (not 2, not 4, exactly 3)
- Each question should have:
  - question_id: A unique identifier (UUID format)
  - question: The actual question text
  - context: Why this question is being asked (what ambiguity it addresses)
- Questions should be focused on clarifying the agent's behavior, not implementation details
- Questions should help prevent misaligned expectations

The AgentSpec should include:
- agent_name: Name of the agent
- primary_goal: Main goal of the agent
- key_capabilities: List of key capabilities
- required_integrations: MCP services needed
- test_scenarios: List of test case intents
- assumptions: What you assume about the agent
- confidence_score: How confident you are (0.0 to 1.0)
"""

    user_prompt = f"""User Request:
{raw_request}

Agent Name: {agent_name}

MCP Services Available: {', '.join(mcp_services) if mcp_services else 'None specified'}

Generated Test Cases ({len(dataset_examples)} total):
{json.dumps([{
    "input": ex.input_message,
    "expected_action": ex.expected_output.expected_action,
    "assertions": [f"{inst.service_name}: {', '.join(inst.instructions)}" 
                   for inst in ex.expected_output.assert_final_state]
} for ex in dataset_examples[:5]], indent=2)}

Based on the user request and test cases above, generate:
1. A comprehensive AgentSpec that captures your understanding
2. Exactly 3 alignment questions to clarify any ambiguities

Focus on questions that will prevent wasted execution time due to misalignment.
"""

    # Use structured output to get spec and questions
    structured_llm = get_llm(
        reasoning_effort=config.eval_reasoning_effort,
    ).with_structured_output(SpecGenerationOutput, method="json_schema", strict=True)
    
    try:
        output: SpecGenerationOutput = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Convert questions to AlignmentQuestion objects
        alignment_questions = []
        for q_dict in output.alignment_questions[:3]:  # Ensure exactly 3
            question = AlignmentQuestion(
                question_id=q_dict.get("question_id", str(uuid4())),
                question=q_dict.get("question", ""),
                context=q_dict.get("context", ""),
            )
            alignment_questions.append(question)
        
        # Ensure we have exactly 3 questions
        while len(alignment_questions) < 3:
            logger.warning(f"Only {len(alignment_questions)} questions generated, adding placeholder")
            alignment_questions.append(AlignmentQuestion(
                question_id=str(uuid4()),
                question="Are there any specific edge cases or error scenarios you want the agent to handle?",
                context="Clarifying error handling requirements",
            ))
        
        # Create AlignmentState
        alignment_state = AlignmentState(
            questions=alignment_questions[:3],  # Exactly 3
            answers={},
            is_complete=False,
        )
        
        logger.info(f"Generated agent spec with {len(alignment_questions)} alignment questions")
        
        return {
            "agent_spec": output.agent_spec,
            "alignment_state": alignment_state,
        }
        
    except Exception as e:
        logger.error(f"Error generating spec: {e}", exc_info=True)
        # Fallback: create minimal spec and questions
        fallback_spec = AgentSpec(
            agent_name=agent_name,
            primary_goal=raw_request[:200] if raw_request else "Agent evaluation",
            key_capabilities=[],
            required_integrations=mcp_services,
            test_scenarios=test_case_intents,
            assumptions=[],
            confidence_score=0.5,
        )
        
        fallback_questions = [
            AlignmentQuestion(
                question_id=str(uuid4()),
                question="What is the primary goal of this agent?",
                context="Clarifying the main objective",
            ),
            AlignmentQuestion(
                question_id=str(uuid4()),
                question="What integrations or services should the agent use?",
                context="Clarifying required integrations",
            ),
            AlignmentQuestion(
                question_id=str(uuid4()),
                question="Are there any specific success criteria or edge cases to consider?",
                context="Clarifying evaluation criteria",
            ),
        ]
        
        return {
            "agent_spec": fallback_spec,
            "alignment_state": AlignmentState(
                questions=fallback_questions,
                answers={},
                is_complete=False,
            ),
        }

