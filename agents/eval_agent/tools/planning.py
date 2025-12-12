"""
Planning phase tools for eval agent Supervisor pattern.
"""
import json
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage

from shared.schema import UserIntent, AgentContext
from shared.logger import get_logger
from shared.config import config

logger = get_logger("eval_agent.tools.planning")


@tool
async def classify_user_intent(runtime: ToolRuntime = None) -> str:
    """
    Classify the user's intent based on their latest message.
    
    Determines if the user is asking for information or requesting an evaluation/agent spec generation.
    
    Returns:
        JSON string with intent_type ("informational" or "evaluation_request"), confidence, and reasoning.
    """
    # Get latest human message from runtime state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"intent_type": "evaluation_request", "confidence": 0.0, "reasoning": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    messages = state.get("messages", [])
    
    # Find last human message
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human" or (isinstance(msg, dict) and msg.get("type") == "human"):
            last_human_message = msg
            break
    
    if not last_human_message:
        logger.warning("No human message found to classify intent. Defaulting to evaluation_request.")
        return json.dumps({"intent_type": "evaluation_request", "confidence": 0.0, "reasoning": "No human message."})
    
    content = last_human_message.content if hasattr(last_human_message, "content") else last_human_message.get("content", "")
    
    instruction = (
        "Classify the user's intent based on their message. "
        "Is the user asking for information or requesting an evaluation/agent spec generation?\n"
        "Respond with a JSON object containing 'intent_type' (either 'informational' or 'evaluation_request'), "
        "'confidence' (a float between 0.0 and 1.0), and 'reasoning' (a brief explanation).\n\n"
        "Examples:\n"
        "- User: 'What can you do?' -> {'intent_type': 'informational', 'confidence': 0.9, 'reasoning': 'User is asking about capabilities.'}\n"
        "- User: 'How does this work?' -> {'intent_type': 'informational', 'confidence': 0.8, 'reasoning': 'User is asking for explanation.'}\n"
        "- User: 'Evaluate my agent at https://github.com/owner/repo' -> {'intent_type': 'evaluation_request', 'confidence': 0.95, 'reasoning': 'User explicitly requests evaluation.'}\n"
        "- User: 'Generate an agent spec for a GitHub PR reviewer' -> {'intent_type': 'evaluation_request', 'confidence': 0.9, 'reasoning': 'User requests agent spec generation.'}\n"
    )
    
    # Use model without reasoning_effort for compatibility
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0).with_structured_output(UserIntent)
    try:
        user_intent: UserIntent = await llm.ainvoke(f"{instruction}\n\nUSER:\n{content}")
        logger.info(f"Classified user intent: {user_intent.intent_type} (Confidence: {user_intent.confidence:.2f})")
        
        # Store in state via return (will be handled by supervisor node)
        intent_dict = user_intent.model_dump()
        return json.dumps(intent_dict)
    except Exception as e:
        logger.error(f"Error classifying user intent: {e}. Defaulting to evaluation_request.")
        return json.dumps({"intent_type": "evaluation_request", "confidence": 0.0, "reasoning": f"LLM classification failed: {e}"})


@tool
async def extract_agent_config(runtime: ToolRuntime = None) -> str:
    """
    Extract agent configuration from user's latest message.
    
    Extracts:
    - GitHub context (repo_url, branch_name) if mentioned
    - User context (description, goals, capabilities)
    - MCP services mentioned
    
    Returns:
        JSON string with extracted config (supervisor node will parse and update state)
    """
    from agents.eval_agent.nodes.plan.ensure_config import ensure_target_agent_config
    from agents.eval_agent.models import EvalAgentPlannerState
    from langchain_core.messages import HumanMessage
    import json
    
    # Get latest human message from runtime state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"error": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    messages = state.get("messages", [])
    
    # Find last human message
    last_human_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human" or (isinstance(msg, dict) and msg.get("type") == "human"):
            last_human_message = msg
            break
    
    if not last_human_message:
        return json.dumps({"error": "No human message found"})
    
    content = last_human_message.content if hasattr(last_human_message, "content") else last_human_message.get("content", "")
    
    try:
        # Use existing node logic
        temp_state = EvalAgentPlannerState(
            messages=[HumanMessage(content=content)],
            context=AgentContext(),
        )
        
        result = await ensure_target_agent_config(temp_state)
        updated_context = result.get("context")
        
        if updated_context:
            # Return as JSON for supervisor node to parse
            context_dict = updated_context.model_dump() if hasattr(updated_context, "model_dump") else updated_context
            logger.info(f"✅ Extracted agent config: repo={updated_context.github_context.repo_url if updated_context.github_context else 'None'}")
            return json.dumps({
                "status": "success",
                "agent_context": context_dict,
                "message": f"✅ Extracted agent config: repo={updated_context.github_context.repo_url if updated_context.github_context else 'None'}, mcp_services={updated_context.mcp_services}"
            })
        
        return json.dumps({
            "status": "success",
            "agent_context": AgentContext().model_dump(),
            "message": "✅ Extracted agent config (no GitHub URL found - plan-only mode)"
        })
    except Exception as e:
        logger.error(f"Error extracting agent config: {e}")
        return json.dumps({"status": "error", "error": str(e)})


@tool
async def provision_target_sandbox(runtime: ToolRuntime = None) -> str:
    """
    Provision E2B sandbox for target agent (if not plan-only mode).
    
    Reads agent_context from state and provisions sandbox if needed.
    
    Returns:
        JSON string with sandbox info or skip message
    """
    from agents.eval_agent.nodes.plan.provision_target import provision_target_agent
    from agents.eval_agent.models import EvalAgentPlannerState
    from langchain_core.messages import HumanMessage, ToolMessage
    import json
    
    # Check plan-only mode
    if config.eval_plan_only_mode:
        return json.dumps({
            "status": "skipped",
            "message": "✅ Plan-only mode - skipping sandbox provisioning"
        })
    
    # Get agent_context from state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"status": "error", "error": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    
    # Try to get agent_context from state first
    agent_context_dict = state.get("agent_context")
    
    # If agent_context not in state, try to extract from previous tool messages
    if not agent_context_dict:
        messages = state.get("messages", [])
        # Look for agent_context in previous tool results (from extract_agent_config)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                content = msg.content if hasattr(msg, "content") else str(msg)
                try:
                    if isinstance(content, str) and (content.startswith("{") or content.startswith("[")):
                        parsed = json.loads(content)
                        if "agent_context" in parsed:
                            agent_context_dict = parsed["agent_context"]
                            logger.info("✅ Found agent_context in previous tool result")
                            break
                except json.JSONDecodeError:
                    continue
    
    if not agent_context_dict:
        return json.dumps({"status": "error", "error": "agent_context not found in state or previous tool results. Please call extract_agent_config first."})
    
    try:
        # Reconstruct AgentContext from dict
        agent_context = AgentContext(**agent_context_dict) if isinstance(agent_context_dict, dict) else agent_context_dict
        
        # Create temporary state for node function
        temp_state = EvalAgentPlannerState(
            messages=[HumanMessage(content="Provision sandbox")],
            context=agent_context,
        )
        
        result = await provision_target_agent(temp_state)
        updated_context = result.get("context")
        
        if updated_context:
            context_dict = updated_context.model_dump() if hasattr(updated_context, "model_dump") else updated_context
            sandbox_id = updated_context.sandbox_context.sandbox_id if updated_context.sandbox_context else None
            logger.info(f"✅ Provisioned sandbox: {sandbox_id}")
            return json.dumps({
                "status": "success",
                "agent_context": context_dict,
                "message": f"✅ Provisioned sandbox: {sandbox_id}"
            })
        
        return json.dumps({"status": "error", "error": "No context returned from provisioning"})
    except Exception as e:
        logger.error(f"Error provisioning sandbox: {e}")
        return json.dumps({"status": "error", "error": str(e)})


@tool
async def generate_agent_spec(runtime: ToolRuntime = None) -> str:
    """
    Generate agent specification from user context and test cases.
    
    Reads agent_context and dataset_examples from state, generates AgentSpec and alignment questions.
    
    Returns:
        JSON string with agent_spec and alignment_state
    """
    from agents.eval_agent.nodes.plan.generate_spec import generate_agent_spec_and_alignment
    from agents.eval_agent.models import EvalAgentState
    from langchain_core.messages import ToolMessage
    import json
    
    # Get state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"status": "error", "error": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    
    # Try to get agent_context from state first
    agent_context_dict = state.get("agent_context")
    dataset_examples_dicts = state.get("dataset_examples", [])
    
    # If agent_context not in state, try to extract from previous tool messages
    if not agent_context_dict:
        messages = state.get("messages", [])
        # Look for agent_context in previous tool results (from extract_agent_config)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                content = msg.content if hasattr(msg, "content") else str(msg)
                try:
                    if isinstance(content, str) and (content.startswith("{") or content.startswith("[")):
                        parsed = json.loads(content)
                        if "agent_context" in parsed:
                            agent_context_dict = parsed["agent_context"]
                            logger.info("✅ Found agent_context in previous tool result")
                            break
                except json.JSONDecodeError:
                    continue
    
    if not agent_context_dict:
        return json.dumps({"status": "error", "error": "agent_context not found in state or previous tool results. Please call extract_agent_config first."})
    
    try:
        # Reconstruct objects from dicts
        agent_context = AgentContext(**agent_context_dict) if isinstance(agent_context_dict, dict) else agent_context_dict
        
        from shared.schema import DatasetExample
        dataset_examples = []
        for ex_dict in dataset_examples_dicts:
            if isinstance(ex_dict, dict):
                dataset_examples.append(DatasetExample(**ex_dict))
            else:
                dataset_examples.append(ex_dict)
        
        # Create temporary state for node function
        temp_state = EvalAgentState(
            messages=[],
            context=agent_context,
            dataset_examples=dataset_examples,
        )
        
        result = await generate_agent_spec_and_alignment(temp_state)
        agent_spec = result.get("agent_spec")
        alignment_state = result.get("alignment_state")
        
        if agent_spec and alignment_state:
            spec_dict = agent_spec.model_dump() if hasattr(agent_spec, "model_dump") else agent_spec
            alignment_dict = alignment_state.model_dump() if hasattr(alignment_state, "model_dump") else alignment_state
            
            logger.info(f"✅ Generated agent spec: {agent_spec.agent_name}")
            return json.dumps({
                "status": "success",
                "agent_spec": spec_dict,
                "alignment_state": alignment_dict,
                "message": f"✅ Generated agent spec: {agent_spec.agent_name} with {len(alignment_state.questions)} alignment questions"
            })
        
        return json.dumps({"status": "error", "error": "Failed to generate spec"})
    except Exception as e:
        logger.error(f"Error generating agent spec: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)})


@tool
async def generate_test_cases(runtime: ToolRuntime = None) -> str:
    """
    Generate test cases using LLM.
    
    Reads agent_context from state and generates dataset_examples.
    
    Returns:
        JSON string with dataset_examples
    """
    from agents.eval_agent.nodes.plan.agentic_eval_generation import agentic_eval_generation
    from agents.eval_agent.models import EvalAgentPlannerState
    from langchain_core.messages import HumanMessage, ToolMessage
    import json
    
    # Get state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"status": "error", "error": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    
    # Try to get agent_context from state first
    agent_context_dict = state.get("agent_context")
    
    # If agent_context not in state, try to extract from previous tool messages
    if not agent_context_dict:
        messages = state.get("messages", [])
        # Look for agent_context in previous tool results (from extract_agent_config)
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                content = msg.content if hasattr(msg, "content") else str(msg)
                try:
                    if isinstance(content, str) and (content.startswith("{") or content.startswith("[")):
                        parsed = json.loads(content)
                        if "agent_context" in parsed:
                            agent_context_dict = parsed["agent_context"]
                            logger.info("✅ Found agent_context in previous tool result")
                            break
                except json.JSONDecodeError:
                    continue
    
    if not agent_context_dict:
        return json.dumps({"status": "error", "error": "agent_context not found in state or previous tool results. Please call extract_agent_config first."})
    
    try:
        # Reconstruct AgentContext
        agent_context = AgentContext(**agent_context_dict) if isinstance(agent_context_dict, dict) else agent_context_dict
        
        # Create temporary state for node function
        temp_state = EvalAgentPlannerState(
            messages=[HumanMessage(content=agent_context.user_context.raw_request)],
            context=agent_context,
            reflections_text="",  # No reflections for first generation
            latest_results=[],  # No previous results
        )
        
        result = await agentic_eval_generation(temp_state)
        dataset_examples = result.get("dataset_examples", [])
        
        if dataset_examples:
            examples_dicts = [ex.model_dump() if hasattr(ex, "model_dump") else ex for ex in dataset_examples]
            logger.info(f"✅ Generated {len(dataset_examples)} test cases")
            return json.dumps({
                "status": "success",
                "dataset_examples": examples_dicts,
                "message": f"✅ Generated {len(dataset_examples)} test cases"
            })
        
        return json.dumps({"status": "error", "error": "No test cases generated"})
    except Exception as e:
        logger.error(f"Error generating test cases: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)})


@tool
async def show_alignment_questions(runtime: ToolRuntime = None) -> str:
    """
    Show alignment questions to the user.
    
    Reads alignment_state from state and formats questions for display.
    
    Returns:
        Formatted string with alignment questions
    """
    import json
    
    # Get state
    if not runtime or not hasattr(runtime, "state"):
        return "❌ No runtime state available"
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    alignment_state_dict = state.get("alignment_state")
    
    if not alignment_state_dict:
        return "❌ No alignment questions found. Generate agent spec first."
    
    try:
        from shared.schema import AlignmentState
        alignment_state = AlignmentState(**alignment_state_dict) if isinstance(alignment_state_dict, dict) else alignment_state_dict
        
        if not alignment_state.questions:
            return "❌ No alignment questions available"
        
        # Format questions
        parts = ["📋 **Alignment Questions**\n"]
        parts.append("To ensure we're aligned, please answer the following questions:\n")
        
        for i, q in enumerate(alignment_state.questions, 1):
            parts.append(f"**Question {i}:** {q.question}")
            parts.append(f"*Context: {q.context}*")
            parts.append(f"*Question ID: {q.question_id}*")
            parts.append("")
        
        parts.append("---")
        parts.append("\nYou can answer by replying with your answers in natural language.")
        parts.append("You can skip questions you don't want to answer.")
        
        return "\n".join(parts)
    except Exception as e:
        logger.error(f"Error showing alignment questions: {e}")
        return f"❌ Error: {str(e)}"


@tool
async def process_alignment_answers(user_answers: str, runtime: ToolRuntime = None) -> str:
    """
    Process user's alignment answers and refine agent spec.
    
    Args:
        user_answers: User's answers in natural language or JSON format
        
    Returns:
        JSON string with updated alignment_state and agent_spec
    """
    from agents.eval_agent.nodes.alignment import alignment_node
    from agents.eval_agent.models import EvalAgentState
    from langchain_core.messages import HumanMessage
    import json
    
    # Get state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"status": "error", "error": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    agent_context_dict = state.get("agent_context")
    alignment_state_dict = state.get("alignment_state")
    agent_spec_dict = state.get("agent_spec")
    
    if not alignment_state_dict or not agent_spec_dict:
        return json.dumps({"status": "error", "error": "alignment_state or agent_spec not found"})
    
    try:
        # Reconstruct objects
        from shared.schema import AgentSpec, AlignmentState
        agent_context = AgentContext(**agent_context_dict) if isinstance(agent_context_dict, dict) else agent_context_dict
        alignment_state = AlignmentState(**alignment_state_dict) if isinstance(alignment_state_dict, dict) else alignment_state_dict
        agent_spec = AgentSpec(**agent_spec_dict) if isinstance(agent_spec_dict, dict) else agent_spec_dict
        
        # Create temporary state for node function
        temp_state = EvalAgentState(
            messages=[HumanMessage(content=user_answers)],
            context=agent_context,
            alignment_state=alignment_state,
            agent_spec=agent_spec,
        )
        
        result = await alignment_node(temp_state)
        updated_alignment = result.get("alignment_state")
        updated_spec = result.get("agent_spec")
        
        if updated_alignment and updated_spec:
            alignment_dict = updated_alignment.model_dump() if hasattr(updated_alignment, "model_dump") else updated_alignment
            spec_dict = updated_spec.model_dump() if hasattr(updated_spec, "model_dump") else updated_spec
            
            logger.info(f"✅ Processed alignment answers, spec refined")
            return json.dumps({
                "status": "success",
                "alignment_state": alignment_dict,
                "agent_spec": spec_dict,
                "message": "✅ Alignment answers processed, agent spec refined"
            })
        
        return json.dumps({"status": "error", "error": "Failed to process answers"})
    except Exception as e:
        logger.error(f"Error processing alignment answers: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)})

