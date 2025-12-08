"""
Supervisor node for eval agent using Supervisor pattern.

This replaces the compiled graph with a single supervisor node that uses create_agent
with tools, enforcing the think() → tool → think() → tool pattern.
"""
import json
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage, BaseMessage
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelRetryMiddleware
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from agents.eval_agent.supervisor_state import EvalAgentState
from agents.eval_agent.tools.core import think, write_todos
from agents.eval_agent.tools.planning import (
    classify_user_intent,
    extract_agent_config,
    provision_target_sandbox,
    generate_agent_spec,
    generate_test_cases,
    show_alignment_questions,
    process_alignment_answers,
)
from agents.eval_agent.tools.execution import execute_test_batch
from agents.eval_agent.tools.reflection import reflect_on_results
from agents.eval_agent.tools.finalization import finalize_evaluation
from shared.logger import get_logger
from shared.config import config
from shared.schema import UserIntent, AgentContext

logger = get_logger("eval_agent.supervisor")


def create_eval_supervisor():
    """
    Creates the Eval Agent Supervisor using Supervisor pattern.
    
    Architecture:
    - Single supervisor node that handles all evaluation phases
    - Uses create_agent with tools for LLM-driven workflow
    - Enforces alternating think() → tool pattern
    - State updates extracted from tool call results
    """
    
    # 1. Define tools
    tools = [
        think,
        write_todos,
        classify_user_intent,
        extract_agent_config,
        provision_target_sandbox,
        generate_agent_spec,
        generate_test_cases,
        show_alignment_questions,
        process_alignment_answers,
        execute_test_batch,
        reflect_on_results,
        finalize_evaluation,
    ]
    
    # 2. Create model with middleware
    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.0,
    )
    
    middleware = [
        ToolCallLimitMiddleware(thread_limit=50, run_limit=20),  # Prevent infinite loops
        ModelRetryMiddleware(max_retries=2),
    ]
    
    # 3. Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware
    )
    
    # 4. Build system prompt
    system_prompt = """You are the Eval Agent Supervisor. You evaluate AI agents through structured phases.

**CURRENT PHASE**: {current_phase}
**CURRENT TODOS**: {todos_text}
**PLAN-ONLY MODE**: {plan_only_mode}

**PHASE PROTOCOL:**

1. **PLANNING PHASE** (when todos include "PLANNING"):
   - Call `think()` to plan your approach
   - Call `classify_user_intent()` to determine if user wants evaluation or information
   - Call `think()` to reflect
   - If evaluation_request: Call `extract_agent_config()` to extract GitHub/user context
   - Call `think()` to reflect
   - Continue with planning tools...
   - Remove "PLANNING" from todos when complete, add "EXECUTION" (if not plan-only mode)

2. **EXECUTION PHASE** (when todos include "EXECUTION"):
   - Call `think()` to plan execution
   - Call execution tools...
   - Remove "EXECUTION" from todos, add "REFLECTION"

3. **REFLECTION PHASE** (when todos include "REFLECTION"):
   - Call `think()` to plan reflection
   - Call reflection tools...
   - Remove "REFLECTION" from todos, add "FINALIZATION"

4. **FINALIZATION PHASE** (when todos include "FINALIZATION"):
   - Call `think()` to plan finalization
   - Call finalization tools...
   - Remove "FINALIZATION" from todos

**CRITICAL: ALTERNATING TOOL CALL PATTERN**
- ALWAYS call `think()` before every action tool
- ALWAYS call `think()` after every action tool
- Pattern: think → tool → think → tool → think → tool

**INFORMATIONAL QUERIES**:
- If user asks informational questions, answer directly
- Do NOT create todos or call eval tools
"""
    
    # 5. Define the supervisor node
    async def supervisor_node(state: EvalAgentState):
        logger.info("🤖 Eval Supervisor Node Active")
        messages = state.get("messages", [])
        todos = state.get("todos", [])
        plan_only_mode = state.get("plan_only_mode", config.eval_plan_only_mode)
        current_phase = state.get("current_phase")
        
        # Normalize messages - ensure all are proper LangChain message objects
        # This is critical: create_agent expects proper BaseMessage subclasses (HumanMessage, AIMessage, etc.)
        # NOT generic BaseMessage instances with type attributes
        normalized_messages = []
        for msg in messages:
            try:
                # Check if it's already a proper LangChain message subclass (HumanMessage, AIMessage, etc.)
                # NOT just BaseMessage - BaseMessage instances with type='human' need to be converted
                if isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
                    normalized_messages.append(msg)
                    continue
                
                # Handle BaseMessage instances that aren't proper subclasses
                # These have type attributes but need to be converted to proper subclasses
                if isinstance(msg, BaseMessage):
                    msg_type = getattr(msg, "type", None)
                    content = getattr(msg, "content", "") or ""
                    id_ = getattr(msg, "id", None)
                    name = getattr(msg, "name", None)
                    tool_calls = getattr(msg, "tool_calls", [])
                    
                    if msg_type == "human" or msg_type == "user":
                        normalized_messages.append(HumanMessage(content=str(content), id=id_, name=name))
                    elif msg_type == "ai" or msg_type == "assistant":
                        ai_msg = AIMessage(content=str(content), id=id_, name=name)
                        if tool_calls:
                            ai_msg.tool_calls = tool_calls
                        normalized_messages.append(ai_msg)
                    elif msg_type == "system":
                        normalized_messages.append(SystemMessage(content=str(content), id=id_, name=name))
                    elif msg_type == "tool":
                        tool_call_id = getattr(msg, "tool_call_id", None) or id_
                        normalized_messages.append(ToolMessage(
                            content=str(content),
                            tool_call_id=tool_call_id or "unknown",
                            id=id_,
                            name=name
                        ))
                    else:
                        # Unknown type, default to HumanMessage
                        normalized_messages.append(HumanMessage(content=str(content), id=id_, name=name))
                    continue
                
                # Handle dict messages (from LangServe or state)
                if isinstance(msg, dict):
                    msg_type = msg.get("type") or msg.get("role", "human")
                    content = msg.get("content", "") or ""
                    
                    if msg_type == "human" or msg_type == "user":
                        normalized_messages.append(HumanMessage(content=str(content)))
                    elif msg_type == "ai" or msg_type == "assistant":
                        ai_msg = AIMessage(content=str(content))
                        tool_calls = msg.get("tool_calls", [])
                        if tool_calls:
                            ai_msg.tool_calls = tool_calls
                        normalized_messages.append(ai_msg)
                    elif msg_type == "tool":
                        tool_call_id = msg.get("tool_call_id") or msg.get("id", "unknown")
                        normalized_messages.append(ToolMessage(
                            content=str(content),
                            tool_call_id=tool_call_id
                        ))
                    else:
                        logger.warning(f"Unknown dict message type '{msg_type}', converting to HumanMessage")
                        normalized_messages.append(HumanMessage(content=str(content)))
                    continue
                
                # Handle objects with message-like attributes (Pydantic models, etc.)
                # These have attributes but aren't BaseMessage instances
                if hasattr(msg, "type") or hasattr(msg, "content"):
                    msg_type = getattr(msg, "type", None) or getattr(msg, "role", "human")
                    content = getattr(msg, "content", "") or ""
                    
                    if msg_type == "human" or msg_type == "user":
                        normalized_messages.append(HumanMessage(content=str(content)))
                    elif msg_type == "ai" or msg_type == "assistant":
                        ai_msg = AIMessage(content=str(content))
                        if hasattr(msg, "tool_calls"):
                            tool_calls = getattr(msg, "tool_calls", [])
                            if tool_calls:
                                ai_msg.tool_calls = tool_calls
                        normalized_messages.append(ai_msg)
                    elif msg_type == "tool":
                        tool_call_id = getattr(msg, "tool_call_id", None) or getattr(msg, "id", "unknown")
                        normalized_messages.append(ToolMessage(
                            content=str(content),
                            tool_call_id=tool_call_id
                        ))
                    else:
                        logger.warning(f"Unknown object message type '{msg_type}', converting to HumanMessage")
                        normalized_messages.append(HumanMessage(content=str(content)))
                    continue
                
                # Last resort: convert to string and create HumanMessage
                logger.warning(f"Unknown message type {type(msg)}, converting to HumanMessage")
                content_str = str(msg)
                if hasattr(msg, "content"):
                    content_str = str(msg.content)
                normalized_messages.append(HumanMessage(content=content_str))
            except Exception as e:
                logger.error(f"Error normalizing message: {e}", exc_info=True)
                # Emergency fallback
                try:
                    normalized_messages.append(HumanMessage(content=f"[Error: {str(msg)[:100]}]"))
                except:
                    continue
        
        # Format system prompt with current state
        todos_text = ", ".join(todos) if todos else "None"
        formatted_system_prompt = system_prompt.format(
            current_phase=current_phase or "None",
            todos_text=todos_text,
            plan_only_mode=plan_only_mode
        )
        
        # Prepend system message
        agent_messages = [SystemMessage(content=formatted_system_prompt)] + normalized_messages
        
        # Final validation: ensure all messages are BaseMessage instances
        # This is critical for create_agent compatibility - it will fail with TypeError if not
        final_messages = []
        for i, msg in enumerate(agent_messages):
            if isinstance(msg, BaseMessage):
                final_messages.append(msg)
            else:
                logger.error(f"Message {i} ({type(msg)}) is not BaseMessage after normalization! Content: {str(msg)[:100]}")
                logger.error(f"Message attributes: {dir(msg)[:20]}")
                # Emergency fallback - convert to HumanMessage
                content = str(msg)
                if hasattr(msg, "content"):
                    content = str(msg.content)
                final_messages.append(HumanMessage(content=content))
        
        # Log final message types for debugging
        logger.debug(f"Final messages: {[type(m).__name__ for m in final_messages]}")
        
        # Invoke agent
        try:
            result = await agent.ainvoke({"messages": final_messages})
            
            # Extract agent's response
            agent_response_raw = result.get("messages", [])
            
            # Normalize agent response messages (create_agent may return objects, not BaseMessage)
            agent_response = []
            for msg in agent_response_raw:
                if isinstance(msg, BaseMessage):
                    agent_response.append(msg)
                elif isinstance(msg, dict):
                    # Convert dict to proper message
                    msg_type = msg.get("type") or msg.get("role", "ai")
                    content = msg.get("content", "") or ""
                    if msg_type == "ai" or msg_type == "assistant":
                        ai_msg = AIMessage(content=str(content))
                        tool_calls = msg.get("tool_calls", [])
                        if tool_calls:
                            ai_msg.tool_calls = tool_calls
                        agent_response.append(ai_msg)
                    elif msg_type == "human":
                        agent_response.append(HumanMessage(content=str(content)))
                    elif msg_type == "tool":
                        agent_response.append(ToolMessage(
                            content=str(msg.get("content", "")),
                            tool_call_id=msg.get("tool_call_id") or msg.get("id", "unknown")
                        ))
                    else:
                        agent_response.append(AIMessage(content=str(content)))
                elif hasattr(msg, "type") or hasattr(msg, "content"):
                    # Object with message-like attributes
                    msg_type = getattr(msg, "type", None) or getattr(msg, "role", "ai")
                    content = getattr(msg, "content", "") or ""
                    if msg_type == "ai" or msg_type == "assistant":
                        agent_response.append(AIMessage(content=str(content)))
                    elif msg_type == "human":
                        agent_response.append(HumanMessage(content=str(content)))
                    else:
                        agent_response.append(AIMessage(content=str(content)))
                else:
                    logger.warning(f"Unknown agent response message type {type(msg)}, converting to AIMessage")
                    agent_response.append(AIMessage(content=str(msg)))
            
            if agent_response:
                # Get last AI message
                last_ai = None
                for msg in reversed(agent_response):
                    if isinstance(msg, AIMessage):
                        last_ai = msg
                        break
                
                if last_ai:
                    # Extract state updates from tool calls
                    state_updates: Dict[str, Any] = {}
                    
                    # Check for tool calls that update state
                    if hasattr(last_ai, "tool_calls") and last_ai.tool_calls:
                        for tool_call in last_ai.tool_calls:
                            tool_name = tool_call.get("name", "")
                            tool_args = tool_call.get("args", {})
                            
                            # Extract todos from write_todos
                            if tool_name == "write_todos":
                                if "todos" in tool_args:
                                    state_updates["todos"] = tool_args["todos"]
                            
                            # Extract user_intent from classify_user_intent result
                            # (will be in tool message, need to parse)
                    
                    # Extract state from tool messages (tool results)
                    for msg in agent_response:
                        if isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, "name", None) or ""
                            content = msg.content if hasattr(msg, "content") else str(msg)
                            
                            # Parse JSON responses from tools
                            try:
                                if content.startswith("{") or content.startswith("["):
                                    parsed = json.loads(content)
                                    
                                    # Update state based on tool result
                                    if "user_intent" in parsed or "intent_type" in parsed:
                                        state_updates["user_intent"] = parsed
                                    
                                    if "agent_context" in parsed:
                                        state_updates["agent_context"] = parsed["agent_context"]
                                    
                                    if "dataset_examples" in parsed:
                                        state_updates["dataset_examples"] = parsed["dataset_examples"]
                                    
                                    if "agent_spec" in parsed:
                                        state_updates["agent_spec"] = parsed["agent_spec"]
                                    
                                    if "alignment_state" in parsed:
                                        state_updates["alignment_state"] = parsed["alignment_state"]
                                    
                                    if "latest_results" in parsed:
                                        state_updates["latest_results"] = parsed["latest_results"]
                                    
                                    if "hypothesis" in parsed:
                                        state_updates["hypothesis"] = parsed["hypothesis"]
                            except json.JSONDecodeError:
                                # Not JSON, skip
                                pass
                    
                    # Update current_phase based on todos
                    if todos:
                        if "PLANNING" in todos:
                            state_updates["current_phase"] = "planning"
                        elif "EXECUTION" in todos:
                            state_updates["current_phase"] = "execution"
                        elif "REFLECTION" in todos:
                            state_updates["current_phase"] = "reflection"
                        elif "FINALIZATION" in todos:
                            state_updates["current_phase"] = "finalization"
                    else:
                        state_updates["current_phase"] = None
                    
                    # Return updates
                    return {
                        "messages": agent_response,
                        **state_updates
                    }
            
            return {"messages": agent_response}
            
        except Exception as e:
            logger.error(f"Error in supervisor node: {e}", exc_info=True)
            error_msg = AIMessage(content=f"Error: {str(e)}")
            return {"messages": [error_msg]}
    
    # 6. Define graph
    workflow = StateGraph(EvalAgentState)
    workflow.add_node("supervisor", supervisor_node)
    
    workflow.add_edge(START, "supervisor")
    
    def should_continue(state: EvalAgentState):
        """Check if we need to continue or finish."""
        todos = state.get("todos", [])
        
        if not todos:
            logger.info("✅ All todos complete. Ending.")
            return END
        
        logger.info(f"🔄 Looping: {len(todos)} todos remaining.")
        return "supervisor"
    
    workflow.add_conditional_edges("supervisor", should_continue, {
        "supervisor": "supervisor",
        END: END
    })
    
    return workflow.compile(checkpointer=MemorySaver())


# Create the graph instance
graph = create_eval_supervisor()

