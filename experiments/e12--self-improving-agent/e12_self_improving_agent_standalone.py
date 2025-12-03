"""
Experiment E12: Self-Improving Agent - Using Reflexion + Neo4j Memory

This version uses:
- Reflexion architecture (Act → Evaluate → Reflect loop)
- Neo4jMemoryStore for persistent memory across runs
- LLM-based evaluation (no hardcoded phrases)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import uuid4

# Load environment variables from Seer root .env file
from dotenv import load_dotenv
from pathlib import Path

# Find Seer root directory (go up from experiments/e12--self-improving-agent)
seer_root = Path(__file__).parent.parent.parent
env_file = seer_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback: try current directory
    load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import Composio directly (matching Seer's approach)
from composio import Composio
from composio_langchain import LangchainProvider

# Import Reflexion and Neo4j memory
from reflexion.agent.graph import create_reflexion
from reflexion.memory.store import Neo4jMemoryStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Models
# ============================================================================

class FailureExplanation(BaseModel):
    """Agent's explanation of why it failed."""
    failure_reason: str = Field(description="Why did this attempt fail?")
    root_cause: str = Field(description="What was the root cause?")
    what_went_wrong: str = Field(description="What specifically went wrong?")
    confidence: float = Field(description="Confidence in explanation (0-1)")

class LearningSummary(BaseModel):
    """Agent's summary of what it learned from failures."""
    key_learnings: List[str] = Field(description="Key things learned from failures")
    patterns_noticed: List[str] = Field(description="Patterns noticed across failures")
    what_to_avoid: List[str] = Field(description="What to avoid next time")
    what_to_try: List[str] = Field(description="What to try next time")

class AdaptationPlan(BaseModel):
    """Agent's plan for how to adapt based on learnings."""
    changes_to_make: List[str] = Field(description="Specific changes to make")
    approach_differences: str = Field(description="How will approach differ from last attempt?")
    expected_improvement: str = Field(description="What improvement is expected?")

class ExperimentRun(BaseModel):
    """Results from a single experiment run."""
    run_id: str
    condition: str  # "with_memory" or "without_memory"
    attempt_number: int
    success: bool
    failure_explanation: Optional[FailureExplanation] = None
    learning_summary: Optional[LearningSummary] = None
    adaptation_plan: Optional[AdaptationPlan] = None
    prompt_version: str = Field(description="Prompt version used")
    execution_time: float
    output: str = ""
    tool_calls_made: bool = False
    tool_call_count: int = 0
    timestamp: datetime

# ============================================================================
# Tools for Learning Agent
# ============================================================================

def create_failure_explanation_tool(model: ChatOpenAI):
    """Tool for agent to explain why it failed. Uses GPT-5.1 with high reasoning."""
    
    # Use GPT-5.1 for high-quality reasoning
    reasoning_model = ChatOpenAI(
        model="gpt-5.1",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    @tool
    def explain_failure(task: str, attempt_output: str, expected_output: str, error_message: str = "") -> str:
        """Explain why this attempt failed. Think deeply about root causes."""
        prompt = f"""You attempted this task but failed. Analyze deeply why.

Task: {task}
Your Output: {attempt_output[:1000]}
Expected Output: {expected_output}
Error: {error_message}

Think step by step:
1. What did you actually do vs what you should have done?
2. What was the root cause of failure?
3. What specifically went wrong?
4. How confident are you in this explanation?

Provide a thorough analysis."""
        response = reasoning_model.with_structured_output(FailureExplanation).invoke([HumanMessage(content=prompt)])
        return json.dumps(response.model_dump(), indent=2)
    
    return explain_failure

def create_learning_summary_tool(model: ChatOpenAI):
    """Tool for agent to summarize what it learned from failures. Uses GPT-5.1."""
    
    reasoning_model = ChatOpenAI(
        model="gpt-5.1",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    @tool
    def summarize_learnings(last_n_failures: int = 3, failure_summaries: str = "") -> str:
        """Summarize what you learned from your last N failures. Think deeply about patterns."""
        prompt = f"""You have failed {last_n_failures} times on similar tasks. Analyze patterns deeply.

{failure_summaries}

Think step by step:
1. What patterns do you see across these failures?
2. What are the key learnings?
3. What should you avoid next time?
4. What should you try differently?

Provide a comprehensive analysis."""
        response = reasoning_model.with_structured_output(LearningSummary).invoke([HumanMessage(content=prompt)])
        return json.dumps(response.model_dump(), indent=2)
    
    return summarize_learnings

def create_adaptation_tool(model: ChatOpenAI):
    """Tool for agent to explain how it will adapt. Uses GPT-5.1."""
    
    reasoning_model = ChatOpenAI(
        model="gpt-5.1",
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    @tool
    def plan_adaptation(learnings: str, next_attempt: str) -> str:
        """Plan how you will adapt based on your learnings. Think strategically."""
        prompt = f"""Based on these learnings:
{learnings}

And this next attempt:
{next_attempt}

Think step by step:
1. What specific changes will you make?
2. How will your approach differ?
3. What improvement do you expect?
4. Why will this work better?

Provide a strategic adaptation plan."""
        response = reasoning_model.with_structured_output(AdaptationPlan).invoke([HumanMessage(content=prompt)])
        return json.dumps(response.model_dump(), indent=2)
    
    return plan_adaptation

# ============================================================================
# Task Executor
# ============================================================================

async def execute_task(
    task: str, 
    agent_prompt: str, 
    tools: List[Any], 
    model: ChatOpenAI,
    memory_store: Optional[Neo4jMemoryStore] = None,
    agent_id: str = "learning_agent"
) -> Dict[str, Any]:
    """Execute a task using Reflexion architecture with optional Neo4j memory."""
    start_time = time.time()
    
    try:
        # Use Reflexion architecture if memory_store provided, else fallback to simple agent
        if memory_store:
            # OPTIMIZED: Use intelligent multi-strategy memory retrieval
            try:
                # Use Reflexion's built-in intelligent retrieval (semantic + entity + recency)
                past_memories_str = memory_store.retrieve_relevant_memories(
                    agent_id=agent_id,
                    query_context=task,
                    llm_model=model,
                    limit=5,
                    strategies=["semantic", "entity", "recency"],  # Multi-strategy
                    enable_multihop=False  # Disable for now (faster)
                )
                
                if past_memories_str and "No memories yet" not in past_memories_str:
                    # Enhance prompt with intelligent memory context
                    agent_prompt = f"""{agent_prompt}

RELEVANT PAST EXPERIENCES:
{past_memories_str}

Learn from these experiences. Avoid repeating past mistakes and build on what worked.
IMPORTANT: Based on past failures, you MUST actually call tools, not just describe them."""
                    
                    logger.info(f"Retrieved relevant memories using multi-strategy search")
                    print(f"✓ Retrieved relevant memories (semantic + entity + recency)", flush=True)
                else:
                    logger.info("No past memories found - this is a fresh start")
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}, continuing without memory context")
                # Fallback to simple retrieval
                try:
                    from reflexion.memory.models import MemoryQuery
                    simple_query = MemoryQuery(agent_id=agent_id, limit=3)
                    fallback_memories = memory_store.retrieve(simple_query, record_access=True)
                    if fallback_memories:
                        memory_context = "\n".join([f"- {m.observation[:150]}..." for m in fallback_memories[:3]])
                        agent_prompt = f"""{agent_prompt}\n\nPAST EXPERIENCES:\n{memory_context}"""
                        logger.info(f"Used fallback retrieval: {len(fallback_memories)} memories")
                except:
                    pass
            
            # Create Reflexion agent with memory
            agent = create_reflexion(
                model=model,
                tools=tools,
                prompt=agent_prompt,
                memory_store=memory_store,
                agent_id=agent_id,
                max_rounds=3,
                eval_threshold=0.8,
                simple_mode=True  # Use simple Act-Evaluate loop
            )
            
            # Execute with Reflexion (includes evaluation loop)
            result = agent.invoke(
                {"messages": [HumanMessage(content=task)], "current_round": 0},
                config={"agent_id": agent_id, "max_rounds": 3}
            )
            
            # Extract messages from Reflexion result
            # Reflexion simple mode returns: {"run_trace": [...messages...], "messages": [...], "evaluation": {...}}
            # The actual tool execution happens in run_trace
            messages = result.get("run_trace", [])
            if not messages:
                messages = result.get("messages", [])
            
            logger.info(f"Reflexion result structure: keys={list(result.keys())}, run_trace length={len(result.get('run_trace', []))}, messages length={len(result.get('messages', []))}")
        else:
            # Fallback: simple agent (for "without_memory" condition)
            from langchain.agents import create_agent
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=agent_prompt
            )
            result = agent.invoke({
                "messages": [HumanMessage(content=task)]
            })
            messages = result.get("messages", [])
        
        # Extract final output
        final_output = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                final_output = msg.content
                break
        
        # Check for actual tool execution in messages
        tool_calls_made = False
        tool_call_count = 0
        
        # Comprehensive tool call detection - check all message types
        for msg in messages:
            # Method 1: Check for tool_calls attribute (AIMessage with tool_calls)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls_made = True
                tool_call_count += len(msg.tool_calls)
                logger.debug(f"Found {len(msg.tool_calls)} tool_calls in message")
            
            # Method 2: Check for ToolMessage type (tool execution result)
            if hasattr(msg, 'type'):
                msg_type = msg.type if isinstance(msg.type, str) else str(msg.type)
                if msg_type == "tool":
                    tool_calls_made = True
                    tool_call_count += 1
                    logger.debug(f"Found ToolMessage")
            
            # Method 3: Check message class name
            msg_class = type(msg).__name__
            if 'Tool' in msg_class or 'tool' in msg_class.lower():
                tool_calls_made = True
                tool_call_count += 1
                logger.debug(f"Found tool message class: {msg_class}")
            
            # Method 4: Check for name attribute (tool name)
            if hasattr(msg, 'name') and msg.name:
                name_str = str(msg.name).lower()
                # Skip if it's a generic name like "messages" or "content"
                if name_str and name_str not in ['messages', 'content', 'ai', 'human']:
                    tool_calls_made = True
                    tool_call_count += 1
                    logger.debug(f"Found tool name: {msg.name}")
        
        logger.info(f"Tool detection: found {tool_call_count} tool calls in {len(messages)} messages")
        
        # Detailed inspection for debugging
        if memory_store and tool_call_count == 0:
            logger.warning("Reflexion returned 0 tool calls - inspecting messages:")
            for i, msg in enumerate(messages[:10]):
                msg_info = {
                    "index": i,
                    "type": type(msg).__name__,
                    "has_content": hasattr(msg, 'content'),
                    "has_tool_calls": hasattr(msg, 'tool_calls'),
                    "has_name": hasattr(msg, 'name'),
                }
                if hasattr(msg, 'type'):
                    msg_info["msg_type"] = msg.type if isinstance(msg.type, str) else str(msg.type)
                if hasattr(msg, 'tool_calls'):
                    msg_info["tool_calls"] = msg.tool_calls
                if hasattr(msg, 'name'):
                    msg_info["name"] = msg.name
                logger.warning(f"  Message {i}: {msg_info}")
        
        # Use LLM to evaluate success - NO HARDCODED PHRASES
        # Create a separate evaluator model with high reasoning
        evaluator_model = ChatOpenAI(
            model="gpt-5.1",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        evaluation_prompt = f"""You are evaluating whether an agent successfully completed a task.

TASK:
{task}

AGENT'S EXECUTION OUTPUT:
{final_output[:2000]}

TOOL CALLS MADE: {tool_calls_made}
TOOL CALL COUNT: {tool_call_count}

SUCCESS CRITERIA (from task description):
- Asana tasks found and updated
- Tasks closed successfully  
- PR details correctly added to tasks

CRITICAL EVALUATION RULES:
1. The agent MUST have called tools (experiments) - no tool calls = automatic failure
2. The agent must have actually EXECUTED the task, not just described how to do it
3. The agent must have GATHERED EVIDENCE before making claims
4. If the agent is asking for input or providing instructions/code instead of executing, it failed
5. If the agent explicitly says it didn't complete the task (e.g., "I have not", "I cannot"), it failed

Evaluate whether the task was successfully completed. Consider:
- Did the agent actually call tools?
- Did the agent execute the task or just describe plans?
- Does the output indicate successful completion?
- Are the success criteria met?

Respond with ONLY a JSON object:
{{
    "success": true/false,
    "reasoning": "Brief explanation of your evaluation",
    "confidence": 0.0-1.0
}}"""

        try:
            eval_response = evaluator_model.invoke([HumanMessage(content=evaluation_prompt)])
            eval_content = eval_response.content
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{[^}]+\}', eval_content, re.DOTALL)
            if json_match:
                eval_result = json.loads(json_match.group())
                success = eval_result.get("success", False)
                reasoning = eval_result.get("reasoning", "")
                confidence = eval_result.get("confidence", 0.5)
                
                logger.info(f"LLM Evaluation: success={success}, confidence={confidence:.2f}, reasoning={reasoning[:100]}")
            else:
                # Fallback: parse from text
                success = "success" in eval_content.lower() and "true" in eval_content.lower()
                logger.warning(f"Could not parse JSON from evaluator, using fallback: {success}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}, using fallback")
            # Fallback: require tool calls
            success = tool_calls_made and len(final_output) > 50
        
        # OPTIMIZED: Store memory for both successes and failures (learn from what works)
        if memory_store:
            try:
                from reflexion.memory.models import Memory
                # Store if success (score >= 0.9) OR failure (score < 0.8)
                # This allows learning from both what works and what doesn't
                evaluation_score = 0.9 if success else 0.3  # Approximate score
                should_store = success or not success  # Store everything
                
                if should_store:
                    memory = Memory(
                        agent_id=agent_id,
                        context=f"experiment.e12.{'success' if success else 'failure'}",
                        entities=["evaluation", "success" if success else "failure", "tool_execution"],
                        observation=f"Task: {task[:200]}\nOutput: {final_output[:500]}\nSuccess: {success}\nTool Calls: {tool_call_count}\nScore: {evaluation_score}"
                    )
                    memory_store.save(memory)
                    logger.info(f"Saved memory: {'success' if success else 'failure'} (score: {evaluation_score})")
            except Exception as e:
                logger.warning(f"Failed to save memory: {e}")
        
        # Log tool call info
        logger.info(f"Tool calls made: {tool_calls_made}, Count: {tool_call_count}, Success: {success}")
        
        execution_time = time.time() - start_time
        
        return {
            "success": success,
            "output": final_output,
            "execution_time": execution_time,
            "tool_calls_made": tool_calls_made,
            "tool_call_count": tool_call_count
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Task execution failed: {e}")
        return {
            "success": False,
            "output": f"Error: {str(e)}",
            "execution_time": execution_time,
            "error": str(e),
            "tool_calls_made": False,
            "tool_call_count": 0
        }

# ============================================================================
# Experiment Runner
# ============================================================================

class SelfImprovingAgentExperiment:
    """Runs the self-improving agent experiment using Reflexion + Neo4j."""
    
    def __init__(self):
        # Initialize model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.model = ChatOpenAI(
            model="gpt-5.1",
            temperature=0.0,
            api_key=api_key
        )
        
        self.results: List[ExperimentRun] = []
        self.tools = None
        
        # Initialize Neo4j memory store (using Seer's standard env vars)
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "")
        
        if not neo4j_password:
            logger.warning("NEO4J_PASSWORD not set in .env. Memory will not persist across runs.")
            self.memory_store = None
        else:
            try:
                self.memory_store = Neo4jMemoryStore(
                    uri=neo4j_uri,
                    username=neo4j_username,
                    password=neo4j_password
                )
                logger.info(f"✓ Neo4j memory store initialized (URI: {neo4j_uri})")
                print(f"✓ Neo4j memory store initialized", flush=True)
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j: {e}")
                logger.warning("Falling back to in-memory mode (no persistence)")
                print(f"⚠️ Neo4j initialization failed: {e}", flush=True)
                self.memory_store = None
        
        # Create learning tools (still used for failure analysis)
        self.failure_tool = create_failure_explanation_tool(self.model)
        self.learning_tool = create_learning_summary_tool(self.model)
        self.adaptation_tool = create_adaptation_tool(self.model)
    
    async def _load_tools(self):
        """Load GitHub and Asana tools - filtered to only what we need."""
        if self.tools is None:
            print("Loading GitHub and Asana tools from Composio...", flush=True)
            # Use Composio with LangchainProvider (matching Seer's approach)
            composio_client = Composio(provider=LangchainProvider())
            user_id = os.getenv("COMPOSIO_USER_ID", "default")
            
            # Get all tools for GitHub and Asana
            all_tools = await asyncio.to_thread(
                composio_client.tools.get,
                user_id=user_id,
                toolkits=["GITHUB", "ASANA"],
                limit=2000
            )
            
            # Filter to only relevant tools for our task:
            # - GitHub: PR operations, issue operations
            # - Asana: Task operations (create, update, search, get)
            relevant_keywords = [
                # GitHub PR operations
                "pull", "pr", "merge", "review",
                # GitHub issue operations  
                "issue",
                # Asana task operations
                "task", "asana"
            ]
            
            filtered_tools = []
            for tool in all_tools:
                tool_name_lower = tool.name.lower()
                tool_desc_lower = (tool.description or "").lower()
                
                # Skip tools with names longer than 64 characters (OpenAI limit)
                if len(tool.name) > 64:
                    continue
                
                # Include if name or description contains relevant keywords
                if any(keyword in tool_name_lower or keyword in tool_desc_lower 
                       for keyword in relevant_keywords):
                    filtered_tools.append(tool)
                    
                    # Stop at 120 tools to stay under OpenAI's 128 limit
                    if len(filtered_tools) >= 120:
                        break
            
            self.tools = filtered_tools
            logger.info(f"Loaded {len(self.tools)} tools (filtered from {len(all_tools)} total)")
            print(f"✓ Loaded {len(self.tools)} tools", flush=True)
            
            if len(self.tools) == 0:
                logger.warning("No tools found! Using first 120 tools with valid names as fallback")
                # Fallback: take first 120 tools with names <= 64 chars
                self.tools = [t for t in all_tools if len(t.name) <= 64][:120]
                
        return self.tools
    
    async def run_condition(
        self, 
        condition: str, 
        task: str, 
        num_attempts: int = 3,
        agent_id: str = "learning_agent"
    ) -> List[ExperimentRun]:
        """Run one condition (with or without memory)."""
        
        runs = []
        base_prompt = """You are an agent that executes tasks using tools. CRITICAL RULES:

1. YOU MUST USE TOOLS - Do not describe plans or provide code. Execute the task using the provided tools.
2. EVIDENCE FIRST - Before making any comment or claim, you must call tools to gather evidence.
3. NO PLANNING - Do not describe what you would do. Actually do it by calling tools.
4. EXPERIMENTS REQUIRED - Your execution must contain tool calls (experiments). Without tool calls, you have not executed anything.
5. ACTUALLY CALL TOOLS - You must use the tool_calls feature to invoke tools. Do not just describe tool usage in text.

When given a task:
- Immediately start calling tools to execute it
- Do not explain your plan first
- Do not ask for clarification unless absolutely necessary
- Use the GitHub and Asana tools provided to actually perform the actions
- You MUST make actual tool calls - use the available tools to search, get, update, and create resources

CRITICAL: You will be evaluated on whether you ACTUALLY CALLED TOOLS, not whether you described calling them.
Success means: You actually called tools and performed the actions, not that you described how to do it."""
        current_prompt = base_prompt
        
        for attempt in range(1, num_attempts + 1):
            logger.info(f"Running {condition}, attempt {attempt}/{num_attempts}")
            
            # If with memory, retrieve failures and update prompt
            if condition == "with_memory" and attempt > 1:
                failures = [r for r in runs if not r.success][-3:]
                
                if failures:
                    # Get failure summaries
                    failure_summaries = "\n".join([
                        f"Attempt {f.attempt_number}: {f.failure_explanation.failure_reason if f.failure_explanation else 'Unknown error'}"
                        for f in failures
                    ])
                    
                    # Get learning summary
                    learnings_json = await self.learning_tool.ainvoke({
                        "last_n_failures": len(failures),
                        "failure_summaries": failure_summaries
                    })
                    learnings = LearningSummary.model_validate_json(learnings_json)
                    
                    # Get adaptation plan
                    adaptation_json = await self.adaptation_tool.ainvoke({
                        "learnings": learnings_json,
                        "next_attempt": f"Attempt {attempt} of {task}"
                    })
                    adaptation = AdaptationPlan.model_validate_json(adaptation_json)
                    
                    # Update prompt with learnings
                    current_prompt = f"""{base_prompt}

Based on your previous failures, you learned:
{chr(10).join('- ' + l for l in learnings.key_learnings)}

You will adapt by:
{chr(10).join('- ' + c for c in adaptation.changes_to_make)}

Your approach will differ: {adaptation.approach_differences}
"""
            
            # Load tools
            tools = await self._load_tools()
            
            # Execute task with Reflexion + Neo4j memory if "with_memory" condition
            memory_store = self.memory_store if condition == "with_memory" else None
            agent_id = f"{condition}_agent_{attempt}"
            
            result = await execute_task(
                task, 
                current_prompt, 
                tools, 
                self.model,
                memory_store=memory_store,
                agent_id=agent_id
            )
            
            # If failed, get failure explanation
            failure_explanation = None
            if not result["success"]:
                explanation_json = await self.failure_tool.ainvoke({
                    "task": task,
                    "attempt_output": result["output"],
                    "expected_output": "Successfully completed task",
                    "error_message": result.get("error", "")
                })
                failure_explanation = FailureExplanation.model_validate_json(explanation_json)
            
            # Create run record
            run = ExperimentRun(
                run_id=str(uuid4()),
                condition=condition,
                attempt_number=attempt,
                success=result["success"],
                failure_explanation=failure_explanation,
                prompt_version=current_prompt[:100] + "..." if len(current_prompt) > 100 else current_prompt,
                execution_time=result["execution_time"],
                output=result["output"],
                tool_calls_made=result.get("tool_calls_made", False),
                tool_call_count=result.get("tool_call_count", 0),
                timestamp=datetime.now()
            )
            
            runs.append(run)
            self.results.append(run)
            
            tool_info = f" ({result.get('tool_call_count', 0)} tool calls)" if result.get('tool_calls_made') else " (NO TOOLS CALLED)"
            logger.info(f"Attempt {attempt}: {'✅ Success' if result['success'] else '❌ Failed'}{tool_info}")
            print(f"Attempt {attempt}: {'✅ Success' if result['success'] else '❌ Failed'}{tool_info}", flush=True)
        
        return runs
    
    async def run_experiment(self, task: str, num_attempts: int = 3) -> Dict[str, Any]:
        """Run full experiment (both conditions)."""
        
        logger.info("Starting Experiment E12: Self-Improving Agent")
        print("\n" + "="*60, flush=True)
        print("Running WITH MEMORY condition...", flush=True)
        print("="*60, flush=True)
        with_memory_runs = await self.run_condition("with_memory", task, num_attempts, "learning_agent")
        
        # Run without memory
        print("\n" + "="*60, flush=True)
        print("Running WITHOUT MEMORY condition...", flush=True)
        print("="*60, flush=True)
        logger.info("Running WITHOUT MEMORY condition...")
        without_memory_runs = await self.run_condition("without_memory", task, num_attempts, "baseline_agent")
        
        # Calculate metrics
        metrics = self._calculate_metrics(with_memory_runs, without_memory_runs)
        
        return {
            "with_memory": [r.model_dump() for r in with_memory_runs],
            "without_memory": [r.model_dump() for r in without_memory_runs],
            "metrics": metrics
        }
    
    def _calculate_metrics(self, with_memory: List[ExperimentRun], without_memory: List[ExperimentRun]) -> Dict[str, Any]:
        """Calculate experiment metrics."""
        def success_rate_by_attempt(runs: List[ExperimentRun]) -> List[float]:
            rates = []
            for attempt in range(1, len(runs) + 1):
                attempt_runs = [r for r in runs if r.attempt_number == attempt]
                if attempt_runs:
                    success_count = sum(1 for r in attempt_runs if r.success)
                    rates.append(success_count / len(attempt_runs))
                else:
                    rates.append(0.0)
            return rates
        
        with_memory_rates = success_rate_by_attempt(with_memory)
        without_memory_rates = success_rate_by_attempt(without_memory)
        
        return {
            "with_memory_success_rates": with_memory_rates,
            "without_memory_success_rates": without_memory_rates,
            "improvement_trend": {
                "with_memory": with_memory_rates[-1] - with_memory_rates[0],
                "without_memory": without_memory_rates[-1] - without_memory_rates[0]
            }
        }

# ============================================================================
# Main
# ============================================================================

async def main():
    """Run the experiment."""
    import sys
    
    # Allow num_attempts to be passed as command line arg, default to 1 for quick test
    num_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    print("="*60, flush=True)
    print("Starting Experiment E12: Self-Improving Agent", flush=True)
    print("Model: GPT-5.1", flush=True)
    print(f"Attempts: {num_attempts} per condition ({num_attempts * 2} total)", flush=True)
    print("="*60, flush=True)
    
    experiment = SelfImprovingAgentExperiment()
    
    # Task: GitHub + Asana integration
    task = """When a GitHub PR is merged in repository seer-engg/buggy-coder:
1. Find related Asana tasks (search by PR title/keywords)
2. Update the Asana tasks with PR details (link, author, merge date)
3. Close the Asana tasks

Success criteria:
- Asana tasks found and updated
- Tasks closed successfully
- PR details correctly added to tasks"""
    
    # Run experiment
    results = await experiment.run_experiment(task, num_attempts=num_attempts)
    
    # Save results
    output_dir = Path(__file__).parent / "results_e12"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"e12_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT E12 RESULTS")
    print("="*60)
    print(f"\nWith Memory Success Rates: {results['metrics']['with_memory_success_rates']}")
    print(f"Without Memory Success Rates: {results['metrics']['without_memory_success_rates']}")
    print(f"\nImprovement Trend:")
    print(f"  With Memory: {results['metrics']['improvement_trend']['with_memory']:.2%}")
    print(f"  Without Memory: {results['metrics']['improvement_trend']['without_memory']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())

