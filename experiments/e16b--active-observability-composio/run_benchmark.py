import asyncio
import sys
import json
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Setup logging with timestamped log file
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'experiment_{log_timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

# Tool call tracking logger
tool_call_logger = logging.getLogger("tool_calls")
tool_call_logger.setLevel(logging.INFO)
tool_call_handler = logging.FileHandler(f'tool_calls_{log_timestamp}.log')
tool_call_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
tool_call_logger.addHandler(tool_call_handler)
tool_call_logger.propagate = False  # Don't propagate to root logger

logger.info(f"📝 Logging to: {log_file}")
logger.info(f"🔧 Tool calls logging to: tool_calls_{log_timestamp}.log")

# Add project root
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))
load_dotenv(project_root / ".env", override=True)

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

# Import Agents
# from agents.baseline_react import create_baseline_agent
from agents.supervisor import create_supervisor

# Import Models
from models import EvaluationResult, BenchmarkResult

# Task Definition
GITHUB_ASANA_TASK = """ACTION REQUIRED: Sync a GitHub PR to Asana.

EXECUTE THESE STEPS IN ORDER:
1. Search GitHub for the most recently merged/closed PR in repository 'seer-engg/buggy-coder'.
2. Extract PR details: title, URL, author, merge/close date.
3. OPTIONALLY search Asana for tasks matching the PR title or keywords (if search fails or finds nothing, proceed to step 4).
4. DECISION:
   - IF task EXISTS (from step 3): Update the task with PR details (add comment with PR URL, author, date).
   - IF task DOES NOT EXIST (search found nothing or search was skipped): Create a new Asana task with PR title and details.
5. Close the Asana task (whether it was updated or newly created).

NOTE: Searching for existing tasks is OPTIONAL. If search fails or exhausts tool calls, proceed directly to creating a new task with PR details.
"""

async def evaluate_success(messages: list, output: str) -> EvaluationResult:
    """Evaluate success using an LLM with structured output."""
    evaluator = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    
    tool_calls_made = any(isinstance(m, AIMessage) and m.tool_calls for m in messages)
    
    prompt = f"""You are evaluating whether an agent successfully completed a GitHub->Asana sync task.

TASK:
{GITHUB_ASANA_TASK}

AGENT'S EXECUTION OUTPUT:
{output[:5000]}

TOOL CALLS MADE: {tool_calls_made}

CRITICAL EVALUATION RULES:
1. The agent MUST have performed actions (found PR, created/updated task).
2. The final output must indicate success.
3. If the agent asks for user input or says "I cannot", it failed.
"""

    try:
        # Use structured output to get EvaluationResult directly
        evaluator_structured = evaluator.with_structured_output(EvaluationResult)
        result = await evaluator_structured.ainvoke([HumanMessage(content=prompt)])
        return result
    except Exception as e:
        print(f"Evaluation failed: {e}")
        # Return default failure result
        return EvaluationResult(
            success=False,
            reasoning=f"Evaluation failed: {str(e)}",
            confidence=0.0
        )

from langgraph.store.memory import InMemoryStore

async def run_condition(condition: str, task: str) -> BenchmarkResult:
    # 1. Setup Agent
    logger.info("Setting up agent...")
    if condition == "baseline":
        # logger.info("Creating baseline agent...")
        # agent = create_baseline_agent()
        # logger.info("✓ Baseline agent created")
        raise ValueError("Baseline agent is deprecated and removed.")
    else:
        logger.info("Creating supervisor agent with Shared Memory Store...")
        store = InMemoryStore() # Create Shared Memory
        agent = create_supervisor(store=store) # Pass to supervisor
        logger.info("✓ Supervisor agent created")
        
    # 2. Setup Tracing
    trace_id = str(uuid.uuid4())
    logger.info(f"🔗 Trace ID: {trace_id}")
    print(f"🔗 Trace ID: {trace_id}")
    
    logger.info("Setting up Langfuse tracing...")
    trace_context = TraceContext(
        session_id=trace_id,
        user_id=f"e16b-benchmark-{condition}",
        tags=["e16b", "benchmark", condition]
    )
    langfuse_handler = CallbackHandler(trace_context=trace_context)
    logger.info("✓ Langfuse handler created")
    
    start_time = time.time()
    
    try:
        # 3. Execute
        logger.info("Preparing execution config...")
        config = {
            "recursion_limit": 10,  # STRICT LIMIT to prevent excessive costs
            "callbacks": [langfuse_handler],
            "configurable": {"thread_id": trace_id}
        }
        logger.info(f"Config: recursion_limit=10, callbacks=[LangfuseHandler], thread_id={trace_id}")
        
        # Initialize state with todos and tool call counters
        logger.info("Initializing agent state...")
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "todos": [],
            "tool_call_counts": {"_total": 0}
        }
        logger.info(f"Initial state: {len(initial_state['messages'])} messages, todos={len(initial_state['todos'])} items")
        
        logger.info("🚀 Starting agent execution...")
        print("🚀 Starting agent execution...")
        result = await agent.ainvoke(initial_state, config=config)
        logger.info("✓ Agent execution completed")
        
        execution_time = time.time() - start_time
        logger.info(f"Execution completed in {execution_time:.2f}s")
        
        messages = result.get("messages", [])
        logger.info(f"Result contains {len(messages)} messages")
        logger.info(f"Result keys: {list(result.keys())}")
        
        final_output = messages[-1].content if messages else ""
        logger.info(f"Final output length: {len(final_output)} chars")
        
        # 4. Metrics
        tool_calls = sum(
            len(m.tool_calls) 
            for m in messages 
            if isinstance(m, AIMessage) and hasattr(m, 'tool_calls')
        )
        logger.info(f"Total tool calls made: {tool_calls}")
        
        # Check tool call counts from state
        tool_call_counts = result.get("tool_call_counts", {})
        logger.info(f"Tool call counts: {tool_call_counts}")
        
        # Rough token estimation (chars / 4)
        total_chars = sum(len(str(m.content)) for m in messages)
        context_size_estimate = total_chars  # This is peak context for the final turn
        logger.info(f"Context size estimate: {context_size_estimate} chars (~{int(context_size_estimate/4)} tokens)")
        
        # 5. Evaluation
        logger.info("🔍 Evaluating Success...")
        print("\n🔍 Evaluating Success...")
        eval_result = await evaluate_success(messages, final_output)
        
        logger.info(f"Evaluation result: success={eval_result.success}, confidence={eval_result.confidence:.2f}")
        print(f"   Result: {'✅ Success' if eval_result.success else '❌ Failed'}")
        print(f"   Reasoning: {eval_result.reasoning}")
        print(f"   Confidence: {eval_result.confidence:.2f}")
        print(f"   ⏱️  Time: {execution_time:.2f}s")
        print(f"   🧠 Context Size: ~{context_size_estimate} chars")
        
        return BenchmarkResult(
            condition=condition,  # type: ignore
            success=eval_result.success,
            execution_time=execution_time,
            total_tokens_estimate=int(context_size_estimate / 4),
            context_size_estimate=context_size_estimate,
            tool_calls=tool_calls,
            reasoning=eval_result.reasoning,
            trace_id=trace_id
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ Run failed after {execution_time:.2f}s: {e}", exc_info=True)
        print(f"❌ Run failed: {e}")
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            condition=condition,
            success=False,
            execution_time=time.time() - start_time,
            total_tokens_estimate=0,
            context_size_estimate=0,
            tool_calls=0,
            reasoning=f"Exception: {str(e)}",
            trace_id=trace_id
        )

async def main():
    logger.info("="*60)
    logger.info("🚀 Starting E16B Experiment: Baseline Delegate Only")
    logger.info("="*60)
    print("🚀 Starting E16B Experiment: Baseline Delegate Only")
    
    results = []
    
    # Run Baseline Delegate only
    logger.info("Starting baseline_delegate condition...")
    results.append(await run_condition("baseline_delegate", GITHUB_ASANA_TASK))
    logger.info("✓ Baseline delegate condition completed")
    
    # Save Results
    timestamp = int(time.time())
    results_file = Path(__file__).parent / f"results/benchmark_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
        
    print(f"\n💾 Results saved to: {results_file}")
    
    # Results Summary
    print("\n📊 RESULTS SUMMARY:")
    result = results[0]
    print(f"   Condition: {result.condition}")
    print(f"   Success: {'✅ Yes' if result.success else '❌ No'}")
    print(f"   Execution Time: {result.execution_time:.2f}s")
    print(f"   Context Size: ~{result.context_size_estimate} chars (~{result.total_tokens_estimate} tokens)")
    print(f"   Tool Calls: {result.tool_calls}")
    print(f"   Trace ID: {result.trace_id}")
    if result.reasoning:
        print(f"   Reasoning: {result.reasoning[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())

