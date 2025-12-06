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

# Task Definitions
# Original E16B task (GitHub ↔ Asana)
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

# Tasks from older experiments (E6, E7) for comparison
TEST_TASKS = [
    {
        "task_id": "github_asana",
        "name": "GitHub ↔ Asana Integration",
        "instruction": GITHUB_ASANA_TASK,
        "metric": "PR found and details extracted; Asana task found/created and updated with PR details; Task closed successfully",
        "services": ["GITHUB", "ASANA"],
        "complexity": "complex"
    },
    {
        "task_id": "weekly_summary",
        "name": "Weekly Work Summary",
        "instruction": """Create a weekly work summary: 
1. Get my Google Calendar events for next week
2. Get my GitHub pull requests from last week (all repos)
3. Get my Slack messages from #engineering channel from last week
4. Combine all information into a structured Google Doc
5. Share the document with my manager (look up their email)
6. Send them a Slack notification with the doc link""",
        "metric": "Check for: (1) Google Doc created with all data, (2) Doc shared with manager, (3) Slack notification sent",
        "services": ["GOOGLECALENDAR", "GITHUB", "SLACK", "GOOGLEDOCS"],
        "complexity": "complex"
    },
    {
        "task_id": "email_task_list",
        "name": "Email Task List",
        "instruction": """Find all unread emails from last month that contain 'meeting' or 'urgent', extract any action items or deadlines mentioned, create a prioritized task list in Google Sheets with columns: Task, Deadline, Priority, and send me a summary email with the sheet link.""",
        "metric": "Google Sheet created with tasks and summary email sent",
        "services": ["GMAIL", "GOOGLESHEETS"],
        "complexity": "complex"
    },
    {
        "task_id": "bug_summary",
        "name": "GitHub Bug Summary",
        "instruction": """Find all GitHub issues assigned to me that mention 'bug' or 'error',
check if any have related Slack discussions in #bugs channel,
create a summary document with: issue title, description, and related Slack context,
and post it to #engineering channel with priority tags based on issue labels""",
        "metric": "Check for: (1) Summary doc created, (2) Message posted to #engineering with doc link",
        "services": ["GITHUB", "SLACK", "GOOGLEDOCS"],
        "complexity": "complex"
    },
    {
        "task_id": "telegram_simple",
        "name": "Telegram Message",
        "instruction": "send good morning message to +1 646-371-6198 via telegram",
        "metric": "Check API response 200 OK from Telegram.",
        "services": ["TELEGRAM"],
        "complexity": "simple"
    },
    {
        "task_id": "twitter_trends",
        "name": "Twitter Trends",
        "instruction": "what are the latest trends going on twitter",
        "metric": "Check Twitter API for post results.",
        "services": ["TWITTER"],
        "complexity": "simple"
    },
    {
        "task_id": "meeting_summary",
        "name": "Meeting Summary Document",
        "instruction": "Create a meeting summary document. Get all attendees from my last Google Calendar meeting, fetch their GitHub activity from the past week, create a Google Doc summarizing the meeting and their contributions, and share it with all attendees via email.",
        "metric": "Google Doc created with meeting summary and shared via email with all attendees",
        "services": ["GOOGLECALENDAR", "GITHUB", "GOOGLEDOCS", "GMAIL"],
        "complexity": "complex"
    },
    {
        "task_id": "deployment_summary",
        "name": "Deployment Summary",
        "instruction": "Find all Slack messages in #engineering from last week that mention 'deploy' or 'release', check corresponding GitHub pull requests, create a deployment summary in Google Sheets with columns: Date, PR, Author, Status, and notify the team in Slack.",
        "metric": "Google Sheet created with deployment summary and Slack notification sent",
        "services": ["SLACK", "GITHUB", "GOOGLESHEETS"],
        "complexity": "complex"
    }
]

async def evaluate_success(messages: list, output: str, task: dict) -> EvaluationResult:
    """Evaluate success using an LLM with structured output."""
    evaluator = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    
    tool_calls_made = any(isinstance(m, AIMessage) and m.tool_calls for m in messages)
    
    prompt = f"""You are evaluating whether an agent successfully completed an integration task.

TASK:
{task['instruction']}

SUCCESS METRIC:
{task.get('metric', 'Task completed successfully')}

AGENT'S EXECUTION OUTPUT:
{output[:5000]}

TOOL CALLS MADE: {tool_calls_made}

CRITICAL EVALUATION RULES:
1. The agent MUST have performed actions using tools (not just planning or asking questions).
2. The final output must indicate success or completion.
3. Check if the success metric criteria are met based on the agent's actions.
4. If the agent asks for user input or says "I cannot" without attempting actions, it failed.
5. Consider partial success if some steps were completed but not all.
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

async def run_condition(condition: str, task: dict) -> BenchmarkResult:
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
        task_instruction = task['instruction'] if isinstance(task, dict) else task
        initial_state = {
            "messages": [HumanMessage(content=task_instruction)],
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
        task_dict = task if isinstance(task, dict) else {"instruction": task, "metric": "Task completed successfully"}
        eval_result = await evaluate_success(messages, final_output, task_dict)
        
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
    logger.info("🚀 Starting E16B Experiment: Baseline Delegate with Multiple Tasks")
    logger.info("="*60)
    print("🚀 Starting E16B Experiment: Baseline Delegate with Multiple Tasks")
    print(f"📋 Running {len(TEST_TASKS)} tasks")
    
    results = []
    
    # Run Baseline Delegate on all tasks
    for i, task in enumerate(TEST_TASKS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i}/{len(TEST_TASKS)}: {task['name']} ({task['task_id']})")
        logger.info(f"{'='*60}")
        print(f"\n{'='*60}")
        print(f"Task {i}/{len(TEST_TASKS)}: {task['name']} ({task['task_id']})")
        print(f"Complexity: {task['complexity']}")
        print(f"{'='*60}")
        
        result = await run_condition("baseline_delegate", task)
        # Add task metadata to result
        result_dict = result.model_dump()
        result_dict['task_id'] = task['task_id']
        result_dict['task_name'] = task['name']
        result_dict['task_complexity'] = task['complexity']
        results.append(result_dict)
        
        logger.info(f"✓ Task {i} completed: {'✅ Success' if result.success else '❌ Failed'}")
        print(f"\n✓ Task {i} completed: {'✅ Success' if result.success else '❌ Failed'}")
    
    # Save Results
    timestamp = int(time.time())
    results_file = Path(__file__).parent / f"results/benchmark_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Results saved to: {results_file}")
    
    # Results Summary
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get('success', False))
    success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    print(f"\nOverall: {successful_tasks}/{total_tasks} tasks succeeded ({success_rate:.1f}%)")
    
    # Group by complexity
    complex_results = [r for r in results if r.get('task_complexity') == 'complex']
    simple_results = [r for r in results if r.get('task_complexity') == 'simple']
    
    if complex_results:
        complex_success = sum(1 for r in complex_results if r.get('success', False))
        print(f"Complex tasks: {complex_success}/{len(complex_results)} succeeded ({complex_success/len(complex_results)*100:.1f}%)")
    
    if simple_results:
        simple_success = sum(1 for r in simple_results if r.get('success', False))
        print(f"Simple tasks: {simple_success}/{len(simple_results)} succeeded ({simple_success/len(simple_results)*100:.1f}%)")
    
    # Per-task summary
    print("\nPer-task results:")
    for r in results:
        status = "✅" if r.get('success', False) else "❌"
        print(f"  {status} {r.get('task_name', 'Unknown')}: {r.get('execution_time', 0):.2f}s, {r.get('tool_calls', 0)} tool calls")
    
    # Average metrics
    if results:
        avg_time = sum(r.get('execution_time', 0) for r in results) / len(results)
        avg_tool_calls = sum(r.get('tool_calls', 0) for r in results) / len(results)
        avg_context = sum(r.get('context_size_estimate', 0) for r in results) / len(results)
        print(f"\nAverage metrics:")
        print(f"  Execution Time: {avg_time:.2f}s")
        print(f"  Tool Calls: {avg_tool_calls:.1f}")
        print(f"  Context Size: ~{int(avg_context)} chars (~{int(avg_context/4)} tokens)")

if __name__ == "__main__":
    asyncio.run(main())

