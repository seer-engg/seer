"""
Experiment E10: Reasoning Effort Impact on Plan Quality

Tests whether varying reasoning_effort (minimal/medium/high) in eval agent's planning
node improves plan quality and execution success. Generates plans, ranks them with
LLM-as-judge, then executes with target agent to validate ranking.
"""
import os
import json
import asyncio
import time
import sys
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tiktoken

# Load environment variables FIRST, before any imports
load_dotenv()

# Set required env vars if missing (for plan-only mode, some may not be needed)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Configure MCP loading
os.environ["EVAL_AGENT_LOAD_DEFAULT_MCPS"] = "true"  # Need MCPs for execution
if not os.getenv("CONTEXT7_API_KEY"):
    os.environ["CONTEXT7_API_KEY"] = ""  # Set empty string to avoid None error

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Add seer root to path
seer_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(seer_root))

from agents.eval_agent.graph import graph
from agents.eval_agent.models import EvalAgentState
from shared.schema import AgentContext, UserContext, GithubContext
from shared.config import config
from shared.llm import get_llm

# ============================================================================
# Configuration
# ============================================================================

TEST_TASK = "Get all Asana tasks from the project. List all tasks with their names and statuses."

REASONING_EFFORT_LEVELS = ["minimal", "medium", "high"]

RESULTS_DIR = Path(__file__).parent / "results_e10"
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize tokenizer for plan length measurement
try:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
except:
    tokenizer = tiktoken.get_encoding("cl100k_base")

# ============================================================================
# Ranking Model
# ============================================================================

class PlanRanking(BaseModel):
    """LLM-as-judge ranking of plans."""
    ranking: List[str] = Field(description="Ranking of plans from best (1st) to worst (3rd). Values: 'minimal', 'medium', 'high'")
    reasoning: str = Field(description="Explanation of the ranking")
    best_plan_qualities: str = Field(description="Key qualities that make the best plan superior (as a single text description)")
    worst_plan_weaknesses: str = Field(description="Key weaknesses of the worst plan (as a single text description)")


# ============================================================================
# Helper Functions
# ============================================================================

def extract_plan_metrics(plan: List[Any]) -> Dict[str, Any]:
    """Extract complexity metrics from a plan."""
    if not plan:
        return {
            "num_test_cases": 0,
            "total_instructions_create": 0,
            "total_instructions_assert": 0,
            "num_services": 0,
            "plan_text_length": 0,
            "plan_token_count": 0,
            "avg_instructions_per_test": 0,
        }
    
    total_instructions_create = 0
    total_instructions_assert = 0
    services_mentioned = set()
    plan_text_parts = []
    
    for example in plan:
        # Handle both Pydantic models and dicts
        if isinstance(example, dict):
            expected_output = example.get("expected_output", {})
        else:
            expected_output = getattr(example, "expected_output", None)
            if expected_output and hasattr(expected_output, "model_dump"):
                expected_output = expected_output.model_dump()
        
        if expected_output:
            # Count create_test_data instructions
            create_data = expected_output.get("create_test_data", [])
            if isinstance(create_data, list):
                for service_inst in create_data:
                    if isinstance(service_inst, dict):
                        service_name = service_inst.get("service_name", "")
                        instructions = service_inst.get("instructions", [])
                    else:
                        service_name = getattr(service_inst, "service_name", "")
                        instructions = getattr(service_inst, "instructions", [])
                    
                    if service_name:
                        services_mentioned.add(service_name)
                    if isinstance(instructions, list):
                        total_instructions_create += len(instructions)
            
            # Count assert_final_state instructions
            assert_data = expected_output.get("assert_final_state", [])
            if isinstance(assert_data, list):
                for service_inst in assert_data:
                    if isinstance(service_inst, dict):
                        service_name = service_inst.get("service_name", "")
                        instructions = service_inst.get("instructions", [])
                    else:
                        service_name = getattr(service_inst, "service_name", "")
                        instructions = getattr(service_inst, "instructions", [])
                    
                    if service_name:
                        services_mentioned.add(service_name)
                    if isinstance(instructions, list):
                        total_instructions_assert += len(instructions)
        
        # Collect text for length measurement
        if isinstance(example, dict):
            plan_text_parts.append(str(example.get("input_message", "")))
            plan_text_parts.append(str(expected_output))
        else:
            plan_text_parts.append(str(getattr(example, "input_message", "")))
            plan_text_parts.append(str(expected_output))
    
    plan_text = "\n".join(plan_text_parts)
    plan_text_length = len(plan_text)
    
    try:
        plan_token_count = len(tokenizer.encode(plan_text))
    except:
        plan_token_count = plan_text_length // 4  # Rough estimate
    
    num_tests = len(plan)
    avg_instructions = (total_instructions_create + total_instructions_assert) / num_tests if num_tests > 0 else 0
    
    return {
        "num_test_cases": num_tests,
        "total_instructions_create": total_instructions_create,
        "total_instructions_assert": total_instructions_assert,
        "total_instructions": total_instructions_create + total_instructions_assert,
        "num_services": len(services_mentioned),
        "plan_text_length": plan_text_length,
        "plan_token_count": plan_token_count,
        "avg_instructions_per_test": avg_instructions,
    }


def format_plan_for_ranking(plan: List[Any], reasoning_level: str) -> str:
    """Format a plan (dataset_examples) for ranking."""
    plan_text = f"=== PLAN (Reasoning: {reasoning_level.upper()}) ===\n\n"
    
    if not plan:
        plan_text += "No test cases generated.\n"
    else:
        for i, example in enumerate(plan, 1):
            plan_text += f"Test Case {i}:\n"
            # Handle both Pydantic models and dicts
            if hasattr(example, 'input_message'):
                plan_text += f"  Input: {example.input_message}\n"
            elif isinstance(example, dict):
                plan_text += f"  Input: {example.get('input_message', 'N/A')}\n"
            
            if hasattr(example, 'expected_output') and example.expected_output:
                plan_text += f"  Expected Output: {example.expected_output}\n"
            elif isinstance(example, dict) and example.get('expected_output'):
                plan_text += f"  Expected Output: {example['expected_output']}\n"
            
            if hasattr(example, 'assertion') and example.assertion:
                plan_text += f"  Assertion: {example.assertion}\n"
            elif isinstance(example, dict) and example.get('assertion'):
                plan_text += f"  Assertion: {example['assertion']}\n"
            
            plan_text += "\n"
    
    return plan_text


async def rank_plans(
    plans: Dict[str, List[Any]],
    task: str,
    model
) -> PlanRanking:
    """Use LLM-as-judge to rank all plans together."""
    
    ranking_prompt = f"""You are evaluating three plans generated for the same task. Your job is to rank them from best to worst.

TASK:
{task}

PLANS:

{format_plan_for_ranking(plans['minimal'], 'minimal')}

{format_plan_for_ranking(plans['medium'], 'medium')}

{format_plan_for_ranking(plans['high'], 'high')}

Evaluate these plans based on:
1. Completeness: Does the plan address all requirements?
2. Quality: Are the test cases well-designed and comprehensive?
3. Feasibility: Can the plan be executed successfully?
4. Edge Cases: Does the plan consider edge cases and error scenarios?
5. Clarity: Is the plan clear and well-structured?

Rank the plans from best (1st) to worst (3rd) based on reasoning effort level ('minimal', 'medium', 'high').
Provide reasoning for your ranking."""

    ranking_llm = model.with_structured_output(PlanRanking, method="function_calling", strict=True)
    
    result = await ranking_llm.ainvoke([HumanMessage(content=ranking_prompt)])
    return result


async def generate_plan_with_reasoning_effort(
    reasoning_effort: str,
    task: str,
    user_id: str = "experiment_user"
) -> Dict[str, Any]:
    """Generate a plan using eval agent with specified reasoning effort."""
    
    print(f"\n{'='*60}")
    print(f"Generating plan with reasoning_effort: {reasoning_effort.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Temporarily override config
    original_reasoning_effort = config.eval_reasoning_effort
    original_plan_only = config.eval_plan_only_mode
    
    try:
        # Set reasoning effort (plan-only mode is False to execute)
        config.eval_reasoning_effort = reasoning_effort
        config.eval_plan_only_mode = False  # Execute plans with target agent
        
        # Format message for eval agent (it expects GitHub context)
        # Use a real GitHub repo for provisioning (even though we won't execute)
        eval_message = f"""Evaluate my agent test_agent at https://github.com/seer-engg/buggy-coder

{task}"""
        
        # Create initial state with minimal context for plan-only mode
        user_context = UserContext(
            raw_request=eval_message,
            user_id=user_id
        )
        
        github_context = GithubContext(
            repo_url="https://github.com/seer-engg/buggy-coder",
            branch_name="main"
        )
        
        agent_context = AgentContext(
            user_context=user_context,
            github_context=github_context,
            agent_name="test_agent",
            mcp_services=["asana", "github"],  # Mentioned in task
        )
        
        # Use full eval agent graph (includes execution)
        initial_state = EvalAgentState(
            context=agent_context,
            messages=[HumanMessage(content=eval_message)],
        )
        
        # Invoke full eval agent graph (plan + execute)
        print(f"  Invoking eval agent graph (plan + execute)...")
        result = await graph.ainvoke(initial_state)
        
        elapsed = time.time() - start_time
        
        plan = result.get("dataset_examples", [])
        num_tests = len(plan)
        
        print(f"  ✅ Generated {num_tests} test cases in {elapsed:.1f}s")
        
        return {
            "reasoning_effort": reasoning_effort,
            "plan": plan,
            "num_tests": num_tests,
            "time": elapsed,
            "state": result
        }
        
    finally:
        # Restore original config
        config.eval_reasoning_effort = original_reasoning_effort
        config.eval_plan_only_mode = original_plan_only


# ============================================================================
# Main Experiment
# ============================================================================

async def main():
    """Run the reasoning effort experiment."""
    print("="*60)
    print("EXPERIMENT E10: REASONING EFFORT IMPACT ON PLAN QUALITY")
    print("="*60)
    print(f"Task: {TEST_TASK}")
    print(f"Reasoning Levels: {', '.join(REASONING_EFFORT_LEVELS)}")
    print(f"Mode: Full execution (plan + execute)")
    print("="*60)
    
    # Initialize model for ranking
    ranking_model = get_llm(
        model="gpt-5.1",
        temperature=0.0,
        reasoning_effort="medium"  # Use medium for ranking
    )
    
    # Generate and execute plans with different reasoning efforts
    results = {}
    plans = {}
    execution_results = {}
    
    for reasoning_effort in REASONING_EFFORT_LEVELS:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {reasoning_effort.upper()} REASONING")
        print(f"{'='*60}")
        
        result = await generate_plan_with_reasoning_effort(
            reasoning_effort=reasoning_effort,
            task=TEST_TASK
        )
        results[reasoning_effort] = result
        plans[reasoning_effort] = result["plan"]
        
        # Extract execution results from state if available
        state = result.get("state", {})
        latest_results = state.get("latest_results", [])
        if latest_results:
            # Extract provisioning verification results
            # Provisioning succeeded if:
            # 1. Result passed (score >= threshold), OR
            # 2. Result has actual_output (target agent was invoked), OR  
            # 3. Judge reasoning doesn't mention "Provisioning verification failed"
            provisioning_successes = []
            provisioning_failures = []
            
            for r in latest_results:
                # Handle both Pydantic models and dicts
                if isinstance(r, dict):
                    passed = r.get("passed", False)
                    actual_output = r.get("actual_output", "")
                    judge_reasoning = r.get("judge_reasoning", "") or r.get("analysis", {}).get("judge_reasoning", "")
                else:
                    passed = getattr(r, 'passed', False) if hasattr(r, 'passed') else False
                    actual_output = getattr(r, 'actual_output', "") if hasattr(r, 'actual_output') else ""
                    judge_reasoning = getattr(r, 'judge_reasoning', "") if hasattr(r, 'judge_reasoning') else ""
                    if not judge_reasoning and hasattr(r, 'analysis'):
                        judge_reasoning = getattr(r.analysis, 'judge_reasoning', "")
                
                # Check if provisioning failed (indicated by "Provisioning verification failed" in reasoning)
                is_provisioning_failure = "Provisioning verification failed" in str(judge_reasoning)
                
                if is_provisioning_failure:
                    provisioning_failures.append(r)
                else:
                    # Provisioning succeeded (target agent was invoked)
                    provisioning_successes.append(r)
            
            execution_results[reasoning_effort] = {
                "num_tests": len(latest_results),
                "provisioning_succeeded": len(provisioning_successes),
                "provisioning_failed": len(provisioning_failures),
                "passed": len(provisioning_successes),  # Provisioning success = passed
                "failed": len(provisioning_failures),
                "results": latest_results
            }
    
    # Rank plans using LLM-as-judge
    print("\n" + "="*60)
    print("RANKING PLANS WITH LLM-AS-JUDGE")
    print("="*60)
    
    ranking = await rank_plans(plans, TEST_TASK, ranking_model)
    
    print(f"\nRanking (best to worst):")
    for i, level in enumerate(ranking.ranking, 1):
        print(f"  {i}. {level.upper()}")
    
    print(f"\nReasoning:")
    print(f"  {ranking.reasoning}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for level in REASONING_EFFORT_LEVELS:
        result = results[level]
        rank = ranking.ranking.index(level) + 1
        print(f"{level.upper()}:")
        print(f"  Test Cases: {result['num_tests']}")
        print(f"  Time: {result['time']:.1f}s")
        print(f"  Rank: {rank}")
    
    print(f"\nBest Plan (Ranking): {ranking.ranking[0].upper()}")
    print(f"Worst Plan (Ranking): {ranking.ranking[-1].upper()}")
    
    # Provisioning verification results (eval agent isolation)
    if execution_results:
        print("\n" + "="*60)
        print("PROVISIONING VERIFICATION RESULTS (Eval Agent Isolation)")
        print("="*60)
        print("This measures whether the eval agent can follow its own plan")
        print("to provision the environment correctly, independent of target agent.")
        print("="*60)
        for level in REASONING_EFFORT_LEVELS:
            if level in execution_results:
                exec_result = execution_results[level]
                provisioning_success_rate = (exec_result["provisioning_succeeded"] / exec_result["num_tests"] * 100) if exec_result["num_tests"] > 0 else 0
                print(f"{level.upper()}:")
                print(f"  Tests: {exec_result['num_tests']}")
                print(f"  Provisioning Succeeded: {exec_result['provisioning_succeeded']}")
                print(f"  Provisioning Failed: {exec_result['provisioning_failed']}")
                print(f"  Provisioning Success Rate: {provisioning_success_rate:.1f}%")
                if exec_result["provisioning_failed"] > 0:
                    print(f"  ⚠️  {exec_result['provisioning_failed']} test(s) failed provisioning - target agent was not invoked")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"e10_results_{timestamp}.json"
    
    # Convert plans to serializable format (extract key fields)
    serializable_results = {}
    for level, result in results.items():
        plan_data = []
        for example in result["plan"]:
            if hasattr(example, 'model_dump'):
                plan_data.append(example.model_dump())
            elif isinstance(example, dict):
                plan_data.append(example)
            else:
                # Fallback: extract common fields
                plan_data.append({
                    "input_message": getattr(example, 'input_message', str(example)),
                    "expected_output": getattr(example, 'expected_output', None),
                    "assertion": getattr(example, 'assertion', None),
                })
        
        # Extract plan metrics
        plan_metrics = extract_plan_metrics(result["plan"])
        
        serializable_results[level] = {
            "reasoning_effort": result["reasoning_effort"],
            "num_tests": result["num_tests"],
            "time": result["time"],
            "plan": plan_data,
            "metrics": plan_metrics  # Add metrics to results
        }
    
    # Serialize execution results (focus on provisioning verification)
    serializable_execution = {}
    for level, exec_result in execution_results.items():
        serializable_execution[level] = {
            "num_tests": exec_result["num_tests"],
            "provisioning_succeeded": exec_result["provisioning_succeeded"],
            "provisioning_failed": exec_result["provisioning_failed"],
            "provisioning_success_rate": (exec_result["provisioning_succeeded"] / exec_result["num_tests"] * 100) if exec_result["num_tests"] > 0 else 0,
            # Legacy fields for compatibility
            "passed": exec_result["passed"],
            "failed": exec_result["failed"],
            "success_rate": (exec_result["passed"] / exec_result["num_tests"] * 100) if exec_result["num_tests"] > 0 else 0
        }
    
    output = {
        "experiment": "e10_reasoning_effort",
        "timestamp": timestamp,
        "task": TEST_TASK,
        "results": serializable_results,
        "ranking": {
            "ranking": ranking.ranking,
            "reasoning": ranking.reasoning,
            "best_plan_qualities": ranking.best_plan_qualities,
            "worst_plan_weaknesses": ranking.worst_plan_weaknesses
        },
        "execution": serializable_execution
    }
    
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        generate_visualizations(output, results_file)
        print("✅ Visualizations saved")
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return output


def generate_visualizations(results: Dict[str, Any], results_file: Path):
    """Generate comparison visualizations for the experiment results."""
    
    # Extract metrics for each reasoning level
    metrics_data = []
    for level in REASONING_EFFORT_LEVELS:
        result = results["results"][level]
        plan_metrics = extract_plan_metrics(result["plan"])
        
        # Get ranking
        rank = results["ranking"]["ranking"].index(level) + 1
        
        # Get execution results if available
        exec_data = results.get("execution", {}).get(level, {})
        
        metrics_data.append({
            "Reasoning Effort": level.upper(),
            "Rank": rank,
            "Time (s)": result["time"],
            "Test Cases": result["num_tests"],
            "Total Instructions": plan_metrics["total_instructions"],
            "Create Instructions": plan_metrics["total_instructions_create"],
            "Assert Instructions": plan_metrics["total_instructions_assert"],
            "Services": plan_metrics["num_services"],
            "Plan Length (chars)": plan_metrics["plan_text_length"],
            "Plan Tokens": plan_metrics["plan_token_count"],
            "Avg Instructions/Test": plan_metrics["avg_instructions_per_test"],
            "Provisioning Success Rate": exec_data.get("provisioning_success_rate", 0),
            "Provisioning Succeeded": exec_data.get("provisioning_succeeded", 0),
            "Provisioning Failed": exec_data.get("provisioning_failed", 0),
            # Legacy fields
            "Success Rate": exec_data.get("success_rate", 0),
            "Tests Passed": exec_data.get("passed", 0),
            "Tests Failed": exec_data.get("failed", 0),
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color palette
    colors = {"MINIMAL": "#e74c3c", "MEDIUM": "#f39c12", "HIGH": "#27ae60"}
    level_colors = [colors[row["Reasoning Effort"]] for _, row in df.iterrows()]
    
    # 1. Ranking comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(df["Reasoning Effort"], df["Rank"], color=level_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel("Rank (1=Best)", fontweight='bold')
    ax1.set_title("Plan Quality Ranking", fontweight='bold', fontsize=12)
    ax1.set_ylim([0, 4])
    ax1.invert_yaxis()  # Lower rank number = better
    for i, (bar, rank) in enumerate(zip(bars, df["Rank"])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f"#{rank}", ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Time comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(df["Reasoning Effort"], df["Time (s)"], color=level_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel("Time (seconds)", fontweight='bold')
    ax2.set_title("Plan Generation Time", fontweight='bold', fontsize=12)
    for bar, time_val in zip(bars, df["Time (s)"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f"{time_val:.1f}s", ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Plan complexity (instructions)
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(df))
    width = 0.35
    bars1 = ax3.bar(x - width/2, df["Create Instructions"], width, label='Create', color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax3.bar(x + width/2, df["Assert Instructions"], width, label='Assert', color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.set_ylabel("Number of Instructions", fontweight='bold')
    ax3.set_title("Plan Complexity: Instructions", fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["Reasoning Effort"])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Plan size (tokens)
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(df["Reasoning Effort"], df["Plan Tokens"], color=level_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel("Tokens", fontweight='bold')
    ax4.set_title("Plan Size (Tokens)", fontweight='bold', fontsize=12)
    for bar, tokens in zip(bars, df["Plan Tokens"]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f"{tokens:,}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Services used
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.bar(df["Reasoning Effort"], df["Services"], color=level_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel("Number of Services", fontweight='bold')
    ax5.set_title("Services Referenced", fontweight='bold', fontsize=12)
    ax5.set_ylim([0, max(df["Services"]) + 1])
    for bar, services in zip(bars, df["Services"]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f"{services}", ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Average instructions per test
    ax6 = fig.add_subplot(gs[1, 2])
    bars = ax6.bar(df["Reasoning Effort"], df["Avg Instructions/Test"], color=level_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel("Avg Instructions per Test", fontweight='bold')
    ax6.set_title("Plan Detail Level", fontweight='bold', fontsize=12)
    for bar, avg in zip(bars, df["Avg Instructions/Test"]):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{avg:.1f}", ha='center', va='bottom', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Provisioning success rate (eval agent isolation metric)
    ax7 = fig.add_subplot(gs[2, 0])
    if "Provisioning Success Rate" in df.columns and df["Provisioning Success Rate"].sum() > 0:
        bars = ax7.bar(df["Reasoning Effort"], df["Provisioning Success Rate"], color=level_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax7.set_ylabel("Success Rate (%)", fontweight='bold')
        ax7.set_title("Provisioning Success Rate\n(Eval Agent Isolation)", fontweight='bold', fontsize=12)
        ax7.set_ylim([0, 105])
        for bar, rate in zip(bars, df["Provisioning Success Rate"]):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f"{rate:.1f}%", ha='center', va='bottom', fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
    else:
        ax7.text(0.5, 0.5, "No provisioning data available", 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12, style='italic')
        ax7.set_title("Provisioning Success Rate\n(Eval Agent Isolation)", fontweight='bold', fontsize=12)
    
    # 8. Comprehensive comparison (radar/spider chart would be better, but bar works)
    ax8 = fig.add_subplot(gs[2, 1:])
    # Normalize metrics for comparison
    normalized_df = df.copy()
    for col in ["Time (s)", "Total Instructions", "Plan Tokens", "Avg Instructions/Test"]:
        if normalized_df[col].max() > 0:
            normalized_df[f"{col}_norm"] = normalized_df[col] / normalized_df[col].max()
    
    x_pos = np.arange(len(df))
    width = 0.25
    metrics_to_plot = ["Time (s)_norm", "Total Instructions_norm", "Plan Tokens_norm"]
    metric_labels = ["Time", "Instructions", "Tokens"]
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        offset = (i - 1) * width
        ax8.bar(x_pos + offset, normalized_df[metric], width, label=label, alpha=0.7, edgecolor='black')
    
    ax8.set_ylabel("Normalized Score", fontweight='bold')
    ax8.set_title("Normalized Plan Metrics Comparison", fontweight='bold', fontsize=12)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(df["Reasoning Effort"])
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)
    ax8.set_ylim([0, 1.1])
    
    # Add overall title
    fig.suptitle("E10: Reasoning Effort Impact on Plan Quality", fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    viz_file = results_file.with_suffix('.png')
    plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  📊 Saved visualization to: {viz_file}")
    
    # Also create a summary metrics table image
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    # Use provisioning success rate if available, otherwise fall back to success rate
    success_col = "Provisioning Success Rate" if "Provisioning Success Rate" in df.columns else "Success Rate"
    table_data = df[["Reasoning Effort", "Rank", "Time (s)", "Test Cases", 
                     "Total Instructions", "Plan Tokens", "Services", success_col]].copy()
    table_data.columns = ["Effort", "Rank", "Time (s)", "Tests", "Instructions", "Tokens", "Services", "Provisioning %"]
    table_data = table_data.round(1)
    
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                    rowLabels=table_data.index, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.08, 0.1, 0.08, 0.12, 0.12, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code rows
    for i in range(len(table_data)):
        color = colors[table_data.iloc[i]["Effort"]]
        for j in range(len(table_data.columns)):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    # Header row
    for j in range(len(table_data.columns)):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title("Experiment E10: Summary Metrics", fontsize=14, fontweight='bold', pad=20)
    
    table_file = results_file.with_name(results_file.stem + '_table.png')
    plt.savefig(table_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  📋 Saved metrics table to: {table_file}")
    
    plt.close('all')


if __name__ == "__main__":
    asyncio.run(main())

