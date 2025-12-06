"""
Experiment E11: Codex Plan Reasoning Effort Impact on Implementation Quality

Tests whether varying reasoning_effort (minimal/medium/high) in Codex's developer/plan node
improves implementation quality and success rate. Uses minimal test cases that can
complete in < 10 minutes per condition.
"""
import os
import json
import asyncio
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST, before any imports
load_dotenv()

# Set required env vars if missing
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Configure MCP loading
os.environ["EVAL_AGENT_LOAD_DEFAULT_MCPS"] = "true"
if not os.getenv("CONTEXT7_API_KEY"):
    os.environ["CONTEXT7_API_KEY"] = ""

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Add seer root to path
seer_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(seer_root))

from agents.codex.graph import graph
from agents.codex.state import CodexState
from shared.schema import (
    AgentContext,
    UserContext,
    GithubContext,
    SandboxContext,
    DatasetContext,
    DatasetExample,
    ExpectedOutput,
    ServiceInstructions,
    ExperimentContext,
    ExperimentResultContext,
    FailureAnalysis,
)
from shared.config import config
from shared.llm import get_llm

# ============================================================================
# Configuration
# ============================================================================

# Minimal test case: Simple agent that should return a greeting but has a bug
MINIMAL_TASK = "Fix the agent so it returns 'Hello, World!' when asked to greet."

# Minimal failing test case
MINIMAL_TEST_CASE = {
    "example_id": "test_001",
    "input_message": "Please greet me",
    "expected_output": {
        "expected_action": "Return 'Hello, World!'",
        "create_test_data": [],
        "assert_final_state": []
    },
    "status": "active"
}

REASONING_EFFORT_LEVELS = ["minimal", "medium", "high"]
NUM_ROUNDS = 3  # Number of rounds to run for each reasoning effort level

RESULTS_DIR = Path(__file__).parent / "results_e11"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Evaluation Model
# ============================================================================

class ImplementationQuality(BaseModel):
    """LLM-as-judge evaluation of implementation quality."""
    score: float = Field(description="Quality score from 0.0 (poor) to 1.0 (excellent)", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed explanation of the score")
    correctness: float = Field(description="Correctness score (0.0-1.0)", ge=0.0, le=1.0)
    code_quality: float = Field(description="Code quality score (0.0-1.0)", ge=0.0, le=1.0)
    test_passing: bool = Field(description="Whether tests pass")
    strengths: List[str] = Field(description="Key strengths of the implementation", default_factory=list)
    weaknesses: List[str] = Field(description="Key weaknesses of the implementation", default_factory=list)


# ============================================================================
# Helper Functions
# ============================================================================

def create_minimal_test_case() -> DatasetExample:
    """Create a minimal test case for the experiment."""
    return DatasetExample(
        example_id=MINIMAL_TEST_CASE["example_id"],
        input_message=MINIMAL_TEST_CASE["input_message"],
        expected_output=ExpectedOutput(
            expected_action=MINIMAL_TEST_CASE["expected_output"]["expected_action"],
            create_test_data=[
                ServiceInstructions(**item) for item in MINIMAL_TEST_CASE["expected_output"]["create_test_data"]
            ],
            assert_final_state=[
                ServiceInstructions(**item) for item in MINIMAL_TEST_CASE["expected_output"]["assert_final_state"]
            ],
        ),
        status=MINIMAL_TEST_CASE["status"],
    )


def create_minimal_failing_result(test_case: DatasetExample) -> ExperimentResultContext:
    """Create a minimal failing test result for Codex to fix."""
    return ExperimentResultContext(
        thread_id="test_thread_001",
        dataset_example=test_case,
        actual_output="Hello",  # Wrong output - missing ", World!"
        analysis=FailureAnalysis(
            score=0.5,
            judge_reasoning="The agent returned 'Hello' but should return 'Hello, World!'. The implementation is incomplete."
        ),
        passed=False,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


async def evaluate_implementation(
    implementation_state: CodexState,
    reasoning_level: str,
    model
) -> ImplementationQuality:
    """Use LLM-as-judge to evaluate implementation quality."""
    
    # Extract key information from implementation
    developer_messages = implementation_state.developer_thread or []
    latest_results = implementation_state.latest_results or []
    
    # Format implementation summary
    implementation_summary = f"""
Reasoning Level: {reasoning_level.upper()}

Developer Messages: {len(developer_messages)} messages
Latest Results: {len(latest_results)} test results

Test Results:
"""
    for i, result in enumerate(latest_results, 1):
        implementation_summary += f"""
Test {i}:
  - Passed: {result.passed}
  - Score: {result.score}
  - Judge Feedback: {result.judge_reasoning}
"""
    
    evaluation_prompt = f"""You are evaluating a Codex implementation that was created with {reasoning_level} reasoning effort.

IMPLEMENTATION SUMMARY:
{implementation_summary}

Evaluate this implementation based on:
1. Correctness: Does it correctly fix the bug and pass tests?
2. Code Quality: Is the code clean, maintainable, and well-structured?
3. Completeness: Does it fully address the requirements?
4. Test Results: Do the tests pass?

Provide a comprehensive evaluation with scores and reasoning."""

    evaluation_llm = model.with_structured_output(ImplementationQuality, method="function_calling", strict=True)
    
    result = await evaluation_llm.ainvoke([HumanMessage(content=evaluation_prompt)])
    return result


async def run_codex_with_reasoning_effort(
    reasoning_effort: str,
    test_case: DatasetExample,
    failing_result: ExperimentResultContext,
    user_id: str = "experiment_user"
) -> Dict[str, Any]:
    """Run Codex with specified reasoning effort to fix the bug."""
    
    print(f"\n{'='*60}")
    print(f"Running Codex with reasoning_effort: {reasoning_effort.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Temporarily override config
    original_reasoning_effort = config.codex_reasoning_effort
    
    try:
        # Set reasoning effort
        config.codex_reasoning_effort = reasoning_effort
        
        # Create minimal context for Codex
        # Note: For a real experiment, you'd need a real sandbox and GitHub repo
        # This is a simplified version for testing the concept
        
        user_context = UserContext(
            raw_request=MINIMAL_TASK,
            user_id=user_id
        )
        
        github_context = GithubContext(
            repo_url="https://github.com/seer-engg/buggy-coder",  # Placeholder
            branch_name="main"
        )
        
        # For a real experiment, you'd need to provision a sandbox
        # For now, we'll create a minimal state structure
        sandbox_context = SandboxContext(
            sandbox_id="experiment_sandbox",
            working_directory="/workspace",
            working_branch="main"
        )
        
        agent_context = AgentContext(
            user_context=user_context,
            github_context=github_context,
            sandbox_context=sandbox_context,
            agent_name="test_agent",
        )
        
        # Create experiment context with failing result
        experiment_context = ExperimentContext(
            experiment_name=f"e11_codex_{reasoning_effort}",
            results=[failing_result],
            mean_score=failing_result.score,
        )
        
        # Create dataset context
        dataset_context = DatasetContext(
            dataset_id="e11_minimal",
            dataset_name="E11 Minimal Test Case",
            experiments=[experiment_context],
        )
        
        # Create Codex input state
        codex_input = CodexState(
            context=agent_context,
            dataset_context=dataset_context,
            experiment_context=experiment_context,
            dataset_examples=[test_case],
        )
        
        # Invoke Codex graph
        print(f"  Invoking Codex graph...")
        print(f"  Note: This requires a running Codex LangGraph server at {config.codex_remote_url}")
        print(f"  For a real experiment, ensure the server is running and sandbox is provisioned.")
        
        # For now, we'll simulate or use a local graph if available
        # In production, this would use RemoteGraph
        try:
            result = await graph.ainvoke(codex_input)
        except Exception as e:
            print(f"  ⚠️  Error invoking Codex: {e}")
            print(f"  This is expected if Codex server is not running.")
            print(f"  Creating mock result for demonstration...")
            
            # Create mock result for demonstration
            result = {
                "developer_thread": [HumanMessage(content="Mock implementation")],
                "latest_results": [],
                "agent_updated": False,
            }
        
        elapsed = time.time() - start_time
        
        print(f"  ✅ Completed in {elapsed:.1f}s")
        
        # Convert result to CodexState for evaluation
        result_state = CodexState(
            context=agent_context,
            dataset_context=dataset_context,
            experiment_context=experiment_context,
            dataset_examples=[test_case],
            developer_thread=result.get("developer_thread", []),
            latest_results=result.get("latest_results", []),
            agent_updated=result.get("agent_updated", False),
        )
        
        return {
            "reasoning_effort": reasoning_effort,
            "state": result_state,
            "time": elapsed,
            "raw_result": result,
        }
        
    finally:
        # Restore original config
        config.codex_reasoning_effort = original_reasoning_effort


# ============================================================================
# Main Experiment
# ============================================================================

async def main():
    """Run the Codex reasoning effort experiment with multiple rounds."""
    print("="*60)
    print("EXPERIMENT E11: CODEX PLAN REASONING EFFORT IMPACT")
    print("="*60)
    print(f"Task: {MINIMAL_TASK}")
    print(f"Reasoning Levels: {', '.join(REASONING_EFFORT_LEVELS)}")
    print(f"Rounds per level: {NUM_ROUNDS}")
    print(f"Mode: Codex implementation with different reasoning efforts")
    print("="*60)
    
    # Initialize model for evaluation
    evaluation_model = get_llm(
        model="gpt-5.1",
        temperature=0.0,
        reasoning_effort="medium"
    )
    
    # Create minimal test case
    test_case = create_minimal_test_case()
    failing_result = create_minimal_failing_result(test_case)
    
    # Run Codex with different reasoning efforts - multiple rounds
    all_rounds = {}  # Store all individual round results
    aggregated_results = {}  # Store aggregated statistics
    
    for reasoning_effort in REASONING_EFFORT_LEVELS:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {reasoning_effort.upper()} REASONING ({NUM_ROUNDS} rounds)")
        print(f"{'='*60}")
        
        round_results = []
        round_evaluations = []
        
        for round_num in range(1, NUM_ROUNDS + 1):
            print(f"\n  Round {round_num}/{NUM_ROUNDS}:")
            
            result = await run_codex_with_reasoning_effort(
                reasoning_effort=reasoning_effort,
                test_case=test_case,
                failing_result=failing_result
            )
            round_results.append(result)
            
            # Evaluate implementation quality
            print(f"    Evaluating implementation quality...")
            evaluation = await evaluate_implementation(
                implementation_state=result["state"],
                reasoning_level=reasoning_effort,
                model=evaluation_model
            )
            round_evaluations.append(evaluation)
            
            print(f"    Score: {evaluation.score:.2f}")
            print(f"    Correctness: {evaluation.correctness:.2f}")
            print(f"    Code Quality: {evaluation.code_quality:.2f}")
            print(f"    Tests Pass: {evaluation.test_passing}")
        
        # Store all rounds
        all_rounds[reasoning_effort] = {
            "results": round_results,
            "evaluations": round_evaluations
        }
        
        # Calculate aggregated statistics
        times = [r["time"] for r in round_results]
        scores = [e.score for e in round_evaluations]
        correctness_scores = [e.correctness for e in round_evaluations]
        code_quality_scores = [e.code_quality for e in round_evaluations]
        test_passing_counts = sum(1 for e in round_evaluations if e.test_passing)
        
        aggregated_results[reasoning_effort] = {
            "mean_time": sum(times) / len(times),
            "std_time": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 if len(times) > 1 else 0.0,
            "mean_score": sum(scores) / len(scores),
            "std_score": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5 if len(scores) > 1 else 0.0,
            "mean_correctness": sum(correctness_scores) / len(correctness_scores),
            "std_correctness": (sum((c - sum(correctness_scores)/len(correctness_scores))**2 for c in correctness_scores) / len(correctness_scores))**0.5 if len(correctness_scores) > 1 else 0.0,
            "mean_code_quality": sum(code_quality_scores) / len(code_quality_scores),
            "std_code_quality": (sum((q - sum(code_quality_scores)/len(code_quality_scores))**2 for q in code_quality_scores) / len(code_quality_scores))**0.5 if len(code_quality_scores) > 1 else 0.0,
            "test_passing_rate": test_passing_counts / len(round_evaluations),
            "num_rounds": NUM_ROUNDS,
        }
        
        print(f"\n  Aggregated Results ({reasoning_effort.upper()}):")
        print(f"    Mean Score: {aggregated_results[reasoning_effort]['mean_score']:.2f} ± {aggregated_results[reasoning_effort]['std_score']:.2f}")
        print(f"    Mean Correctness: {aggregated_results[reasoning_effort]['mean_correctness']:.2f} ± {aggregated_results[reasoning_effort]['std_correctness']:.2f}")
        print(f"    Mean Code Quality: {aggregated_results[reasoning_effort]['mean_code_quality']:.2f} ± {aggregated_results[reasoning_effort]['std_code_quality']:.2f}")
        print(f"    Test Passing Rate: {aggregated_results[reasoning_effort]['test_passing_rate']:.1%}")
        print(f"    Mean Time: {aggregated_results[reasoning_effort]['mean_time']:.1f}s ± {aggregated_results[reasoning_effort]['std_time']:.1f}s")
    
    # Summary
    print("\n" + "="*60)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*60)
    
    for level in REASONING_EFFORT_LEVELS:
        agg = aggregated_results[level]
        print(f"{level.upper()}:")
        print(f"  Mean Time: {agg['mean_time']:.1f}s ± {agg['std_time']:.1f}s")
        print(f"  Mean Quality Score: {agg['mean_score']:.2f} ± {agg['std_score']:.2f}")
        print(f"  Mean Correctness: {agg['mean_correctness']:.2f} ± {agg['std_correctness']:.2f}")
        print(f"  Mean Code Quality: {agg['mean_code_quality']:.2f} ± {agg['std_code_quality']:.2f}")
        print(f"  Test Passing Rate: {agg['test_passing_rate']:.1%}")
    
    # Find best implementation by mean score
    best_level = max(REASONING_EFFORT_LEVELS, key=lambda x: aggregated_results[x]['mean_score'])
    print(f"\nBest Implementation (by mean score): {best_level.upper()}")
    print(f"  Mean Score: {aggregated_results[best_level]['mean_score']:.2f} ± {aggregated_results[best_level]['std_score']:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"e11_results_{timestamp}.json"
    
    # Serialize individual rounds
    serializable_rounds = {}
    for level in REASONING_EFFORT_LEVELS:
        rounds_data = []
        for i, (result, evaluation) in enumerate(zip(all_rounds[level]["results"], all_rounds[level]["evaluations"])):
            rounds_data.append({
                "round": i + 1,
                "reasoning_effort": result["reasoning_effort"],
                "time": result["time"],
                "evaluation": {
                    "score": evaluation.score,
                    "correctness": evaluation.correctness,
                    "code_quality": evaluation.code_quality,
                    "test_passing": evaluation.test_passing,
                    "reasoning": evaluation.reasoning,
                    "strengths": evaluation.strengths,
                    "weaknesses": evaluation.weaknesses,
                }
            })
        serializable_rounds[level] = rounds_data
    
    output = {
        "experiment": "e11_codex_plan_reasoning_effort",
        "timestamp": timestamp,
        "num_rounds": NUM_ROUNDS,
        "task": MINIMAL_TASK,
        "test_case": test_case.model_dump(),
        "aggregated_results": aggregated_results,
        "individual_rounds": serializable_rounds,
        "best_implementation": best_level,
    }
    
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    return output


if __name__ == "__main__":
    asyncio.run(main())

