"""
E16: Active Observability - Trace Summarization Impact on Reflexion

Experimental designs for comparing Reflexion with/without trace summarization.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ExperimentDesign(Enum):
    """Different experimental setups to test."""
    
    # Design 1: Pre-Act Summarization
    # Inject compressed trace summary BEFORE each Act node
    # Hypothesis: Agent can better plan if it sees what went wrong last time
    PRE_ACT_SUMMARIZATION = "pre_act"
    
    # Design 2: Post-Evaluate Summarization  
    # Inject summary AFTER evaluation, BEFORE reflection
    # Hypothesis: Reflection quality improves with compressed context
    POST_EVAL_SUMMARIZATION = "post_eval"
    
    # Design 3: Progressive Summarization
    # Summarize after every N tool calls during Act phase
    # Hypothesis: Prevents "lost in the middle" during long tool sequences
    PROGRESSIVE_SUMMARIZATION = "progressive"
    
    # Design 4: Failure-Only Summarization
    # Only summarize when evaluation score < threshold
    # Hypothesis: Reduces overhead, focuses on learning from mistakes
    FAILURE_ONLY_SUMMARIZATION = "failure_only"
    
    # Design 5: Hybrid: Pre-Act + Progressive
    # Combine Design 1 + Design 3
    # Hypothesis: Best of both worlds - planning + mid-execution awareness
    HYBRID_SUMMARIZATION = "hybrid"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    design: ExperimentDesign
    problem_id: str  # HumanEval problem ID
    max_rounds: int = 5
    eval_threshold: float = 0.8
    n_seeds: int = 5  # Number of runs per condition
    enable_memory: bool = True
    compression_threshold: int = 5  # For progressive: summarize after N tool calls


# Hard HumanEval problems (candidates for testing)
HUMANEVAL_HARD_PROBLEMS = [
    {
        "id": "HumanEval/0",  # Example - replace with actual hard problem
        "name": "Multi-step reasoning with tool composition",
        "difficulty": "hard",
        "typical_rounds": 3,
        "description": "Requires chaining multiple tools, has edge cases"
    },
    # Add more hard problems here
]


def get_experiment_designs() -> List[Dict[str, Any]]:
    """
    Return creative experimental designs for user feedback.
    
    Each design specifies:
    - When to inject summaries
    - What context to provide
    - Expected impact
    """
    return [
        {
            "id": "design_1_pre_act",
            "name": "Pre-Act Summarization",
            "description": """
            Inject compressed trace summary BEFORE each Act node.
            
            Flow: [Previous Round Summary] → Act → Evaluate → Reflect → [New Summary] → Act
            
            Hypothesis: Agent can better plan next attempt if it sees compressed 
            "what went wrong" summary from previous round.
            
            Pros: 
            - Clear separation: planning vs execution
            - Agent gets "lessons learned" before acting
            
            Cons:
            - May add latency (fetch + compress before each act)
            - Might lose fine-grained tool-level details
            """,
            "implementation": "Modify Reflexion's act() node to fetch and inject summary",
            "metrics": ["rounds_to_success", "reflection_quality_score", "token_efficiency"]
        },
        {
            "id": "design_2_post_eval",
            "name": "Post-Evaluate Summarization",
            "description": """
            Inject summary AFTER evaluation, BEFORE reflection.
            
            Flow: Act → Evaluate → [Compressed Trace Summary] → Reflect → Act
            
            Hypothesis: Reflection quality improves when agent sees compressed 
            "what happened" instead of raw trace scroll.
            
            Pros:
            - Reflection is where learning happens - focus optimization here
            - Can highlight failures explicitly
            
            Cons:
            - Reflection might miss subtle patterns in raw trace
            """,
            "implementation": "Modify Reflexion's reflect() node to use compressed trace",
            "metrics": ["reflection_quality_score", "rounds_to_success", "memory_retrieval_quality"]
        },
        {
            "id": "design_3_progressive",
            "name": "Progressive Summarization (Mid-Execution)",
            "description": """
            Summarize after every N tool calls DURING the Act phase.
            
            Flow: Act [Tool1, Tool2, Tool3] → [Summarize] → [Tool4, Tool5] → [Summarize] → ...
            
            Hypothesis: Prevents "lost in the middle" during long tool sequences.
            Agent maintains awareness of progress without token explosion.
            
            Pros:
            - Handles long execution traces gracefully
            - Agent can "check in" mid-execution
            
            Cons:
            - More complex implementation
            - May interrupt agent's flow
            """,
            "implementation": "Middleware that tracks tool calls, triggers compression every N calls",
            "metrics": ["trace_length_reduction", "mid_execution_awareness", "success_rate"]
        },
        {
            "id": "design_4_failure_only",
            "name": "Failure-Only Summarization",
            "description": """
            Only summarize when evaluation score < threshold (i.e., on failures).
            
            Flow: Act → Evaluate → [If score < threshold: Summarize] → Reflect → Act
            
            Hypothesis: Reduces overhead, focuses compression on learning moments.
            Successful rounds don't need compression (they're already done).
            
            Pros:
            - Minimal overhead (only on failures)
            - Focuses on learning from mistakes
            
            Cons:
            - Might miss patterns in "almost successful" attempts
            """,
            "implementation": "Conditional compression in evaluate() node",
            "metrics": ["overhead_reduction", "failure_recovery_rate", "token_efficiency"]
        },
        {
            "id": "design_5_hybrid",
            "name": "Hybrid: Pre-Act + Progressive",
            "description": """
            Combine Design 1 (Pre-Act) + Design 3 (Progressive).
            
            Flow: [Previous Round Summary] → Act [Tool1-Tool3] → [Summarize] → [Tool4-Tool6] → ...
            
            Hypothesis: Best of both worlds - planning awareness + mid-execution awareness.
            
            Pros:
            - Comprehensive observability
            - Handles both planning and execution phases
            
            Cons:
            - Most complex implementation
            - Highest overhead
            """,
            "implementation": "Combine pre-act injection + progressive middleware",
            "metrics": ["comprehensive_awareness", "rounds_to_success", "overall_efficiency"]
        },
        {
            "id": "design_6_adaptive",
            "name": "Adaptive Summarization",
            "description": """
            Dynamically decide when to summarize based on trace characteristics.
            
            Rules:
            - If trace > 20 steps: Summarize
            - If error detected: Summarize immediately
            - If round > 2: Summarize (agent struggling)
            - Otherwise: No summary (keep raw trace)
            
            Hypothesis: Smart compression that adapts to situation.
            
            Pros:
            - Optimal balance of detail vs compression
            - Context-aware
            
            Cons:
            - Complex logic
            - Harder to reason about
            """,
            "implementation": "Decision logic in middleware based on trace metrics",
            "metrics": ["adaptive_effectiveness", "overhead_reduction", "success_rate"]
        }
    ]


if __name__ == "__main__":
    designs = get_experiment_designs()
    print("🧪 E16 Experimental Designs\n")
    print("=" * 70)
    for i, design in enumerate(designs, 1):
        print(f"\n{i}. {design['name']} ({design['id']})")
        print("-" * 70)
        print(design['description'])
        print(f"\nMetrics: {', '.join(design['metrics'])}")

