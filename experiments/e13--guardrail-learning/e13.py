"""
Experiment E13: Guardrail Learning - Bayesian Analysis

Tests if guardrails improve success rate and tool use using Bayesian inference.
Runs multiple task variants to gather sufficient data for probabilistic analysis.

Key innovation: LLM-based guardrail evaluation + Bayesian statistical analysis.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from collections import Counter
import math

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from composio import Composio
from composio_langchain import LangchainProvider
from langchain.agents import create_agent

# Visualization & Bayesian Analysis
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, some Bayesian analysis will be limited")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualization")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Use GPT-5.1 for all operations
MODEL_NAME = "gpt-5.1"

# Task variants for Bayesian analysis - Using better tasks from E7
TASK_VARIANTS = [
    {
        "name": "github_issues_summary",
        "task": "Find all GitHub issues assigned to me that mention 'bug' or 'error', check if any have related Slack discussions in #bugs channel, create a summary document with: issue title, description, and related Slack context, and post it to #engineering channel with priority tags based on issue labels.",
        "description": "Complex multi-step task: GitHub → Slack → Document → Post"
    },
    {
        "name": "deployment_summary",
        "task": "Find all Slack messages in #engineering from last week that mention 'deploy' or 'release', check corresponding GitHub pull requests, create a deployment summary in Google Sheets with columns: Date, PR, Author, Status, and notify the team in Slack.",
        "description": "Complex multi-step task: Slack → GitHub → Sheets → Slack notification"
    },
    {
        "name": "work_summary",
        "task": "Create a weekly work summary: Get my Google Calendar events for next week, get my GitHub pull requests from last week (all repos), get my Slack messages from #engineering channel from last week, combine all information into a structured Google Doc, and share the document with my manager.",
        "description": "Complex multi-step task: Calendar → GitHub → Slack → Doc → Share"
    },
    {
        "name": "email_task_extraction",
        "task": "Find all unread emails from last month that contain 'meeting' or 'urgent', extract any action items or deadlines mentioned, create a prioritized task list in Google Sheets with columns: Task, Deadline, Priority, and send me a summary email with the sheet link.",
        "description": "Complex multi-step task: Email → Extract → Sheets → Email"
    }
]

# ============================================================================
# Guardrail Models (same as before)
# ============================================================================

class Guardrail(BaseModel):
    guardrail_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    condition: str
    weight: float = Field(default=0.5)
    evidence_for: List[str] = Field(default_factory=list)
    evidence_against: List[str] = Field(default_factory=list)
    created_from: str = ""
    failure_type: str = Field(default="E")
    last_validated: Optional[datetime] = None
    validation_score: float = Field(default=0.0)
    validation_history: List[float] = Field(default_factory=list)

class GuardrailPrediction(BaseModel):
    guardrail_id: str
    predicted_outcome: bool
    confidence: float
    reasoning: str

class GuardrailApplicationResult(BaseModel):
    violations: List[str] = Field(default_factory=list)
    predictions: List[GuardrailPrediction] = Field(default_factory=list)

class GuardrailUpdate(BaseModel):
    guardrail_id: str
    action: str
    new_weight: Optional[float] = None
    reason: str

class SuccessEvaluation(BaseModel):
    success: bool
    reasoning: str
    confidence: float

class FailurePattern(BaseModel):
    failure_type: str
    description: str
    confidence: float

class ExperimentRun(BaseModel):
    run_id: str
    condition: str
    attempt_number: int
    task_variant: str
    success: bool
    success_reasoning: str = ""
    success_confidence: float = 0.0
    failure_pattern: Optional[FailurePattern] = None
    tool_calls_made: bool = False
    tool_call_count: int = 0
    guardrails_applied: List[str] = Field(default_factory=list)
    guardrail_violations: List[str] = Field(default_factory=list)
    guardrail_predictions: Dict[str, bool] = Field(default_factory=dict)
    execution_time: float = 0.0
    output: str = ""
    timestamp: datetime

# ============================================================================
# Guardrail System (same as before, condensed)
# ============================================================================

class GuardrailSystem:
    def __init__(self, model: ChatOpenAI):
        self.model = model
        self.guardrails: Dict[str, Guardrail] = {}
    
    async def create_guardrail_from_failure(
        self,
        failure_reason: str,
        execution_output: str,
        tool_calls_made: int,
        tool_call_count: int,
        actual_success: bool
    ) -> Guardrail:
        prompt = f"""Analyze this failure and create a guardrail:

Failure: {failure_reason}
Output: {execution_output[:1000]}
Tool Calls: {tool_calls_made}/{tool_call_count}
Outcome: {"Success" if actual_success else "Failure"}

Create a guardrail with:
- name: Short descriptive name
- description: What it checks
- condition: How to check it (natural language)
- failure_type: A (ask input), B (no tools), C (permissions), D (fake success), E (other)
"""

        response = self.model.with_structured_output(Guardrail).invoke([HumanMessage(content=prompt)])
        guardrail = Guardrail(**response.model_dump())
        guardrail.created_from = failure_reason[:200]
        self.guardrails[guardrail.guardrail_id] = guardrail
        return guardrail
    
    async def apply_guardrails(
        self,
        execution_output: str,
        tool_calls_made: int,
        tool_call_count: int,
        task: str
    ) -> GuardrailApplicationResult:
        if not self.guardrails:
            return GuardrailApplicationResult()
        
        guardrails_json = json.dumps([g.model_dump() for g in self.guardrails.values()], default=str)
        prompt = f"""Evaluate execution against guardrails:

TASK: {task}
OUTPUT: {execution_output[:2000]}
TOOL CALLS: {tool_calls_made}/{tool_call_count}
GUARDRAILS: {guardrails_json}

For each guardrail, predict success/failure with confidence.
Return violations (guardrail IDs that predicted failure) and predictions.
"""

        try:
            response = self.model.with_structured_output(
                GuardrailApplicationResult,
                method="function_calling"
            ).invoke([HumanMessage(content=prompt)])
            return response
        except Exception as e:
            logger.error(f"Guardrail application failed: {e}")
            return GuardrailApplicationResult()
    
    async def evaluate_guardrails(
        self,
        actual_outcome: bool,
        guardrail_predictions: Dict[str, GuardrailPrediction]
    ) -> List[GuardrailUpdate]:
        evaluations = []
        for guardrail_id, prediction in guardrail_predictions.items():
            guardrail = self.guardrails.get(guardrail_id)
            if not guardrail:
                continue
            
            was_correct = (prediction.predicted_outcome == actual_outcome)
            
            if guardrail.validation_score == 0.0:
                guardrail.validation_score = 1.0 if was_correct else 0.0
            else:
                alpha = 0.3
                guardrail.validation_score = alpha * (1.0 if was_correct else 0.0) + (1 - alpha) * guardrail.validation_score
            
            guardrail.validation_history.append(guardrail.validation_score)
            guardrail.last_validated = datetime.now()
            
            if was_correct:
                guardrail.weight = min(1.0, guardrail.weight + 0.1 * prediction.confidence)
            else:
                guardrail.weight = max(0.0, guardrail.weight - 0.1 * prediction.confidence)
            
            action = "remove" if guardrail.validation_score < 0.3 and len(guardrail.evidence_against) > 2 else "keep"
            evaluations.append(GuardrailUpdate(
                guardrail_id=guardrail_id,
                action=action,
                new_weight=guardrail.weight,
                reason=f"Validation: {guardrail.validation_score:.2f}"
            ))
        
        return evaluations
    
    async def update_guardrails(self, updates: List[GuardrailUpdate]):
        for update in updates:
            if update.action == "remove" and update.guardrail_id in self.guardrails:
                del self.guardrails[update.guardrail_id]

# ============================================================================
# Helper Functions
# ============================================================================

async def categorize_failure_pattern(
    execution_output: str,
    success_reasoning: str,
    tool_calls_made: bool,
    model: ChatOpenAI
) -> Optional[FailurePattern]:
    if not execution_output:
        return None
    
    prompt = f"""Categorize failure: A (ask input), B (no tools), C (permissions), D (fake success), E (other)

Output: {execution_output[:1500]}
Reasoning: {success_reasoning[:500]}
Tool Calls: {tool_calls_made}
"""
    
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content.upper()
        failure_type = "E"
        for t in ["A", "B", "C", "D"]:
            if f"TYPE {t}" in content or content.startswith(t):
                failure_type = t
                break
        return FailurePattern(failure_type=failure_type, description=content[:200], confidence=0.8)
    except Exception as e:
        return FailurePattern(failure_type="E", description="Categorization failed", confidence=0.0)

async def evaluate_success_llm(
    task: str,
    execution_output: str,
    tool_calls_made: bool,
    tool_call_count: int,
    model: ChatOpenAI
) -> SuccessEvaluation:
    prompt = f"""Evaluate if task was completed successfully.

TASK: {task}
OUTPUT: {execution_output[:2000]}
TOOL CALLS: {tool_calls_made}, COUNT: {tool_call_count}

Rules:
1. Must have called tools (no tool calls = failure)
2. Must have executed, not just described
3. Must have evidence before claims
4. Asking for input = failure
5. Explicit "I cannot" = failure

Extract success criteria from TASK description.
"""
    
    try:
        response = model.with_structured_output(SuccessEvaluation).invoke([HumanMessage(content=prompt)])
        return response
    except Exception as e:
        success = tool_calls_made and tool_call_count > 0
        return SuccessEvaluation(success=success, reasoning=f"Fallback: {e}", confidence=0.5)

async def execute_task(
    task: str,
    agent_prompt: str,
    tools: List[Any],
    model: ChatOpenAI,
    guardrail_system: Optional[GuardrailSystem] = None
) -> Dict[str, Any]:
    start_time = time.time()
    
    try:
        agent = create_agent(model=model, tools=tools, system_prompt=agent_prompt)
        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        
        messages = result.get("messages", [])
        final_output = ""
        tool_calls_made = False
        tool_call_count = 0
        
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                final_output = msg.content
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls_made = True
                tool_call_count += len(msg.tool_calls)
            if hasattr(msg, 'name') and msg.name and 'tool' in str(msg.name).lower():
                tool_calls_made = True
                tool_call_count += 1
        
        guardrail_result = None
        if guardrail_system:
            guardrail_result = await guardrail_system.apply_guardrails(
                final_output, tool_calls_made, tool_call_count, task
            )
        
        evaluator_model = ChatOpenAI(model="gpt-5.1", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
        success_eval = await evaluate_success_llm(
            task, final_output, tool_calls_made, tool_call_count, evaluator_model
        )
        
        if guardrail_result and guardrail_result.violations and len(guardrail_result.violations) >= 2:
            success_eval.success = False
            success_eval.reasoning += f" Multiple guardrails violated ({len(guardrail_result.violations)})."
        
        return {
            "success": success_eval.success,
            "success_reasoning": success_eval.reasoning,
            "success_confidence": success_eval.confidence,
            "output": final_output,
            "execution_time": time.time() - start_time,
            "tool_calls_made": tool_calls_made,
            "tool_call_count": tool_call_count,
            "guardrail_result": guardrail_result.model_dump() if guardrail_result else None
        }
    except Exception as e:
        return {
            "success": False,
            "success_reasoning": f"Error: {str(e)}",
            "success_confidence": 0.0,
            "output": f"Error: {str(e)}",
            "execution_time": time.time() - start_time,
            "error": str(e),
            "tool_calls_made": False,
            "tool_call_count": 0,
            "guardrail_result": None
        }

# ============================================================================
# Bayesian Analysis
# ============================================================================

def bayesian_success_rate_analysis(
    with_guardrails_successes: int,
    with_guardrails_total: int,
    without_guardrails_successes: int,
    without_guardrails_total: int,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Dict[str, Any]:
    """
    Bayesian analysis of success rates.
    Uses Beta-Binomial conjugate prior.
    
    Returns:
    - Posterior distributions for both conditions
    - Probability that guardrails improve success rate
    - Credible intervals
    """
    
    # Posterior parameters (Beta distribution)
    # Beta(alpha + successes, beta + failures)
    with_alpha_post = alpha + with_guardrails_successes
    with_beta_post = beta + (with_guardrails_total - with_guardrails_successes)
    
    without_alpha_post = alpha + without_guardrails_successes
    without_beta_post = beta + (without_guardrails_total - without_guardrails_successes)
    
    # Expected success rates (posterior mean)
    with_rate = with_alpha_post / (with_alpha_post + with_beta_post)
    without_rate = without_alpha_post / (without_alpha_post + without_beta_post)
    
    # Sample from posterior distributions to compute P(guardrails > baseline)
    if HAS_SCIPY:
        from scipy.stats import beta as beta_dist
        
        # Sample from posteriors
        n_samples = 10000
        with_samples = beta_dist.rvs(with_alpha_post, with_beta_post, size=n_samples)
        without_samples = beta_dist.rvs(without_alpha_post, without_beta_post, size=n_samples)
        
        # Probability that guardrails improve success rate
        prob_improvement = np.mean(with_samples > without_samples)
        
        # Credible intervals (95%)
        with_ci = beta_dist.interval(0.95, with_alpha_post, with_beta_post)
        without_ci = beta_dist.interval(0.95, without_alpha_post, without_beta_post)
        
        return {
            "with_guardrails": {
                "posterior_mean": float(with_rate),
                "credible_interval_95": [float(with_ci[0]), float(with_ci[1])],
                "alpha": float(with_alpha_post),
                "beta": float(with_beta_post)
            },
            "without_guardrails": {
                "posterior_mean": float(without_rate),
                "credible_interval_95": [float(without_ci[0]), float(without_ci[1])],
                "alpha": float(without_alpha_post),
                "beta": float(without_beta_post)
            },
            "prob_guardrails_improve": float(prob_improvement),
            "improvement": float(with_rate - without_rate)
        }
    else:
        # Fallback without scipy
        return {
            "with_guardrails": {
                "posterior_mean": float(with_rate),
                "alpha": float(with_alpha_post),
                "beta": float(with_beta_post)
            },
            "without_guardrails": {
                "posterior_mean": float(without_rate),
                "alpha": float(without_alpha_post),
                "beta": float(without_beta_post)
            },
            "improvement": float(with_rate - without_rate)
        }

def test_independence(
    with_guardrails_successes: int,
    with_guardrails_total: int,
    without_guardrails_successes: int,
    without_guardrails_total: int
) -> Dict[str, Any]:
    """
    Test if success is independent of guardrail condition.
    Uses Chi-square test of independence.
    """
    
    if not HAS_SCIPY:
        return {"error": "scipy not available"}
    
    from scipy.stats import chi2_contingency
    
    # Contingency table
    #              Success  Failure
    # With GR      a        b
    # Without GR   c        d
    
    a = with_guardrails_successes
    b = with_guardrails_total - with_guardrails_successes
    c = without_guardrails_successes
    d = without_guardrails_total - without_guardrails_successes
    
    contingency_table = [[a, b], [c, d]]
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return {
        "chi_squared": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "independent": p_value > 0.05,  # Standard threshold
        "contingency_table": contingency_table,
        "expected_frequencies": expected.tolist()
    }

def bayesian_tool_use_analysis(
    with_guardrails_tool_counts: List[int],
    without_guardrails_tool_counts: List[int]
) -> Dict[str, Any]:
    """
    Bayesian analysis of tool use (continuous variable).
    Uses Normal-Gamma conjugate prior.
    """
    
    if not HAS_SCIPY or len(with_guardrails_tool_counts) == 0 or len(without_guardrails_tool_counts) == 0:
        return {"error": "Insufficient data or scipy not available"}
    
    from scipy.stats import t as t_dist
    
    with_mean = np.mean(with_guardrails_tool_counts)
    without_mean = np.mean(without_guardrails_tool_counts)
    
    with_std = np.std(with_guardrails_tool_counts, ddof=1) if len(with_guardrails_tool_counts) > 1 else 0
    without_std = np.std(without_guardrails_tool_counts, ddof=1) if len(without_guardrails_tool_counts) > 1 else 0
    
    # Bayesian t-test (assuming equal variances)
    pooled_std = np.sqrt(((len(with_guardrails_tool_counts) - 1) * with_std**2 + 
                          (len(without_guardrails_tool_counts) - 1) * without_std**2) /
                         (len(with_guardrails_tool_counts) + len(without_guardrails_tool_counts) - 2))
    
    se_diff = pooled_std * np.sqrt(1/len(with_guardrails_tool_counts) + 1/len(without_guardrails_tool_counts))
    t_stat = (with_mean - without_mean) / se_diff if se_diff > 0 else 0
    df = len(with_guardrails_tool_counts) + len(without_guardrails_tool_counts) - 2
    
    # Two-tailed p-value
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df)) if se_diff > 0 else 1.0
    
    return {
        "with_guardrails": {
            "mean": float(with_mean),
            "std": float(with_std),
            "n": len(with_guardrails_tool_counts)
        },
        "without_guardrails": {
            "mean": float(without_mean),
            "std": float(without_std),
            "n": len(without_guardrails_tool_counts)
        },
        "difference": float(with_mean - without_mean),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05
    }

# ============================================================================
# Experiment Runner
# ============================================================================

class GuardrailLearningExperiment:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.model = ChatOpenAI(model=MODEL_NAME, temperature=0.0, api_key=api_key)
        self.results: List[ExperimentRun] = []
        self.tools = None
        self.guardrail_system = GuardrailSystem(self.model)
    
    async def _load_tools(self):
        if self.tools is None:
            print("Loading GitHub and Asana tools...", flush=True)
            composio_client = Composio(provider=LangchainProvider())
            user_id = os.getenv("COMPOSIO_USER_ID", "default")
            
            # Load tools for all services used in task variants
            all_tools = await asyncio.to_thread(
                composio_client.tools.get,
                user_id=user_id,
                toolkits=["GITHUB", "ASANA", "SLACK", "GOOGLEDOCS", "GOOGLESHEETS", "GOOGLECALENDAR", "GMAIL"],
                limit=2000
            )
            
            filtered_tools = []
            for tool in all_tools:
                if len(tool.name) > 64:
                    continue
                tool_name_lower = tool.name.lower()
                # Broader keyword matching for complex tasks
                if any(kw in tool_name_lower for kw in [
                    "pull", "pr", "task", "asana", "merge", "search", "update", "create", "list", "get",
                    "issue", "slack", "message", "channel", "doc", "document", "sheet", "calendar", 
                    "event", "email", "gmail", "post", "send", "share", "notify"
                ]):
                    filtered_tools.append(tool)
                    if len(filtered_tools) >= 150:  # Increased for more complex tasks
                        break
            
            self.tools = filtered_tools
            print(f"✓ Loaded {len(self.tools)} tools", flush=True)
        
        return self.tools
    
    async def run_condition(
        self,
        condition: str,
        task: str,
        task_variant: str,
        num_attempts: int = 3
    ) -> List[ExperimentRun]:
        runs = []
        base_prompt = """You are an agent that executes tasks using tools. 

CRITICAL: You must use tools to execute tasks. Do not describe plans - execute them.
Before making any claim, you must gather evidence by calling tools.

For ambiguous tasks, make reasonable inferences and proceed. Do not ask for input unless absolutely necessary."""
        
        for attempt in range(1, num_attempts + 1):
            logger.info(f"Running {condition}, variant {task_variant}, attempt {attempt}/{num_attempts}")
            print(f"  Attempt {attempt}/{num_attempts}...", flush=True)
            
            if condition == "with_guardrails" and attempt > 1:
                failures = [r for r in runs if not r.success]
                if failures:
                    for failure in failures[-2:]:
                        if failure.output:
                            await self.guardrail_system.create_guardrail_from_failure(
                                failure_reason=f"Attempt {failure.attempt_number} failed: {failure.success_reasoning[:200]}",
                                execution_output=failure.output,
                                tool_calls_made=failure.tool_call_count,
                                tool_call_count=failure.tool_call_count,
                                actual_success=False
                            )
                    
                    guardrails_summary = "\n".join([
                        f"- {g.name}: {g.description} (weight: {g.weight:.2f})"
                        for g in list(self.guardrail_system.guardrails.values())[-5:]
                    ])
                    current_prompt = f"""{base_prompt}

GUARDRAILS (learned from past failures):
{guardrails_summary}

These guardrails help you avoid past mistakes."""
                else:
                    current_prompt = base_prompt
            else:
                current_prompt = base_prompt
            
            tools = await self._load_tools()
            guardrail_system = self.guardrail_system if condition == "with_guardrails" else None
            result = await execute_task(task, current_prompt, tools, self.model, guardrail_system)
            
            guardrail_updates = []
            if condition == "with_guardrails" and result.get("guardrail_result"):
                guardrail_result_dict = result["guardrail_result"]
                predictions_data = guardrail_result_dict.get("predictions", {})
                if isinstance(predictions_data, list):
                    predictions_dict = {pred.get("guardrail_id", ""): pred for pred in predictions_data if isinstance(pred, dict)}
                else:
                    predictions_dict = predictions_data
                
                guardrail_predictions = {}
                for gid, pred_data in predictions_dict.items():
                    if isinstance(pred_data, dict):
                        guardrail_predictions[gid] = GuardrailPrediction(
                            guardrail_id=gid,
                            predicted_outcome=pred_data.get("predicted_outcome", False),
                            confidence=pred_data.get("confidence", 0.5),
                            reasoning=pred_data.get("reasoning", "")
                        )
                
                if guardrail_predictions:
                    guardrail_updates = await self.guardrail_system.evaluate_guardrails(
                        actual_outcome=result["success"],
                        guardrail_predictions=guardrail_predictions
                    )
                    await self.guardrail_system.update_guardrails(guardrail_updates)
            
            failure_pattern = None
            if not result["success"]:
                failure_pattern = await categorize_failure_pattern(
                    result.get("output", ""),
                    result.get("success_reasoning", ""),
                    result["tool_calls_made"],
                    self.model
                )
            
            guardrail_result_dict = result.get("guardrail_result", {})
            run = ExperimentRun(
                run_id=str(uuid4()),
                condition=condition,
                attempt_number=attempt,
                task_variant=task_variant,
                success=result["success"],
                success_reasoning=result.get("success_reasoning", ""),
                success_confidence=result.get("success_confidence", 0.0),
                failure_pattern=failure_pattern,
                tool_calls_made=result["tool_calls_made"],
                tool_call_count=result["tool_call_count"],
                guardrails_applied=list(self.guardrail_system.guardrails.keys()) if condition == "with_guardrails" else [],
                guardrail_violations=guardrail_result_dict.get("violations", []) if guardrail_result_dict else [],
                guardrail_predictions={},
                execution_time=result["execution_time"],
                output=result["output"],
                timestamp=datetime.now()
            )
            
            runs.append(run)
            self.results.append(run)
            
            status = "✅ Success" if result["success"] else "❌ Failed"
            print(f"    {status} ({result['tool_call_count']} tools)", flush=True)
        
        return runs
    
    async def run_bayesian_experiment(self, num_attempts_per_variant: int = 3) -> Dict[str, Any]:
        """Run Bayesian experiment across all task variants."""
        
        print("="*70, flush=True)
        print("E13 BAYESIAN EXPERIMENT", flush=True)
        print(f"Model: {MODEL_NAME}", flush=True)
        print(f"Task Variants: {len(TASK_VARIANTS)}", flush=True)
        print(f"Attempts per variant: {num_attempts_per_variant}", flush=True)
        print("="*70, flush=True)
        
        all_results = {}
        
        for variant in TASK_VARIANTS:
            print(f"\n{'='*70}", flush=True)
            print(f"Variant: {variant['name']}", flush=True)
            print(f"Task: {variant['task']}", flush=True)
            print(f"{'='*70}", flush=True)
            
            # Reset guardrail system for each variant
            self.guardrail_system = GuardrailSystem(self.model)
            
            # Run with guardrails
            print(f"\nRunning WITH GUARDRAILS...", flush=True)
            with_runs = await self.run_condition("with_guardrails", variant['task'], variant['name'], num_attempts_per_variant)
            
            # Reset for without guardrails
            self.guardrail_system = GuardrailSystem(self.model)
            
            # Run without guardrails
            print(f"\nRunning WITHOUT GUARDRAILS...", flush=True)
            without_runs = await self.run_condition("without_guardrails", variant['task'], variant['name'], num_attempts_per_variant)
            
            all_results[variant['name']] = {
                "variant": variant,
                "with_guardrails": [r.model_dump() for r in with_runs],
                "without_guardrails": [r.model_dump() for r in without_runs],
                "guardrails_created": {gid: g.model_dump() for gid, g in self.guardrail_system.guardrails.items()}
            }
        
        # Aggregate data for Bayesian analysis
        all_with_runs = []
        all_without_runs = []
        
        for variant_name, variant_results in all_results.items():
            all_with_runs.extend(variant_results["with_guardrails"])
            all_without_runs.extend(variant_results["without_guardrails"])
        
        # Calculate aggregate metrics
        with_successes = sum(1 for r in all_with_runs if r.get("success"))
        with_total = len(all_with_runs)
        without_successes = sum(1 for r in all_without_runs if r.get("success"))
        without_total = len(all_without_runs)
        
        with_tool_counts = [r.get("tool_call_count", 0) for r in all_with_runs]
        without_tool_counts = [r.get("tool_call_count", 0) for r in all_without_runs]
        
        # Bayesian analysis
        bayesian_success = bayesian_success_rate_analysis(
            with_successes, with_total,
            without_successes, without_total
        )
        
        independence_test = test_independence(
            with_successes, with_total,
            without_successes, without_total
        )
        
        bayesian_tools = bayesian_tool_use_analysis(
            with_tool_counts,
            without_tool_counts
        )
        
        return {
            "variants": all_results,
            "aggregate": {
                "with_guardrails": {
                    "successes": with_successes,
                    "total": with_total,
                    "success_rate": with_successes / with_total if with_total > 0 else 0,
                    "tool_counts": with_tool_counts
                },
                "without_guardrails": {
                    "successes": without_successes,
                    "total": without_total,
                    "success_rate": without_successes / without_total if without_total > 0 else 0,
                    "tool_counts": without_tool_counts
                }
            },
            "bayesian_analysis": {
                "success_rate": bayesian_success,
                "independence": independence_test,
                "tool_use": bayesian_tools
            }
        }

# ============================================================================
# Visualization
# ============================================================================

def create_bayesian_visualization(results: Dict[str, Any], output_file: str):
    """Create Bayesian analysis visualization."""
    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not available)")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('E13 Bayesian Guardrail Learning Analysis', fontsize=16, fontweight='bold')
    
    bayesian = results['bayesian_analysis']
    aggregate = results['aggregate']
    
    # 1. Success Rate Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    with_rate = aggregate['with_guardrails']['success_rate']
    without_rate = aggregate['without_guardrails']['success_rate']
    bars = ax1.bar(['With Guardrails', 'Without Guardrails'], [with_rate, without_rate],
                   color=['#2ecc71', '#e74c3c'], alpha=0.7)
    ax1.set_ylabel('Success Rate', fontsize=11)
    ax1.set_title('Aggregate Success Rates', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Bayesian Posterior Distributions (Top Middle)
    if HAS_SCIPY and 'prob_guardrails_improve' in bayesian['success_rate']:
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy.stats import beta as beta_dist
        
        with_params = bayesian['success_rate']['with_guardrails']
        without_params = bayesian['success_rate']['without_guardrails']
        
        x = np.linspace(0, 1, 1000)
        with_pdf = beta_dist.pdf(x, with_params['alpha'], with_params['beta'])
        without_pdf = beta_dist.pdf(x, without_params['alpha'], without_params['beta'])
        
        ax2.plot(x, with_pdf, label='With Guardrails', linewidth=2, color='#2ecc71')
        ax2.plot(x, without_pdf, label='Without Guardrails', linewidth=2, color='#e74c3c', linestyle='--')
        ax2.set_xlabel('Success Rate', fontsize=11)
        ax2.set_ylabel('Probability Density', fontsize=11)
        ax2.set_title('Bayesian Posterior Distributions', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        prob_improve = bayesian['success_rate']['prob_guardrails_improve']
        ax2.text(0.5, 0.95, f'P(improvement) = {prob_improve:.2f}', 
                transform=ax2.transAxes, ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Independence Test (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'p_value' in bayesian['independence']:
        p_val = bayesian['independence']['p_value']
        independent = bayesian['independence']['independent']
        
        ax3.barh(['Independent?'], [1], color='green' if independent else 'red', alpha=0.7)
        ax3.set_xlim([0, 1])
        ax3.set_title('Independence Test (Chi-square)', fontsize=12, fontweight='bold')
        ax3.text(0.5, 0, f'p-value: {p_val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax3.text(0.5, 0.5, f'{"Independent" if independent else "Dependent"}', 
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax3.set_xticks([])
    
    # 4. Tool Use Comparison (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    with_tools = aggregate['with_guardrails']['tool_counts']
    without_tools = aggregate['without_guardrails']['tool_counts']
    
    ax4.boxplot([with_tools, without_tools], labels=['With GR', 'Without GR'])
    ax4.set_ylabel('Tool Calls', fontsize=11)
    ax4.set_title('Tool Use Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Success by Variant (Middle)
    ax5 = fig.add_subplot(gs[1, 1:])
    variants = list(results['variants'].keys())
    with_rates_by_variant = []
    without_rates_by_variant = []
    
    for variant_name in variants:
        variant_data = results['variants'][variant_name]
        with_successes = sum(1 for r in variant_data['with_guardrails'] if r.get('success'))
        with_total = len(variant_data['with_guardrails'])
        without_successes = sum(1 for r in variant_data['without_guardrails'] if r.get('success'))
        without_total = len(variant_data['without_guardrails'])
        
        with_rates_by_variant.append(with_successes / with_total if with_total > 0 else 0)
        without_rates_by_variant.append(without_successes / without_total if without_total > 0 else 0)
    
    x = np.arange(len(variants))
    width = 0.35
    ax5.bar(x - width/2, with_rates_by_variant, width, label='With Guardrails', color='#2ecc71', alpha=0.7)
    ax5.bar(x + width/2, without_rates_by_variant, width, label='Without Guardrails', color='#e74c3c', alpha=0.7)
    ax5.set_xlabel('Task Variant', fontsize=11)
    ax5.set_ylabel('Success Rate', fontsize=11)
    ax5.set_title('Success Rate by Task Variant', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([v[:15] for v in variants], rotation=45, ha='right')
    ax5.legend()
    ax5.set_ylim([0, 1.1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Credible Intervals (Bottom)
    if HAS_SCIPY and 'credible_interval_95' in bayesian['success_rate'].get('with_guardrails', {}):
        ax6 = fig.add_subplot(gs[2, :])
        with_ci = bayesian['success_rate']['with_guardrails']['credible_interval_95']
        without_ci = bayesian['success_rate']['without_guardrails']['credible_interval_95']
        with_mean = bayesian['success_rate']['with_guardrails']['posterior_mean']
        without_mean = bayesian['success_rate']['without_guardrails']['posterior_mean']
        
        ax6.errorbar([0], [with_mean], yerr=[[with_mean - with_ci[0]], [with_ci[1] - with_mean]], 
                    fmt='o', capsize=10, capthick=2, label='With Guardrails', color='#2ecc71', markersize=10)
        ax6.errorbar([1], [without_mean], yerr=[[without_mean - without_ci[0]], [without_ci[1] - without_mean]], 
                    fmt='s', capsize=10, capthick=2, label='Without Guardrails', color='#e74c3c', markersize=10)
        ax6.set_xticks([0, 1])
        ax6.set_xticklabels(['With Guardrails', 'Without Guardrails'])
        ax6.set_ylabel('Success Rate (95% CI)', fontsize=11)
        ax6.set_title('Bayesian Credible Intervals', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim([0, 1])
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Bayesian visualization saved to: {output_file}", flush=True)

# ============================================================================
# Main
# ============================================================================

async def main():
    """Run Bayesian experiment."""
    experiment = GuardrailLearningExperiment()
    
    # Run Bayesian experiment across all variants
    results = await experiment.run_bayesian_experiment(num_attempts_per_variant=3)
    
    # Save results
    output_file = Path(__file__).parent / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create visualization
    viz_file = Path(__file__).parent / "visualization.png"
    create_bayesian_visualization(results, str(viz_file))
    
    # Print Bayesian analysis results
    print("\n" + "="*70, flush=True)
    print("BAYESIAN ANALYSIS RESULTS", flush=True)
    print("="*70, flush=True)
    
    bayesian = results['bayesian_analysis']
    aggregate = results['aggregate']
    
    print(f"\n📊 AGGREGATE SUCCESS RATES:", flush=True)
    print(f"  With Guardrails: {aggregate['with_guardrails']['successes']}/{aggregate['with_guardrails']['total']} = {aggregate['with_guardrails']['success_rate']:.2%}", flush=True)
    print(f"  Without Guardrails: {aggregate['without_guardrails']['successes']}/{aggregate['without_guardrails']['total']} = {aggregate['without_guardrails']['success_rate']:.2%}", flush=True)
    
    if 'prob_guardrails_improve' in bayesian['success_rate']:
        print(f"\n🔬 BAYESIAN SUCCESS RATE ANALYSIS:", flush=True)
        print(f"  P(Guardrails Improve Success Rate) = {bayesian['success_rate']['prob_guardrails_improve']:.2%}", flush=True)
        print(f"  Improvement: {bayesian['success_rate']['improvement']:+.3f}", flush=True)
        if 'credible_interval_95' in bayesian['success_rate']['with_guardrails']:
            with_ci = bayesian['success_rate']['with_guardrails']['credible_interval_95']
            without_ci = bayesian['success_rate']['without_guardrails']['credible_interval_95']
            print(f"  With Guardrails 95% CI: [{with_ci[0]:.3f}, {with_ci[1]:.3f}]", flush=True)
            print(f"  Without Guardrails 95% CI: [{without_ci[0]:.3f}, {without_ci[1]:.3f}]", flush=True)
    
    if 'p_value' in bayesian['independence']:
        print(f"\n🔬 INDEPENDENCE TEST:", flush=True)
        print(f"  Chi-squared: {bayesian['independence']['chi_squared']:.3f}", flush=True)
        print(f"  p-value: {bayesian['independence']['p_value']:.4f}", flush=True)
        print(f"  Independent: {'Yes' if bayesian['independence']['independent'] else 'No'} (p > 0.05)", flush=True)
    
    if 'difference' in bayesian['tool_use']:
        print(f"\n🔬 TOOL USE ANALYSIS:", flush=True)
        print(f"  With Guardrails: {bayesian['tool_use']['with_guardrails']['mean']:.2f} ± {bayesian['tool_use']['with_guardrails']['std']:.2f}", flush=True)
        print(f"  Without Guardrails: {bayesian['tool_use']['without_guardrails']['mean']:.2f} ± {bayesian['tool_use']['without_guardrails']['std']:.2f}", flush=True)
        print(f"  Difference: {bayesian['tool_use']['difference']:+.2f}", flush=True)
        if 'p_value' in bayesian['tool_use']:
            print(f"  p-value: {bayesian['tool_use']['p_value']:.4f}", flush=True)
            print(f"  Significant: {'Yes' if bayesian['tool_use']['significant'] else 'No'} (p < 0.05)", flush=True)
    
    print(f"\n📁 Results saved to: {output_file}", flush=True)
    print(f"📊 Visualization saved to: {viz_file}", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
