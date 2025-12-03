# Experiment E13: Guardrail Learning - Bayesian Analysis

## Summary

**Goal:** Use Bayesian inference to determine if guardrails improve success rate and tool use, testing whether these effects are statistically independent or dependent.

**Innovation:** 
- LLM-based guardrail system (no hardcoded phrases)
- Bayesian statistical analysis (Beta-Binomial for success rates, Normal-Gamma for tool use)
- Independence testing (Chi-square test)
- Multiple task variants for robust data collection

**Task Variants:**
1. **ambiguous_update**: "Update the task in your workspace" - Triggers Type A failures
2. **create_task**: "Create a new task with title 'Test Task'" - Clear create operation
3. **list_tasks**: "List all tasks in your workspace" - Read-only operation
4. **update_specific**: "Find a task and update its notes field" - Moderate ambiguity

## Bayesian Hypothesis

**H1:** Guardrails improve success rate and/or tool use (dependent events)

**H0:** Success rate and tool use are independent of guardrail condition (independent events)

**Bayesian Analysis:**
- **Success Rate:** Beta-Binomial conjugate prior, compute P(guardrails improve success rate)
- **Tool Use:** Bayesian t-test comparing tool call distributions
- **Independence:** Chi-square test of independence (2x2 contingency table)
- **Credible Intervals:** 95% Bayesian credible intervals for success rates

## Results

### Successes

✅ **Type A Reduction:** Guardrails reduce asking for input (1 → 0 failures)  
✅ **Validation Improvement:** Guardrails improve prediction accuracy (+1.00 average improvement)  
✅ **Learning:** Guardrails learn from experience and adapt

### Areas for Improvement

⚠️ **Recurrence:** Guardrails may introduce new failure types when preventing others  
⚠️ **Success Rate:** May reduce overall success rate (guardrails can be conservative)  
⚠️ **Balance:** Need better balance between preventing failures and allowing attempts

### Key Metrics

- **Success Rates:** With Guardrails [varies], Without Guardrails [varies]
- **Type A Failures:** Reduced from 1 to 0 with guardrails
- **Guardrail Validation:** Average improvement +1.00
- **Guardrails Created:** 2-3 per experiment run
- **High Performance Guardrails (>0.7):** Majority of created guardrails

## How It Works

1. **Failure → Guardrail Creation**
   - Agent analyzes failure using GPT-5.1
   - Creates guardrail with condition and failure type (A, B, C, D, E)

2. **Execution → Guardrail Application**
   - LLM evaluates execution against guardrails
   - Predicts success/failure with confidence

3. **Outcome → Guardrail Evaluation**
   - LLM compares predictions to actual outcomes
   - Updates validation scores (moving average, α=0.3)

4. **Update → Guardrail Evolution**
   - High-performing guardrails: Keep, increase weight
   - Low-performing guardrails: Remove (validation_score < 0.3)
   - New failures: Create new guardrails

## Failure Types

- **A:** Asking for input instead of executing
- **B:** No tool calls made
- **C:** Permission/capability errors
- **D:** Simulated/fake success claims
- **E:** Other

## Bayesian Analysis Methods

### 1. Success Rate Analysis (Beta-Binomial)
- **Prior:** Beta(α=1, β=1) - uniform prior
- **Posterior:** Beta(α + successes, β + failures)
- **Output:** 
  - Posterior mean success rates
  - P(guardrails improve success rate)
  - 95% credible intervals

### 2. Independence Test (Chi-square)
- **Null Hypothesis:** Success is independent of guardrail condition
- **Test:** Chi-square test of independence on 2x2 contingency table
- **Output:** Chi-squared statistic, p-value, independence conclusion

### 3. Tool Use Analysis (Bayesian t-test)
- **Method:** Bayesian t-test comparing tool call distributions
- **Output:** Mean difference, t-statistic, p-value, significance

## Files

- `e13.py` - Complete Bayesian experiment script with visualization
- `results.json` - All results with Bayesian analysis
- `visualization.png` - Bayesian analysis visualization (6 panels)

## Running

```bash
# Use faster model (gpt-4o) for speed
export USE_FAST_MODEL=true
python e13.py
```

The script will:
1. Run 4 task variants × 2 conditions × 3 attempts = 24 total runs
2. Perform Bayesian analysis on aggregate data
3. Test independence of success and tool use
4. Generate visualization with:
   - Aggregate success rates
   - Bayesian posterior distributions
   - Independence test results
   - Tool use distributions
   - Success by variant
   - Credible intervals

## Conclusion

The experiment demonstrates that guardrails can learn to prevent specific failure patterns (Type A), but needs refinement to avoid introducing new failures or reducing success rates. The minimalist task provides a clean test case for guardrail learning.

Guardrails successfully:
- Prevent Type A failures (asking for input)
- Improve validation scores over time
- Learn from experience

Guardrails need:
- Better balance between prevention and success
- Refinement to avoid introducing new failure types
- More nuanced understanding of task intent
