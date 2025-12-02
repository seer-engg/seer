# Experiment E9c: Context Level Impact on End-to-End Evaluation

## Hypothesis

**H0:** Enriching target agent messages with more context (system goal, expected actions, MCP services) does not significantly impact evaluation effectiveness.

**H1:** More context leads to:
- Higher test pass rates (target agent performs better)
- Better test case generation (eval agent understands what target will receive)
- Optimal context level exists that balances adversarial testing with realistic behavior

## Experimental Design

**Independent Variable:** Context level (0, 1, 2, 3)
- **Level 0 (Minimal):** Only `input_message` from dataset example
- **Level 1 (System Goal):** Level 0 + system goal description
- **Level 2 (System Goal + Action):** Level 1 + expected action from `expected_output`
- **Level 3 (Full Context):** Level 2 + MCP services list + resource hints

**Controlled Variables:**
- LLM: GPT-5.1, Temperature: 0.0
- Test scenario: Fixed buggy_coder evaluation
- Tools available: Same across all levels
- Evaluation criteria: Same reflection mechanism
- Simplified execution: 1 test case, 1 round, no Codex handoff

**Dependent Variables (Metrics Captured):**
- `execution_time`: Total workflow execution time (seconds)
- `num_test_cases`: Number of test cases generated (`len(dataset_examples)`)
- `num_results`: Number of test results (`len(latest_results)`)
- `passed`: Count of tests that passed
- `failed`: Count of tests that failed
- `pass_rate`: `passed / num_results` (0.0 to 1.0)

## Test Scenario

```
Evaluate my agent buggy_coder at https://github.com/seer-engg/buggy-coder

The agent should sync Asana ticket updates when a GitHub PR is merged. 
Whenever I merge a PR, it should search for related Asana tickets and update/close them.
```

## Results Structure

Results are saved to `e9c_results/level_{level}_result.json` with the following structure:

```json
{
  "success": true,
  "timestamp": "ISO timestamp",
  "context_level": 0-3,
  "metrics": {
    "context_level": 0-3,
    "execution_time": float,
    "num_test_cases": int,
    "num_results": int,
    "passed": int,
    "failed": int,
    "pass_rate": float
  },
  "result": {
    "dataset_examples": [...],  // Generated test cases
    "latest_results": [...]     // Execution results with pass/fail
  }
}
```

Each `latest_results` entry contains:
- `thread_id`: Target agent thread ID
- `dataset_example`: The test case that was executed
- `actual_output`: Target agent's output
- `analysis`: FailureAnalysis with score, judge_reasoning
- `passed`: Boolean pass/fail
- `started_at`, `completed_at`: Timestamps

## Execution

```bash
# Run for each context level
python3 e9c_end_to_end.py --level 0
python3 e9c_end_to_end.py --level 1
python3 e9c_end_to_end.py --level 2
python3 e9c_end_to_end.py --level 3
```

## Expected Findings

**If H1 is supported:**
- Higher context levels → higher pass rates (target agent performs better)
- Context level affects test case quality/comprehensiveness
- Optimal level exists (e.g., Level 1 or 2)

**If H0 is supported:**
- Pass rates remain similar across levels
- Context level doesn't meaningfully impact evaluation

## Results

**Note:** Result JSON files were deleted during cleanup, but results were recovered from execution logs.

### Level 0 (Minimal)
- **Status:** Failed or incomplete (no results in logs)
- Execution Time: N/A
- Test Cases Generated: N/A
- Pass Rate: N/A
- Key Observations: Level 0 experiment did not complete successfully

### Level 1 (System Goal)
- Execution Time: **366.92 seconds** (~6.1 minutes)
- Test Cases Generated: **1**
- Pass Rate: **0 / 1 (0%)**
- Key Observations: 
  - Single test case generated
  - Test failed (0% pass rate)
  - Long execution time (~6 minutes)

### Level 2 (System Goal + Action)
- Execution Time: **367.56 seconds** (~6.1 minutes)
- Test Cases Generated: **1**
- Pass Rate: **0 / 1 (0%)**
- Key Observations:
  - Single test case generated
  - Test failed (0% pass rate)
  - Similar execution time to Level 1 (within 1 second)

### Level 3 (Full Context)
- Execution Time: **338.47 seconds** (~5.6 minutes)
- Test Cases Generated: **1**
- Pass Rate: **0 / 1 (0%)**
- Key Observations:
  - Single test case generated
  - Test failed (0% pass rate)
  - **Fastest execution time** (~30 seconds faster than Levels 1-2)

## Analysis

### Comparison Across Levels

**Pass Rate Trend:**
- All completed levels (1, 2, 3) showed **0% pass rate** - all tests failed
- No difference in pass rate across context levels
- **Insufficient data** - only 1 test case per level, Level 0 incomplete

**Execution Time Trend:**
- Level 1: 366.92s
- Level 2: 367.56s (virtually identical to Level 1)
- Level 3: 338.47s (**~30 seconds faster** than Levels 1-2)
- **Pattern:** Level 3 (full context) executed fastest, suggesting more context may reduce processing overhead

**Test Case Quality Trend:**
- All levels generated exactly **1 test case** (as configured with `EVAL_N_TEST_CASES=1`)
- Cannot assess quality differences with single test case per level

### Key Insights

1. **No Pass Rate Difference:** All context levels resulted in 0% pass rate. This could indicate:
   - The test case was inherently difficult/adversarial
   - Context level doesn't affect target agent performance for this scenario
   - Single test case is insufficient to detect differences

2. **Execution Time Improvement:** Level 3 (full context) was fastest, contrary to expectation that more context = more processing time. Possible explanations:
   - More context enables faster decision-making (less ambiguity)
   - Target agent processes richer context more efficiently
   - Random variation (need more runs to confirm)

3. **Limited Statistical Power:** With only 1 test case per level and Level 0 incomplete, cannot draw strong conclusions. Need:
   - Multiple test cases per level (n ≥ 3-5)
   - Multiple runs per level for statistical significance
   - Level 0 completion for baseline comparison

### Conclusion

**Hypothesis Status:** **Inconclusive** - insufficient data

- **H0 (no impact) vs H1 (context matters):** Cannot determine with current data
- **Pass rates:** Identical across levels (all 0%), but sample size too small
- **Execution time:** Level 3 fastest, but need more runs to confirm pattern

**Recommended Next Steps:**
1. Re-run Level 0 to complete baseline
2. Increase `EVAL_N_TEST_CASES` to 3-5 per level
3. Run multiple iterations per level for statistical power
4. Analyze test case content differences (if any) across levels
5. Investigate why Level 3 execution was faster

**Current Recommendation:** **Cannot recommend a context level** - experiment needs to be re-run with proper sample sizes.
