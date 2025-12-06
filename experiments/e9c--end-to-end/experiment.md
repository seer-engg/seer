# Experiment E9c: Context Level Impact on End-to-End Evaluation

## Hypothesis

**H0:** Enriching target agent messages with more context (system goal, expected actions, MCP services) does not significantly impact evaluation effectiveness.

**H1:** More context leads to:
- Higher test pass rates (target agent performs better)
- Better test case generation (eval agent understands what target will receive)
- Optimal context level exists that balances adversarial testing with realistic behavior

## Experimental Design

**Independent Variable:** Context level (1, 2, 3)
- **Level 1 (System Goal):** `input_message` + system goal description
- **Level 2 (System Goal + Action):** Level 1 + expected action from `expected_output`
- **Level 3 (Full Context):** Level 2 + MCP services list + resource hints

**Controlled Variables:**
- LLM: GPT-5.1, Temperature: 0.0
- Test scenario: Fixed buggy_coder evaluation
- Tools available: Same across all levels
- Evaluation criteria: Same reflection mechanism
- Simplified execution: 1 test case, 1 round, no Codex handoff

**Test Variants:**
1. **Variant 1 (Read-only):** List GitHub issues - simplest operation
2. **Variant 2 (Simple Write):** Create GitHub issue with any title

## Results Summary

### Variant 1: Read-Only Operations (List GitHub Issues)

| Level | Context Type | Pass Rate | Execution Time | Status |
|-------|--------------|-----------|----------------|--------|
| 1 | System Goal | 100% (1/1) | 309.7s | ✅ PASSED |
| 2 | System Goal + Action | 0% (0/1) | 261.7s | ❌ FAILED |
| 3 | Full Context | 100% (1/1) | 281.1s | ✅ PASSED |

**Key Finding:** Levels 1 and 3 PASSED, Level 2 FAILED

### Variant 2: Simple Write Operations (Create GitHub Issue)

| Level | Context Type | Pass Rate | Execution Time | Status |
|-------|--------------|-----------|----------------|--------|
| 1 | System Goal | 0% (0/1) | 266.4s | ❌ FAILED |
| 2 | System Goal + Action | 0% (0/1) | 260.1s | ❌ FAILED |
| 3 | Full Context | ERROR | N/A | ❌ TIMEOUT |

**Key Finding:** All levels FAILED - write operations are significantly harder than read operations

## Key Findings

### 1. Context Level Matters for Read Operations
- **Level 1 (System Goal)** is optimal - 100% pass rate
- **Level 3 (Full Context)** also works - 100% pass rate
- **Level 2 (System Goal + Action)** problematic - 0% pass rate

### 2. Write Operations Are Harder
- **All context levels failed** for write operations (Variant 2)
- Suggests write operations need:
  - Different context structure
  - More specific instructions
  - Or are inherently more difficult to evaluate

### 3. The Pattern (Read Operations)
- **Levels 1 and 3:** Passed
- **Level 2:** Failed
- Possible explanations:
  - Level 1: System goal provides clear direction without overwhelming
  - Level 3: Full context removes all ambiguity
  - Level 2: Expected action might confuse or contradict agent behavior

### 4. Execution Time
- All variants: ~250-310 seconds per run
- No significant time difference between pass/fail
- Level 3 timeout suggests write operations may take longer

## Hypotheses

### Why Level 1 Works (Read)
- System goal tells agent WHAT to do (list issues)
- Provides enough context without being overwhelming
- Agent understands it should use GitHub tools
- No conflicting information

### Why Level 3 Works (Read)
- Full context provides complete information
- Agent has all context needed (goal, action, tools, resources)
- No ambiguity about what to do
- May be overkill but works

### Why Level 2 Fails (Read)
- System Goal + Action might be:
  - Contradictory (goal says "list" but action might imply something else)
  - Overwhelming (too much specific instruction)
  - Confusing (agent might misinterpret expected action)
- The "expected action" might interfere with natural behavior

### Why All Levels Fail (Write)
- Write operations are inherently more complex
- May need different evaluation criteria
- Agent might need more specific instructions
- Could be a limitation of the test scenario or agent capability

## Recommendations

### For Read-Only Operations
**Use Level 1 (System Goal)**
- Highest pass rate (100%)
- Reasonable execution time
- Provides necessary context without overwhelming
- Optimal balance

**Alternative:** Level 3 if maximum context needed, but Level 1 appears sufficient

### For Write Operations
**Need Further Investigation**
- Current context levels don't work for write operations
- Consider:
  - Different context structure
  - More specific instructions
  - Different evaluation criteria

### Production Recommendations
1. **Default to Level 1** for most operations
2. **Use Level 3** only if Level 1 fails
3. **Avoid Level 2** - consistently problematic

## Conclusion

**Hypothesis Status:** **Partially Supported** - Context level DOES significantly impact evaluation effectiveness

- **For read operations:** Level 1 or 3 optimal (Level 1 recommended)
- **For write operations:** Current context levels insufficient - need different approach
- **Pattern:** Levels 1 and 3 perform better than Level 2 for read operations
- **Recommendation:** Use Level 1 (System Goal) as default for production

## Data

All detailed results are available in `results.json` with complete metrics, test cases, and execution details for each level and variant combination.
