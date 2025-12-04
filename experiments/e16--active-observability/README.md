# E16: Active Observability - Trace Summarization Impact on Reflexion

## Overview
This experiment measures the impact of **active trace summarization** on Reflexion agent performance using a hard problem from HumanEval dataset.

## Research Question
**Does providing compressed trace summaries improve Reflexion's ability to learn from failures and converge faster?**

## Experimental Setup

### Baseline: Reflexion Simple Mode (No Trace Summarization)
- Standard Reflexion simple mode: Act → Evaluate → Reflect → Act (loop)
- Agent sees full trace history (all steps, all tool calls)
- Standard memory retrieval from Neo4j (if enabled)

### Treatment: Reflexion Simple Mode + Active Trace Summarization  
- Same Reflexion loop, but:
- **Before each Act**: Agent receives compressed summary of previous attempt
- **Compression logic**: Collapse successful steps, expand failures with details
- **Memory**: Both raw traces (for debugging) and summaries (for agent consumption)

### Test Problem
- **Source**: HumanEval dataset (hard problem, requires multiple attempts)
- **Criteria**: Problem that typically takes 2+ rounds to solve
- **Example candidates**: 
  - Multi-step reasoning problems
  - Problems requiring tool composition
  - Problems with subtle edge cases

## Metrics
1. **Convergence speed**: Rounds to success (lower is better)
2. **Token efficiency**: Total tokens used (lower is better)  
3. **Success rate**: % of runs that succeed within max_rounds
4. **Trace compression ratio**: Original trace size vs compressed size
5. **Quality of reflection**: Does compressed trace lead to better reflection?

## Implementation Notes
- Use LangFuse for trace storage
- Use `smart_trace_logic.py` for compression
- Inject summary into Reflexion's reflection prompt
- Run multiple seeds per condition for statistical significance
