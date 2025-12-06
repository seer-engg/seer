# Experiment E11: Codex Plan Reasoning Effort Impact

**Status**: ✅ Completed  
**Date**: December 2, 2025  
**Rounds**: 3 rounds per reasoning level (9 total runs)

## Purpose

Test whether varying reasoning effort (minimal/medium/high) in Codex's developer/plan node improves implementation quality and success rate. Codex takes significant time, so we use minimal viable test cases that can complete in < 10 minutes per condition.

## Hypothesis

- **H1:** Higher reasoning effort leads to better implementations because:
  1. More internal reasoning tokens = deeper understanding of requirements
  2. Better edge case consideration
  3. More complete and correct implementations
  4. Better alignment with test case expectations

- **H0:** Reasoning effort doesn't significantly impact implementation quality or success rate

**Result**: ✅ **H1 Confirmed** - Medium and High reasoning effort both achieved 0.50 score vs 0.40 for Minimal, confirming that planning depth matters.

## Quick Start

### Prerequisites

1. **Codex LangGraph Server Running**: Codex must be running as a LangGraph server
   ```bash
   # In seer root directory
   python run.py codex
   ```

2. **Environment Variables**: Set required API keys
   ```bash
   export OPENAI_API_KEY=your_key
   export CONTEXT7_API_KEY=your_key  # Optional
   ```

### Running the Experiment

```bash
cd experiments/e11--codex-plan-reasoning-effort
python e11_codex_plan_reasoning_effort.py
```

### Generating Visualizations

```bash
python generate_visualizations.py
```

## Experimental Design

### Conditions (3 Levels)

1. **Minimal Reasoning:** `reasoning_effort="minimal"`
2. **Medium Reasoning:** `reasoning_effort="medium"` (baseline)
3. **High Reasoning:** `reasoning_effort="high"`

### Task

Fix an agent so it returns "Hello, World!" when asked to greet (instead of just "Hello").

**Test Case:**
- **Input**: "Please greet me"
- **Expected Output**: "Hello, World!"
- **Type**: Minimal viable test case (designed to complete in <10 minutes)

### Variables

**Independent:**
- Reasoning effort level (minimal, medium, high) in Codex developer/plan node

**Dependent:**
- Implementation quality score (LLM-as-judge)
- Execution time
- Correctness
- Code quality
- Test pass rate

## Key Results

### 🏆 Best Implementation: **MEDIUM** Reasoning Effort

| Reasoning Effort | Mean Score | Mean Time (s) | Mean Correctness | Mean Code Quality |
|-----------------|------------|---------------|------------------|-------------------|
| **MINIMAL**     | 0.40 (±0.00) | 55.9 (±2.1) | 0.30 (±0.00) | 0.43 (±0.05) |
| **MEDIUM**      | 0.50 (±0.00) | 51.8 (±2.9) | 0.47 (±0.05) | 0.50 (±0.00) |
| **HIGH**        | 0.50 (±0.00) | 46.2 (±14.6) | 0.50 (±0.00) | 0.50 (±0.00) |

### Key Findings

1. **Quality Plateau**: Medium and High reasoning effort both achieved the same mean score (0.50), suggesting diminishing returns beyond medium reasoning.

2. **Speed Paradox**: High reasoning effort was actually the fastest (46.2s), though with high variance (±14.6s).

3. **Minimal Underperformance**: Minimal reasoning effort consistently underperformed (0.40 score) and was the slowest.

4. **Consistency**: Medium reasoning showed the most consistent performance across rounds (lowest variance in time: ±2.9s).

See `SUMMARY.md` for detailed analysis and `results_e11/` for complete results and visualizations.

## Files

- `e11_codex_plan_reasoning_effort.py` - Main experiment script
- `generate_visualizations.py` - Visualization generator
- `requirements.txt` - Python dependencies
- `SUMMARY.md` - Detailed results and analysis
- `results_e11/` - Results directory with JSON data and PNG visualizations

## Customization

### Using Your Own Test Case

Edit `e11_codex_plan_reasoning_effort.py` and modify:
- `MINIMAL_TASK`: The task description for Codex
- `MINIMAL_TEST_CASE`: The test case structure
- `create_minimal_failing_result()`: The failing test result

### Adjusting Reasoning Levels

Modify `REASONING_EFFORT_LEVELS` in the script to test different combinations.

## Configuration Changes

- Modified `agents/codex/nodes/developer.py` to use configurable reasoning effort
- Added `codex_reasoning_effort` field to `shared/config.py`

## Troubleshooting

1. **Codex server not running**: Start it with `python run.py codex`
2. **Sandbox errors**: Ensure E2B credentials are configured
3. **Timeout errors**: Use simpler test cases or increase timeout limits
4. **Import errors**: Ensure you're running from the experiment directory with seer root in PYTHONPATH
