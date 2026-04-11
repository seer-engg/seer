# Seer Evals

## Run

```bash
python3 scripts/run_evals.py              # run all, show dashboard
python3 scripts/run_evals.py --baseline   # run + save snapshot
python3 scripts/run_evals.py --compare    # diff against last baseline
python3 scripts/run_evals.py --only tool  # run just one suite
```

No API key needed for `tool` and `mcp` suites. Retrieval suite needs `OPENAI_API_KEY` in `.env`.

## What Each Eval Catches

| Suite | File | What | Catches |
|-------|------|------|---------|
| **tool** | `test_tool_call_eval.py` | Calls 9 unified tool impls directly | JSON structure regressions, error handling, missing fields |
| **mcp** | `test_mcp_response_eval.py` | Measures response sizes, latency, structure | Token bloat, latency regressions, schema changes |
| **retrieval** | `test_retrieval_eval.py` | Tests tool/trigger search accuracy | Embedding drift, missing results, wrong-domain leaks |

## Pattern: Discriminator-Generator

Every eval follows the same pattern:

```
Generator (produces output)     Discriminator (checks output)
─────────────────────────       ─────────────────────────────
search_tools("send email")  →   assert "gmail" in top_match.tool
list_tools()                 →   assert total > 0, check structure
async_search_tools(prompt)  →   assert expected_tool in top-10
```

The generator calls a function. The discriminator asserts the result is correct.
Discriminators are deterministic and inspectable — no LLM in the check.

## Adding a New Eval

1. Create `tests/eval/test_your_eval.py`
2. Add `pytestmark = pytest.mark.eval`
3. Follow the discriminator-generator pattern
4. Add entry to `EVAL_SUITES` in `scripts/run_evals.py`

## Baselines

Snapshots saved to `tests/eval/baselines/<timestamp>.json`. Track:
- Response sizes (bytes, tokens)
- Latency (ms)
- Pass rates (%)

Compare: `python3 scripts/run_evals.py --compare` shows regressions in red.
