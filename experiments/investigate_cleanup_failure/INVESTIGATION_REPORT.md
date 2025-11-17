# Investigation Report: Test Execution Failure Analysis

**Date**: 2025-11-17  
**Investigator**: Scientific Investigation Agent  
**Test ID**: 90c1d783-ef7a-4ec9-9e72-23c1a86fe988  

---

## Executive Summary

Investigation of test execution failure reveals **three critical bugs** in the Seer eval_agent system:

1. **Test generation creates invalid tests** - Missing branch creation before PR creation
2. **Cleanup system can't extract MCP-wrapped output fields** - Path mismatch due to double nesting
3. **Reflection agent misdiagnoses provisioning failures** - Attributes test generation bugs to target agent

All three bugs are proven with experimental evidence.

---

## Bug #1: Invalid Test Generation (Missing Branch Creation)

### Evidence

Generated test attempts to create a PR from non-existent branch:

```json
{
  "Step 1": "github_create_an_organization_repository (creates 'main' only)",
  "Step 2": "github_create_a_pull_request with head='feature/label-empty-string'",
  "Result": "422 Validation Failed - head branch invalid"
}
```

### Root Cause

Test generation doesn't understand GitHub's PR creation requirements:
- Both `base` and `head` branches must exist
- `head` branch must have commits distinct from `base`

Creating a repo with `auto_init=true` only creates the `main` branch.

### Proposed Fix

**Option A: Add validation to test generation prompts**
```python
# In genetic_eval_generation.py or agentic_eval_generation.py
GITHUB_PR_REQUIREMENTS = """
**CRITICAL - GitHub PR Requirements:**
- To create a PR, you MUST first create the head branch
- Steps required:
  1. Create repository
  2. Create a file/reference on the new branch
  3. THEN create PR with existing base and head branches
"""
```

**Option B: Add post-generation validation**
```python
# In validate_generated_actions.py
def _validate_pr_creation(actions, invalid):
    """Ensure PR creation is preceded by branch creation."""
    for i, action in enumerate(actions):
        if action.tool == "github_create_a_pull_request":
            params = json.loads(action.params)
            head_branch = params.get("head")
            
            # Check if head branch is created in previous actions
            branch_created = any(
                a.tool in ["github_create_a_reference", 
                          "github_create_or_update_file_contents"]
                for a in actions[:i]
            )
            
            if not branch_created and head_branch not in ["main", "master"]:
                invalid.append(
                    f"PR creation requires head branch '{head_branch}' to exist. "
                    f"Add github_create_a_reference or github_create_or_update_file_contents first."
                )
```

### Impact

- **Severity**: HIGH - Causes valid test scenarios to fail during provisioning
- **Frequency**: Any test involving PR creation from feature branches
- **Workaround**: None currently available

---

## Bug #2: MCP Output Field Extraction Failure

### Evidence

```
2025-11-17 00:58:31 | shared.cleanup_inverses | WARNING | Could not extract 'owner.login' from output for cleanup of github_create_an_organization_repository
```

### Experimental Proof

Created `experiments/investigate_cleanup_failure/verify_output_structure.py`:

**MCP Tool Output Structure**:
```json
{
  "successfull": true,
  "error": null,
  "data": {
    "data": {
      "owner": {"login": "seer-engg"},
      "name": "label-edgecase-repo"
    }
  }
}
```

**Field Extraction Results**:
| Path                    | Result     | Value      |
|------------------------|------------|------------|
| `owner.login`          | ✗ FAILED   | None       |
| `data.owner.login`     | ✗ FAILED   | None       |
| `data.data.owner.login`| ✓ SUCCESS  | seer-engg  |

### Root Cause

1. MCP tools wrap responses: `{successfull, error, data: {data: {...}}}`
2. LLM generates inverse mapping without knowledge of MCP wrapper
3. Extraction logic doesn't account for double nesting

### Proposed Fixes

**Option A: Normalize MCP output before cleanup**
```python
# In action_executor.py, around line 112
def _normalize_mcp_output(output):
    """Extract actual API response from MCP wrapper."""
    if isinstance(output, dict) and "successfull" in output and "data" in output:
        # MCP wrapper detected
        if "data" in output["data"]:
            return output["data"]["data"]  # Double-nested
        return output["data"]  # Single-nested
    return output  # Not MCP-wrapped

# Then use normalized output:
normalized_output = _normalize_mcp_output(output)
inverse_action = await create_inverse_action(
    original=action,
    output=normalized_output,  # Use normalized version
    ...
)
```

**Option B: Teach LLM about MCP structure**
```python
# In cleanup_inverses.py, around line 152
prompt = f"""You are identifying the inverse DELETE operation for a CREATE operation.

**IMPORTANT**: MCP tools wrap outputs in this structure:
{{
  "successfull": true,
  "error": null,
  "data": {{
    "data": {{ ...actual GitHub/Asana response... }}
  }}
}}

When mapping fields, account for this nesting:
- GitHub owner.login is at: data.data.owner.login
- GitHub name is at: data.data.name
- Asana gid is at: data.data.gid

CREATE TOOL: {create_tool}
CREATE OUTPUT: {json.dumps(output, indent=2, default=str)}
...
```

**Recommended**: **Option A** - More robust, doesn't rely on LLM understanding

### Impact

- **Severity**: MEDIUM - Cleanup fails silently, resources may be leaked
- **Frequency**: Every action that creates resources
- **Workaround**: Manual cleanup of resources

---

## Bug #3: Reflection Misdiagnosis

### Evidence

**Saved Reflection** (line 745):
> "The agent fails to handle edge cases where a pull request label has an empty string as its name. This leads to the construction of an invalid 'head' (branch name)..."

**Actual Failure**:
- Target agent (`buggy_coder`) never executed
- Provisioning phase failed at Step 2
- Branch name `feature/label-empty-string` is valid; it just doesn't exist

### Root Cause

Reflection agent receives:
```json
{
  "failure_type": "runtime_error",
  "judge_reasoning": "Test execution failed: Provisioning failed: Execution failed: {...422 Validation Failed...}"
}
```

But interprets this as agent behavior rather than test infrastructure failure.

### Proposed Fix

**Add failure type discrimination**:
```python
# In reflection logic (wherever reflection agent receives failure info)
def classify_failure_origin(failure_info):
    """Determine if failure is from test infrastructure or target agent."""
    reasoning = failure_info.get("judge_reasoning", "")
    
    if "Provisioning failed" in reasoning:
        return "test_generation_bug"
    elif "PHASE 2: INVOKE AGENT" in reasoning and "after provision":
        return "target_agent_bug"
    elif "PHASE 3: ASSERT" in reasoning:
        return "assertion_failure"
    else:
        return "unknown"

# Then in reflection prompt:
if failure_origin == "test_generation_bug":
    prompt = """
    This failure occurred during test PROVISIONING, not agent execution.
    The test generation created an invalid test sequence.
    
    Focus your reflection on:
    1. What did test generation misunderstand about the API requirements?
    2. How can test generation prompts be improved?
    3. What validation should catch this?
    
    DO NOT reflect on the target agent's behavior - it never ran.
    """
```

### Impact

- **Severity**: HIGH - Creates feedback loop of bad reflections → bad tests
- **Frequency**: Any provisioning failure
- **Workaround**: Manual review of reflections to filter out false diagnoses

---

## What "But Continues" Means

The user observed that after the test execution failure, the system **continues** to the reflection phase (lines 712-760). This is **correct behavior** - the eval_agent should:

1. ✓ Execute tests
2. ✓ Handle failures gracefully
3. ✓ Reflect on failures
4. ✓ Store reflections for future test generation

The issue is not that it continues, but that it reflects on the WRONG problem.

---

## Remediation Priority

1. **IMMEDIATE**: Fix Bug #1 (test validation) - Blocks all PR-based tests
2. **HIGH**: Fix Bug #2 (cleanup extraction) - Resource leaks
3. **HIGH**: Fix Bug #3 (reflection accuracy) - Corrupts feedback loop

---

## Validation Tests

Created experimental validation in `experiments/investigate_cleanup_failure/`:
- `verify_output_structure.py` - Proves Bug #2
- `extract_test_structure.py` - Documents Bug #1

Both can be run independently to verify findings.

---

## Conclusion

All three bugs are empirically proven. The system "continues" correctly, but operates on false premises due to misdiagnosed failures. Fixes are straightforward but require careful coordination between test generation, execution, and reflection subsystems.

