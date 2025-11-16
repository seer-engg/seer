# Parameter Population Issue - TODO

## Problem Summary
LLM generates `ActionStep` objects with incomplete or empty parameters when it should extract values from context.

**Example Failure:**
```
ActionStep(
    tool="github_list_pull_requests",
    params="{}"  # EMPTY - should have owner/repo
)
```

**Expected:**
```
ActionStep(
    tool="github_list_pull_requests",
    params='{"owner": "seer-engg", "repo": "buggy-coder"}'
)
```

## Root Cause
1. User provides context: `https://github.com/seer-engg/buggy-coder`
2. LLM sees context in prompt but doesn't extract structured variables
3. LLM generates actions without populating required parameters
4. Tool invocation fails with "missing required fields"

## Proposed Solution: Two-Stage Parameter Resolution

### Stage 1: Context Extraction (Pre-LLM)
Extract structured variables from context BEFORE planning:

```python
def extract_action_context(
    user_context: UserContext,
    github_context: GithubContext,
    mcp_resources: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract all variables for parameter population."""
    context_vars = {}
    
    # GitHub: parse repo_url into owner/repo
    if github_context and github_context.repo_url:
        parts = github_context.repo_url.rstrip('/').split('/')
        if len(parts) >= 2:
            context_vars['github_owner'] = parts[-2]
            context_vars['github_repo'] = parts[-1].replace('.git', '')
    
    # MCP resources: flatten for easy reference
    for key, resource in mcp_resources.items():
        if isinstance(resource, dict):
            for field, value in resource.items():
                context_vars[f"{key}_{field}"] = value
    
    return context_vars
```

Include in LLM prompt explicitly:
```
**Available Context Variables:**
{
  "github_owner": "seer-engg",
  "github_repo": "buggy-coder",
  "asana_workspace_gid": "123..."
}

Use these when populating ActionStep params.
```

### Stage 2: Smart Parameter Completion (Post-LLM)
After LLM generates actions, auto-complete missing required parameters:

```python
async def complete_action_parameters(
    action: ActionStep,
    tool_entries: Dict[str, ToolEntry],
    context_vars: Dict[str, Any]
) -> ActionStep:
    """Intelligently complete missing required parameters."""
    
    schema = tool_entries[action.tool].pydantic_schema
    required_fields = schema.get('required', [])
    params_dict = json.loads(action.params or "{}")
    
    for field in required_fields:
        if field not in params_dict:
            inferred = _infer_parameter(field, action.tool, context_vars)
            if inferred:
                params_dict[field] = inferred
    
    return action
```

## Implementation Location
- Create: `shared/context_extractor.py`
- Create: `shared/parameter_completion.py`
- Modify: `agents/eval_agent/nodes/plan/provision_target.py` in `_plan_provisioning_actions()`

## Status
**DEFERRED** - Documented for later implementation. Focusing on dynamic cleanup first.

## Related Issues
- Log line 645-653: `github_list_pull_requests` called with empty params
- Log line 895-900: `asana_delete_project` called with wrong param name (`id` vs `project_gid`)

