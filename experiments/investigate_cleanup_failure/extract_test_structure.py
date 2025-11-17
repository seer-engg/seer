"""
Extract and analyze the test structure from the log to understand what was generated.
"""

# From log line 717, the provision_actions are:
test_provision_actions = [
    {
        "tool": "github_create_an_organization_repository",
        "params": '{"auto_init": true, "description": "Test repo for PR label edge case", "has_issues": true, "name": "label-edgecase-repo", "org": "seer-engg", "private": false, "visibility": "public"}',
        "assign_to_var": "repo",
        "assert_field": "",
        "assert_expected": ""
    },
    {
        "tool": "github_create_a_pull_request",
        "params": '{"base": "main", "body": "Implements feature Z. Related to Asana task https://app.asana.com/0/1211928407052666/1122334455667788", "draft": false, "head": "feature/label-empty-string", "owner": "seer-engg", "repo": "label-edgecase-repo", "title": "Feature Z: handle empty label name"}',
        "assign_to_var": "pr",
        "assert_field": "",
        "assert_expected": ""
    },
    {
        "tool": "github_add_labels_to_an_issue",
        "params": '{"issue_number": 1, "labels": ["", "valid-label"], "owner": "seer-engg", "repo": "label-edgecase-repo"}',
        "assign_to_var": "",
        "assert_field": "",
        "assert_expected": ""
    }
]

print("=" * 80)
print("ANALYSIS: Generated Test Provision Actions")
print("=" * 80)
print()

for i, action in enumerate(test_provision_actions, 1):
    print(f"Step {i}: {action['tool']}")
    import json
    params = json.loads(action['params'])
    print(f"  Parameters: {json.dumps(params, indent=4)}")
    if action['assign_to_var']:
        print(f"  Assigns to variable: {action['assign_to_var']}")
    print()

print("=" * 80)
print("ROOT CAUSE IDENTIFICATION")
print("=" * 80)
print()
print("ISSUE: Step 2 tries to create a PR from branch 'feature/label-empty-string'")
print("PROBLEM: This branch does NOT exist!")
print()
print("SEQUENCE ANALYSIS:")
print("  1. ✓ Creates repository with auto_init=true (creates 'main' branch only)")
print("  2. ✗ Tries to create PR with head='feature/label-empty-string' (FAILS - branch doesn't exist)")
print("  3. - Never reached (Step 2 fails)")
print()
print("REQUIRED FIX:")
print("  The test generation MUST include a step to create the branch before creating the PR.")
print("  Missing actions between Step 1 and 2:")
print("    - github_create_or_update_file_contents (to create a commit on the new branch)")
print("    - OR github_create_a_reference (to create the branch)")
print()
print("GITHUB API REQUIREMENT:")
print("  To create a PR, BOTH 'base' and 'head' branches must already exist in the repository.")
print("  The head branch must have at least one commit that differs from base.")
print()

