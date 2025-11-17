"""
Experiment to verify the output structure from MCP tools and why cleanup fails.

This script simulates the cleanup inverse generation to understand:
1. What the actual MCP tool output structure looks like
2. Why 'owner.login' extraction fails
3. What the correct extraction path should be
"""

import json

# This is what we see in the log: line 663
# "Stored '{"successfull":true,"error":null,"data":{"data":{"'"
# The output is truncated, but we can see the structure

# Simulated MCP tool output based on log evidence
simulated_mcp_output = {
    "successfull": True,
    "error": None,
    "data": {
        "data": {
            "id": 123456789,
            "name": "label-edgecase-repo",
            "full_name": "seer-engg/label-edgecase-repo",
            "owner": {
                "login": "seer-engg",
                "id": 12345,
                "type": "Organization"
            },
            "private": False,
            "html_url": "https://github.com/seer-engg/label-edgecase-repo"
        }
    }
}

def extract_nested_field(data, field_path):
    """Replicate the extraction logic from cleanup_inverses.py"""
    if not isinstance(data, dict):
        return None
    
    parts = field_path.split(".")
    value = data
    
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    
    return value

# Test extraction with different paths
test_paths = [
    "owner.login",           # What LLM probably suggested
    "data.owner.login",      # One level of nesting
    "data.data.owner.login", # Full MCP nesting
    "name",                  # Direct field
    "data.name",             # One level
    "data.data.name",        # Full path
]

print("=" * 80)
print("EXPERIMENT: MCP Output Structure and Field Extraction")
print("=" * 80)
print()
print("Simulated MCP Output Structure:")
print(json.dumps(simulated_mcp_output, indent=2))
print()
print("-" * 80)
print("Testing field extraction with different paths:")
print("-" * 80)

for path in test_paths:
    result = extract_nested_field(simulated_mcp_output, path)
    status = "✓ SUCCESS" if result else "✗ FAILED"
    print(f"{status:12} | Path: {path:30} | Result: {result}")

print()
print("=" * 80)
print("FINDINGS:")
print("=" * 80)
print("1. MCP tools wrap GitHub API responses in: {successfull, error, data: {data: {...}}}")
print("2. Cleanup inverse needs to extract from 'data.data.FIELD' not just 'FIELD'")
print("3. Current cleanup system doesn't account for MCP wrapper structure")
print()

