"""
Verification test for schema extraction fix.

This test ensures that tool schemas are properly extracted from MCP tools
where args_schema is already a dict (not a Pydantic model).
"""
import asyncio
import pytest
from shared.tool_service import get_tool_service
from shared.tools.schema_formatter import format_tool_schemas_for_llm


@pytest.mark.asyncio
async def test_github_tool_schemas_extracted():
    """Verify GitHub tools have schemas extracted correctly."""
    tool_service = get_tool_service()
    await tool_service.initialize(["github"])
    
    tool_entries = tool_service.get_tool_entries()
    
    # Should have loaded GitHub tools
    assert len(tool_entries) > 0, "No tool entries loaded"
    
    # Check github_list_pull_requests specifically (the tool that was failing)
    github_list_prs = tool_entries.get("github_list_pull_requests")
    assert github_list_prs is not None, "github_list_pull_requests not found"
    
    # Should have a schema now
    assert github_list_prs.pydantic_schema is not None, \
        "github_list_pull_requests should have a schema"
    
    # Schema should be a dict in JSON Schema format
    schema = github_list_prs.pydantic_schema
    assert isinstance(schema, dict), \
        f"Schema should be dict, got {type(schema)}"
    
    # Should have standard JSON Schema fields
    assert 'properties' in schema, "Schema should have 'properties'"
    assert 'required' in schema, "Schema should have 'required'"
    
    # Should know about owner and repo params
    properties = schema['properties']
    assert 'owner' in properties, "Should have 'owner' parameter"
    assert 'repo' in properties, "Should have 'repo' parameter"
    
    # Should know these are required
    required = schema['required']
    assert 'owner' in required, "'owner' should be required"
    assert 'repo' in required, "'repo' should be required"
    
    print(f"✓ github_list_pull_requests schema extracted correctly")
    print(f"  Properties: {list(properties.keys())}")
    print(f"  Required: {required}")


@pytest.mark.asyncio
async def test_schema_formatter_works():
    """Verify schema formatter can format extracted schemas."""
    tool_service = get_tool_service()
    await tool_service.initialize(["github"])
    
    tool_entries = tool_service.get_tool_entries()
    
    # Format schemas for LLM
    formatted = format_tool_schemas_for_llm(
        tool_entries,
        ["github_list_pull_requests"]
    )
    
    # Should NOT say "(No parameters)"
    assert "(No parameters)" not in formatted, \
        "Schema formatter should find parameters"
    
    # Should include owner and repo in output
    assert "owner" in formatted.lower(), "Should mention 'owner' param"
    assert "repo" in formatted.lower(), "Should mention 'repo' param"
    
    # Should show required params
    assert "Required Params:" in formatted, \
        "Should have 'Required Params' section"
    
    print(f"✓ Schema formatter works correctly")
    print(f"\nFormatted output:\n{formatted}")


@pytest.mark.asyncio
async def test_all_github_tools_have_schemas():
    """Verify ALL GitHub tools have schemas extracted."""
    tool_service = get_tool_service()
    await tool_service.initialize(["github"])
    
    tool_entries = tool_service.get_tool_entries()
    
    github_tools = {
        name: entry 
        for name, entry in tool_entries.items() 
        if name.startswith("github")
    }
    
    assert len(github_tools) > 0, "Should have GitHub tools"
    
    tools_without_schema = [
        name for name, entry in github_tools.items()
        if not entry.pydantic_schema
    ]
    
    assert len(tools_without_schema) == 0, \
        f"These GitHub tools are missing schemas: {tools_without_schema}"
    
    print(f"✓ All {len(github_tools)} GitHub tools have schemas")
    
    # Verify they're all dicts
    for name, entry in github_tools.items():
        assert isinstance(entry.pydantic_schema, dict), \
            f"{name} schema should be dict, got {type(entry.pydantic_schema)}"
    
    print(f"✓ All schemas are in dict format")


if __name__ == "__main__":
    print("=" * 80)
    print("SCHEMA EXTRACTION FIX VERIFICATION")
    print("=" * 80)
    print()
    
    asyncio.run(test_github_tool_schemas_extracted())
    print()
    asyncio.run(test_schema_formatter_works())
    print()
    asyncio.run(test_all_github_tools_have_schemas())
    
    print()
    print("=" * 80)
    print("✓ ALL TESTS PASSED - SCHEMA EXTRACTION FIX VERIFIED")
    print("=" * 80)

