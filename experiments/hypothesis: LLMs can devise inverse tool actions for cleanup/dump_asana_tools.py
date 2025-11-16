"""Extract Asana tool schemas for inverse detection experiment."""
import asyncio
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.tool_service import get_tool_service
from shared.logger import get_logger

logger = get_logger("experiments.dump_asana_tools")


async def dump_asana_tools():
    """Extract all Asana tool schemas for the experiment."""
    logger.info("Initializing tool service for Asana...")
    tool_service = get_tool_service()
    await tool_service.initialize(["asana"])
    
    tool_entries = tool_service.get_tool_entries()
    tools_dict = tool_service.get_tools()
    
    # Filter Asana tools
    asana_tools = []
    for name, entry in tool_entries.items():
        if name.startswith("asana"):
            # Get the actual tool object for schema
            tool_obj = tools_dict.get(name)
            schema = entry.pydantic_schema
            
            # If schema is null, try to get it from the tool object
            if not schema and tool_obj:
                args_schema = getattr(tool_obj, "args_schema", None)
                if args_schema and hasattr(args_schema, "model_json_schema"):
                    schema = args_schema.model_json_schema()
            
            tool_data = {
                "name": entry.name,
                "description": entry.description,
                "schema": schema
            }
            asana_tools.append(tool_data)
    
    # Save to file
    output_path = Path(__file__).parent / "asana_tools_dump.json"
    with open(output_path, "w") as f:
        json.dump(asana_tools, f, indent=2)
    
    logger.info(f"Extracted {len(asana_tools)} Asana tools to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXTRACTED {len(asana_tools)} ASANA TOOLS")
    print(f"{'='*60}\n")
    
    # Group by operation type
    creates = [t for t in asana_tools if 'create' in t['name'].lower()]
    deletes = [t for t in asana_tools if 'delete' in t['name'].lower()]
    updates = [t for t in asana_tools if 'update' in t['name'].lower()]
    gets = [t for t in asana_tools if 'get' in t['name'].lower()]
    others = [t for t in asana_tools if t not in creates + deletes + updates + gets]
    
    print(f"CREATE operations: {len(creates)}")
    for t in creates:
        print(f"  - {t['name']}")
    
    print(f"\nDELETE operations: {len(deletes)}")
    for t in deletes:
        print(f"  - {t['name']}")
    
    print(f"\nUPDATE operations: {len(updates)}")
    print(f"\nGET operations: {len(gets)}")
    print(f"\nOTHER operations: {len(others)}")
    
    return asana_tools


if __name__ == "__main__":
    tools = asyncio.run(dump_asana_tools())

