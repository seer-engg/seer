"""
LLM Inverse Detection Experiment

Tests if an LLM can reliably identify inverse tool pairs (create/delete)
from tool names and descriptions alone.
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from shared.logger import get_logger

logger = get_logger("experiments.llm_inverse_detection")


class InversePair(BaseModel):
    """A detected inverse operation pair."""
    create_tool: str = Field(description="The tool that creates a resource")
    delete_tool: str = Field(description="The tool that deletes that resource")
    resource_type: str = Field(description="What resource type they operate on (e.g., 'project', 'task')")
    output_field: str = Field(description="Field in CREATE output that contains the resource ID")
    input_field: str = Field(description="Parameter in DELETE input that needs the resource ID")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    reasoning: str = Field(description="Why these are inverses")


class InverseDetectionResult(BaseModel):
    """Complete detection result."""
    pairs: List[InversePair] = Field(description="Detected inverse pairs")
    unmatched_creates: List[str] = Field(description="CREATE tools with no DELETE inverse found")
    unmatched_deletes: List[str] = Field(description="DELETE tools with no CREATE inverse found")


async def detect_inverse_pairs(tools: List[Dict[str, Any]]) -> InverseDetectionResult:
    """
    Use LLM to detect inverse pairs from tool data.
    
    This is the core experiment - can the LLM figure out inverses
    from names and descriptions alone?
    """
    logger.info(f"Running LLM inverse detection on {len(tools)} tools...")
    
    # Use a reasoning model for better analysis
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,  # Deterministic
    )
    
    # Format tools for LLM
    tool_list = []
    for t in tools:
        tool_list.append({
            "name": t["name"],
            "description": t["description"] or "No description"
        })
    
    prompt = f"""You are analyzing Asana API tools to identify INVERSE OPERATIONS.

DEFINITION: Two tools are inverses if one CREATES a resource and the other DELETES that same resource type.

AVAILABLE TOOLS:
{json.dumps(tool_list, indent=2)}

YOUR TASK:
1. Identify all CREATE tools (look for: create, add, new in the name/description)
2. For each CREATE tool, find its DELETE inverse
3. Determine how to map the resource ID from CREATE output to DELETE input

GUIDELINES:
- Match by resource type (e.g., "project", "task", "allocation")
- CREATE typically returns a field like "gid", "id", or "[resource]_gid"
- DELETE typically needs a parameter like "[resource]_gid", "id"
- Only include pairs where you're confident (>0.8)
- If a CREATE has no DELETE, list it in unmatched_creates
- If a DELETE has no CREATE, list it in unmatched_deletes

IMPORTANT:
- Be conservative - only mark as inverse if you're sure
- Consider the resource type carefully (allocation ≠ task)
- READ the descriptions to understand what each tool does

Return your analysis following the InverseDetectionResult schema."""
    
    structured_llm = llm.with_structured_output(InverseDetectionResult)
    result = await structured_llm.ainvoke(prompt)
    
    logger.info(f"LLM detected {len(result.pairs)} inverse pairs")
    logger.info(f"Unmatched CREATEs: {len(result.unmatched_creates)}")
    logger.info(f"Unmatched DELETEs: {len(result.unmatched_deletes)}")
    
    return result


def format_results_for_validation(result: InverseDetectionResult) -> str:
    """Format results in a human-readable table for validation."""
    output = []
    
    output.append("=" * 100)
    output.append("LLM INVERSE DETECTION RESULTS")
    output.append("=" * 100)
    output.append("")
    
    output.append(f"DETECTED INVERSE PAIRS: {len(result.pairs)}")
    output.append("-" * 100)
    output.append(f"{'#':<3} {'CREATE TOOL':<40} {'DELETE TOOL':<40} {'CONF':<6}")
    output.append("-" * 100)
    
    for i, pair in enumerate(result.pairs, 1):
        output.append(f"{i:<3} {pair.create_tool:<40} {pair.delete_tool:<40} {pair.confidence:.2f}")
        output.append(f"    Resource: {pair.resource_type}")
        output.append(f"    Mapping: {pair.output_field} -> {pair.input_field}")
        output.append(f"    Reasoning: {pair.reasoning}")
        output.append("")
    
    if result.unmatched_creates:
        output.append("")
        output.append(f"UNMATCHED CREATES ({len(result.unmatched_creates)}):")
        output.append("-" * 100)
        for tool in result.unmatched_creates:
            output.append(f"  - {tool}")
    
    if result.unmatched_deletes:
        output.append("")
        output.append(f"UNMATCHED DELETES ({len(result.unmatched_deletes)}):")
        output.append("-" * 100)
        for tool in result.unmatched_deletes:
            output.append(f"  - {tool}")
    
    output.append("")
    output.append("=" * 100)
    output.append("VALIDATION INSTRUCTIONS:")
    output.append("For each pair above, mark as:")
    output.append("  ✅ CORRECT - Delete truly undoes create")
    output.append("  ❌ WRONG   - Not inverses")
    output.append("  ⚠️  PARTIAL - Right idea but wrong field mapping")
    output.append("=" * 100)
    
    return "\n".join(output)


async def main():
    """Run the experiment."""
    # Load dumped tools
    tools_file = Path(__file__).parent / "asana_tools_dump.json"
    
    if not tools_file.exists():
        logger.error(f"Tool dump not found at {tools_file}. Run dump_asana_tools.py first.")
        return
    
    with open(tools_file) as f:
        tools = json.load(f)
    
    logger.info(f"Loaded {len(tools)} Asana tools from {tools_file}")
    
    # Run LLM detection
    result = await detect_inverse_pairs(tools)
    
    # Save results
    results_file = Path(__file__).parent / "llm_inverse_detection_results.json"
    with open(results_file, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    
    logger.info(f"Saved results to {results_file}")
    
    # Print validation table
    validation_output = format_results_for_validation(result)
    print("\n" + validation_output)
    
    # Save validation table to file
    validation_file = Path(__file__).parent / "validation_table.txt"
    with open(validation_file, "w") as f:
        f.write(validation_output)
    
    logger.info(f"Saved validation table to {validation_file}")


if __name__ == "__main__":
    asyncio.run(main())

