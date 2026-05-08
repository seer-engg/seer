"""
CLI entry point for permit search pipeline.

Usage:
    python -m permit_search enrich "Brightmark"
    python -m permit_search export result.json
    python -m permit_search export result.json --dry-run
    python -m permit_search pipeline "Brightmark"       # full pipeline: search + enrich + export
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from seer.agents.permit_search.agent import create_permit_search_agent
from seer.agents.permit_search.enrich import enrich_permit_data
from seer.agents.permit_search.obsidian import export_to_obsidian
from seer.agents.permit_search.state import PermitSearchState
from seer.logger import get_logger

logger = get_logger(__name__)


async def _run_pipeline(company_name: str) -> dict[str, Any]:
    """Run the full pipeline: search WARP -> enrich with EXA."""
    logger.info("Starting pipeline for '%s'", company_name)

    # Step 1: Permit search
    agent = create_permit_search_agent()
    state: PermitSearchState = {
        "messages": [HumanMessage(content=f"Search for {company_name}")],
        "company_name": company_name,
        "resolved_company": None,
        "permit_results": None,
        "error": None,
    }
    result = await agent.ainvoke(state)

    if result.get("error") or not result.get("permit_results"):
        error = result.get("error", "No permit results found")
        logger.error("Permit search failed: %s", error)
        return {"error": error}

    permit_data = result["permit_results"]
    logger.info("Permit search complete: %s permits", permit_data.get("total_permits_found", 0))

    # Step 2: EXA enrichment
    enriched = await enrich_permit_data(permit_data)
    return enriched


async def _run_enrich(data: dict[str, Any]) -> dict[str, Any]:
    """Run EXA enrichment on permit data."""
    return await enrich_permit_data(data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WARP permit search pipeline: search, enrich, export to Obsidian"
    )
    sub = parser.add_subparsers(dest="command")

    # pipeline command
    p = sub.add_parser("pipeline", help="Run full pipeline: search + enrich + export")
    p.add_argument("company", help="Company name to search")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing files")

    # enrich command
    e = sub.add_parser("enrich", help="Enrich existing permit data with EXA")
    e.add_argument("input", help="JSON file with permit data or company name")

    # export command
    x = sub.add_parser("export", help="Export JSON data to Obsidian vault")
    x.add_argument("input", help="JSON file with enriched permit data")
    x.add_argument("--dry-run", action="store_true", help="Preview without writing files")

    args = parser.parse_args()

    if args.command == "pipeline":
        enriched = asyncio.run(_run_pipeline(args.company))
        if "error" in enriched:
            print(f"Error: {enriched['error']}", file=sys.stderr)
            return 1
        result = export_to_obsidian(enriched, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "enrich":
        input_path = Path(args.input)
        if input_path.suffix == ".json":
            data = json.loads(input_path.read_text(encoding="utf-8"))
        else:
            data = {"company_name": args.input, "permits": [], "contacts": []}
        enriched = asyncio.run(_run_enrich(data))
        print(json.dumps(enriched, indent=2, default=str))
        return 0

    if args.command == "export":
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = export_to_obsidian(data, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
