#!/usr/bin/env python3
"""
Export the `WorkflowSpec` Pydantic model JSON schema to a file.

Run from the repository root with PYTHONPATH=. or install the package so imports resolve.
Example:
  PYTHONPATH=. python scripts/export_workflow_schema.py -o workflow_spec_schema.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure the repo root is importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_compiler.schema.models import WorkflowSpec


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WorkflowSpec JSON schema")
    parser.add_argument("-o", "--output", default="workflow_spec_schema.json", help="Output JSON file path")
    args = parser.parse_args()

    # Generate JSON Schema (Pydantic v2 API)
    schema = WorkflowSpec.model_json_schema(ref_template="#/definitions/{model}")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(schema, indent=2))

    print(f"Wrote WorkflowSpec JSON schema to: {out_path}")


if __name__ == "__main__":
    main()
