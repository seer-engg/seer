#!/usr/bin/env python3
# pylint: disable=invalid-name  # Script uses kebab-case naming convention
"""
Mutation testing runner for seer core modules.

This script provides targeted mutation testing for high-priority modules
in the workflow compiler and expression evaluator.

Usage:
    uv run python scripts/run_mutation_tests.py run evaluator
    uv run python scripts/run_mutation_tests.py run parser
    uv run python scripts/run_mutation_tests.py browse
    uv run python scripts/run_mutation_tests.py report
    uv run python scripts/run_mutation_tests.py list
"""
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModuleConfig:
    """Configuration for a module to mutate."""
    name: str
    source_path: str
    test_paths: list[str]
    priority: int
    description: str


# Module configurations ordered by priority (1 = highest)
# Priority is based on: module criticality, test coverage, and code size
MODULES = {
    "evaluator": ModuleConfig(
        name="evaluator",
        source_path="src/seer/core/expr/evaluator.py",
        test_paths=["tests/unit/core/test_evaluator.py"],
        priority=1,
        description="Expression evaluator - critical for all workflow expressions",
    ),
    "lower_control_flow": ModuleConfig(
        name="lower_control_flow",
        source_path="src/seer/core/compiler/lower_control_flow.py",
        test_paths=[
            "tests/unit/core/test_control_flow_traces.py",
            "tests/unit/core/test_for_each_loops.py",
            "tests/unit/core/test_if_conditionals.py",
            "tests/unit/core/test_nested_control_flow.py",
            "tests/unit/core/test_loop_body_detection.py",
        ],
        priority=2,
        description="Control flow lowering - transforms if/for blocks to graph nodes",
    ),
    "emit_langgraph": ModuleConfig(
        name="emit_langgraph",
        source_path="src/seer/core/compiler/emit_langgraph.py",
        test_paths=["tests/unit/core/"],
        priority=3,
        description="LangGraph emission - generates executable workflow graphs",
    ),
    "parser": ModuleConfig(
        name="parser",
        source_path="src/seer/core/expr/parser.py",
        test_paths=["tests/unit/core/test_expr_parser.py"],
        priority=4,
        description="Expression parser - parses {{ expr }} syntax",
    ),
    "type_env": ModuleConfig(
        name="type_env",
        source_path="src/seer/core/compiler/type_env.py",
        test_paths=["tests/unit/core/test_type_env.py"],
        priority=5,
        description="Type environment - tracks variable types during compilation",
    ),
    "validate_refs": ModuleConfig(
        name="validate_refs",
        source_path="src/seer/core/compiler/validate_refs.py",
        test_paths=["tests/unit/core/test_validate_refs.py"],
        priority=6,
        description="Reference validation - ensures all variable references are valid",
    ),
}


def run_mutmut(module_name: str | None = None, max_children: int | None = None) -> int:
    """Run mutmut on a specific module or all configured modules.

    For mutmut 3.x, we use the configuration from pyproject.toml.
    Per-module runs require modifying the config temporarily.
    """
    cmd = ["uv", "run", "mutmut", "run"]

    if max_children:
        cmd.extend(["--max-children", str(max_children)])

    if module_name:
        if module_name not in MODULES:
            print(f"❌ Unknown module: {module_name}")
            print(f"Available modules: {', '.join(MODULES.keys())}")
            return 1

        config = MODULES[module_name]
        print(f"🧬 Running mutation tests on {config.name}")
        print(f"   Source: {config.source_path}")
        print(f"   Tests: {', '.join(config.test_paths)}")
        print()
        print("Note: mutmut 3.x reads config from pyproject.toml.")
        print("For single-module runs, edit [tool.mutmut].paths_to_mutate temporarily.")
        print()

        # For now, just inform the user how to configure
        # A full implementation would require tomli_w which may not be installed
        print(f"Suggested pyproject.toml config for {module_name}:")
        print(f'  paths_to_mutate = ["{config.source_path}"]')
        print()

    # Run mutmut with default config from pyproject.toml
    print("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def show_results() -> int:
    """Show mutmut results summary."""
    cmd = ["uv", "run", "mutmut", "results"]
    result = subprocess.run(cmd, check=False)
    return result.returncode


def generate_html_report() -> int:
    """Generate HTML report in mutmut-report/ directory."""
    print("📊 Generating HTML report...")
    cmd = ["uv", "run", "mutmut", "html"]
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        report_path = Path("mutmut-report/index.html")
        if report_path.exists():
            print(f"✅ Report generated: {report_path.absolute()}")
    return result.returncode


def browse_results() -> int:
    """Open interactive browser for mutmut results."""
    cmd = ["uv", "run", "mutmut", "browse"]
    result = subprocess.run(cmd, check=False)
    return result.returncode


def list_modules() -> None:
    """List available modules with their priority and description."""
    print("Available modules for mutation testing:")
    print()
    print(f"{'Priority':<10} {'Module':<20} {'Description'}")
    print("-" * 80)

    sorted_modules = sorted(MODULES.values(), key=lambda m: m.priority)
    for config in sorted_modules:
        print(f"{config.priority:<10} {config.name:<20} {config.description}")

    print()
    print("To run mutation tests on a specific module, update pyproject.toml:")
    print('  [tool.mutmut]')
    print('  paths_to_mutate = ["src/seer/core/expr/evaluator.py"]')
    print()
    print("Then run: uv run mutmut run")
    print()
    print("Or run on all configured modules (expr/ + compiler/):")
    print("  uv run python scripts/run_mutation_tests.py run")


def main():
    parser = argparse.ArgumentParser(
        description="Run mutation tests on seer core modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Run mutation tests (uses pyproject.toml config):
        uv run python scripts/run_mutation_tests.py run

    Run with limited parallelism:
        uv run python scripts/run_mutation_tests.py run --max-children 4

    Show results:
        uv run python scripts/run_mutation_tests.py results

    Generate HTML report:
        uv run python scripts/run_mutation_tests.py report

    Open interactive browser:
        uv run python scripts/run_mutation_tests.py browse

    List available modules:
        uv run python scripts/run_mutation_tests.py list

Mutation test legend:
    🎉 Killed    - Test caught the mutation (good!)
    🫥 Equivalent - Mutation is functionally identical
    🙁 Survived  - Test did NOT catch the mutation (needs improvement)
    ⏰ Timeout   - Test took too long
    🤔 Suspicious - Unexpected behavior
    🔇 Skipped   - Mutant was skipped
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run command
    run_parser = subparsers.add_parser("run", help="Run mutation tests")
    run_parser.add_argument(
        "module",
        nargs="?",
        help="Module to test (optional, runs all if not specified)",
    )
    run_parser.add_argument(
        "--max-children",
        type=int,
        help="Maximum parallel test processes",
    )

    # results command
    subparsers.add_parser("results", help="Show mutation test results")

    # report command
    subparsers.add_parser("report", help="Generate HTML report")

    # browse command
    subparsers.add_parser("browse", help="Open interactive results browser")

    # list command
    subparsers.add_parser("list", help="List available modules")

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(run_mutmut(args.module, getattr(args, "max_children", None)))
    elif args.command == "results":
        sys.exit(show_results())
    elif args.command == "report":
        sys.exit(generate_html_report())
    elif args.command == "browse":
        sys.exit(browse_results())
    elif args.command == "list":
        list_modules()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
