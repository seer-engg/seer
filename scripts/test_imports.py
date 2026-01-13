#!/usr/bin/env python3
"""Pre-commit hook to test that modified Python files can be imported.

Catches circular imports and basic import errors before commit.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _should_skip_file(path: Path) -> bool:
    """Check if file should be skipped."""
    if path.suffix != ".py":
        return True
    if path.name.startswith("test_"):
        return True
    if not path.exists():
        return True
    return False


def _path_to_module(path: Path) -> str | None:
    """Convert file path to module name, or None if invalid."""
    parts = path.with_suffix("").parts
    # Remove any leading path components that aren't part of the module
    if parts and parts[0] in (".", ".."):
        parts = parts[1:]
    module_name = ".".join(parts)
    # Skip if not a valid Python module path
    if not module_name or module_name.startswith("."):
        return None
    return module_name


def test_imports(files):
    """Attempt to import each modified Python file."""
    errors = []
    for filepath in files:
        path = Path(filepath)
        if _should_skip_file(path):
            continue

        module_name = _path_to_module(path)
        if not module_name:
            continue

        try:
            __import__(module_name)
        except ImportError as e:
            errors.append(f"{filepath}: {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Catch all import-time errors
            errors.append(f"{filepath}: {type(e).__name__}: {e}")

    if errors:
        print("Import test failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(test_imports(sys.argv[1:]))
