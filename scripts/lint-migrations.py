#!/usr/bin/env python3
# pylint: disable=invalid-name  # Script uses kebab-case naming convention
"""Lint Aerich migration files for dangerous operations."""
import re
import sys
from pathlib import Path

DANGEROUS_PATTERNS = {
    "NOT NULL without DEFAULT": re.compile(
        r'ADD\s+(?:COLUMN\s+)?["\w]+\s+\w+\s+NOT NULL(?!\s+DEFAULT)',
        re.IGNORECASE
    ),
    "DROP COLUMN": re.compile(r'DROP\s+COLUMN', re.IGNORECASE),
    "RENAME COLUMN": re.compile(r'RENAME\s+COLUMN', re.IGNORECASE),
    "UNIQUE constraint": re.compile(
        r'ADD\s+CONSTRAINT.*UNIQUE|CREATE\s+UNIQUE\s+INDEX',
        re.IGNORECASE
    ),
    "Large UPDATE": re.compile(r'UPDATE\s+"\w+"\s+SET', re.IGNORECASE),
}

def lint_migration(file_path: Path) -> list[str]:
    """Check a migration file for dangerous patterns."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    issues = []
    for name, pattern in DANGEROUS_PATTERNS.items():
        if pattern.search(content):
            issues.append(f"⚠️  {name} detected in {file_path.name}")

    return issues

def main():
    migrations_dir = Path(__file__).parent.parent / "migrations" / "models"

    # Get all migration files
    migration_files = sorted(migrations_dir.glob("*.py"))

    # Only lint recent migrations (last 5) to avoid noise
    recent_migrations = migration_files[-5:] if len(migration_files) > 5 else migration_files

    all_issues = []
    for migration_file in recent_migrations:
        issues = lint_migration(migration_file)
        all_issues.extend(issues)

    if all_issues:
        print("❌ Migration linting failed:\n")
        for issue in all_issues:
            print(f"  {issue}")
        print("\nPlease review these migrations for safety.")
        print("See documentation/docs/guides/safe-migrations.md for guidance.")
        sys.exit(1)
    else:
        print("✅ Migration linting passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
