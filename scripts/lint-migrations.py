#!/usr/bin/env python3
# pylint: disable=invalid-name  # Script uses kebab-case naming convention
"""Lint Aerich migration files for dangerous operations."""
import re
import sys
from pathlib import Path

# Migrations that contain unsafe patterns but are already deployed to production.
# These migrations predate the linter and were accepted before safety checks existed.
# Adding new migrations to this list should be RARE and require explicit approval.
# Each entry should include:
# - Migration filename
# - Date added to allowlist
# - Reason why it's unsafe but acceptable
ALLOWED_UNSAFE_MIGRATIONS = {
    "9_20260109055152_add_form_feilds.py": {
        "added_to_allowlist": "2026-01-23",
        "reason": "Contains DROP COLUMN in downgrade(). Already deployed to production on main/dev since Jan 9, 2026. Predates linter introduction.",
        "unsafe_patterns": ["DROP COLUMN"]
    },
    # Future entries should be added here only after careful review
    # Example format:
    # "10_YYYYMMDDHHMMSS_migration_name.py": {
    #     "added_to_allowlist": "YYYY-MM-DD",
    #     "reason": "Brief explanation of why this is safe despite being flagged",
    #     "unsafe_patterns": ["PATTERN_NAME"]
    # },
}

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

def is_migration_allowed(file_path: Path) -> bool:
    """Check if a migration is in the allowlist of known safe but flagged migrations."""
    return file_path.name in ALLOWED_UNSAFE_MIGRATIONS

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
    skipped_allowed = []

    for migration_file in recent_migrations:
        # Skip migrations in the allowlist
        if is_migration_allowed(migration_file):
            skipped_allowed.append(migration_file.name)
            continue

        issues = lint_migration(migration_file)
        all_issues.extend(issues)

    # Show what was skipped (for transparency)
    if skipped_allowed:
        print(f"ℹ️  Skipped {len(skipped_allowed)} allowed legacy migration(s):")
        for name in skipped_allowed:
            info = ALLOWED_UNSAFE_MIGRATIONS[name]
            print(f"   - {name}: {info['reason']}")
        print()

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
