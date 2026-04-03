#!/usr/bin/env python3
# pylint: disable=invalid-name  # Script uses kebab-case naming convention
"""Lint Aerich migration files for dangerous operations."""
import re
import sys
from pathlib import Path

# Migrations created out of numerical order that have been manually verified/fixed.
# When migrations are created out of order (e.g., #48 on March 9, then #46 on March 10),
# the MODELS_STATE chain breaks because aerich applies migrations in NUMBER order,
# but the MODELS_STATE was captured at an earlier point in time.
# Migrations in this allowlist have had their MODELS_STATE manually corrected.
ALLOWED_OUT_OF_ORDER_MIGRATIONS = {
    "48_20260309100000_global_variables.py": "MODELS_STATE manually updated 2026-03-11",
    "49_20260309110000_avatar_url_length.py": "MODELS_STATE manually updated 2026-03-11",
}

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
    "60_20260326175047_remove_workflow_is_active.py": {
        "added_to_allowlist": "2026-03-26",
        "reason": "Drops is_active column that was never checked in any execution path. Safe to remove.",
        "unsafe_patterns": ["DROP COLUMN"]
    },
    "63_20260403115350_add_provider_to_byok.py": {
        "added_to_allowlist": "2026-04-03",
        "reason": "DROP COLUMN only in downgrade() for rollback of new provider/provider_config columns. Upgrade is safe.",
        "unsafe_patterns": ["DROP COLUMN"]
    },
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


def check_migration_order(migrations_dir: Path) -> list[str]:
    """
    Check that migration numbers and timestamps are in chronological order.

    Aerich uses datetime.now() for timestamps (server local time, often UTC).
    If migration #N has a timestamp EARLIER than migration #(N-1), the MODELS_STATE
    chain will break because aerich applies migrations in NUMBER order, but the
    MODELS_STATE was captured at an earlier point in time.
    """
    migration_files = sorted(migrations_dir.glob("*.py"))

    # Parse migration info: (number, timestamp, filename)
    migrations = []
    pattern = re.compile(r'^(\d+)_(\d{14})_.*\.py$')

    for f in migration_files:
        match = pattern.match(f.name)
        if match:
            num = int(match.group(1))
            ts = match.group(2)  # YYYYMMDDHHMMSS format
            migrations.append((num, ts, f.name))

    # Sort by number and check timestamp ordering
    migrations.sort(key=lambda x: x[0])
    issues = []

    for i in range(1, len(migrations)):
        prev_num, prev_ts, prev_name = migrations[i - 1]
        curr_num, curr_ts, curr_name = migrations[i]

        # Skip if in allowlist
        if curr_name in ALLOWED_OUT_OF_ORDER_MIGRATIONS:
            continue

        # If current migration has lower timestamp than previous, it's out of order
        if curr_ts < prev_ts:
            # Format timestamps for readability
            prev_dt = f"{prev_ts[:4]}-{prev_ts[4:6]}-{prev_ts[6:8]} {prev_ts[8:10]}:{prev_ts[10:12]}:{prev_ts[12:14]}"
            curr_dt = f"{curr_ts[:4]}-{curr_ts[4:6]}-{curr_ts[6:8]} {curr_ts[8:10]}:{curr_ts[10:12]}:{curr_ts[12:14]}"

            issues.append(
                f"❌ OUT-OF-ORDER MIGRATION DETECTED\n"
                f"   Migration #{curr_num}: {curr_name}\n"
                f"   Created at: {curr_dt} (server time)\n"
                f"   \n"
                f"   But migration #{prev_num}: {prev_name}\n"
                f"   Was created at: {prev_dt} (server time)\n"
                f"   \n"
                f"   PROBLEM: #{curr_num} will apply AFTER #{prev_num}, but its MODELS_STATE\n"
                f"   was captured BEFORE #{prev_num} existed. This breaks aerich's state tracking.\n"
                f"   \n"
                f"   FIX: Update MODELS_STATE in #{curr_num} to match current Python models.\n"
                f"   See documentation for the 'reference migration' technique."
            )

    return issues


def main():
    migrations_dir = Path(__file__).parent.parent / "migrations" / "models"

    # Check migration ordering first (applies to ALL migrations)
    order_issues = check_migration_order(migrations_dir)
    if order_issues:
        print("❌ Migration ordering issues detected:\n")
        for issue in order_issues:
            print(f"  {issue}")
        print()

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

    if all_issues or order_issues:
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
