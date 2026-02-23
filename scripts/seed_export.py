#!/usr/bin/env python3
"""
Export OAuth seed data to S3.

Exports users, OAuth connections, integration resources, and secrets
from the database to an S3 JSON file for environment setup.

Usage:
    uv run scripts/seed_export.py
    uv run scripts/seed_export.py --filename backup-2025-02-22.json
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Add src to Python path for imports (must happen before seer imports)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# pylint: disable=wrong-import-position  # Reason: sys.path must be modified before importing seer modules
from seer.database import close_db, init_db
from seer.database.seed.exporter import SeedExporter
from seer.logger import get_logger

logger = get_logger("scripts.seed_export")


async def main(filename: str, merge: bool) -> int:
    """
    Export seed data to S3.

    Args:
        filename: Output filename in S3.
        merge: If True, merge with existing S3 data (upsert).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        await init_db()
        logger.info("Database initialized")

        exporter = SeedExporter()
        s3_path = await exporter.export_all(filename=filename, merge=merge)

        mode = "merged with" if merge else "exported to"
        print(f"Seed data {mode}: {s3_path}")
        return 0

    except ValueError as e:
        # Configuration errors (e.g., missing S3 bucket)
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: CLI needs to catch all errors for graceful exit
        logger.exception("Export failed: %s", e)
        print(f"Export failed: {e}", file=sys.stderr)
        return 1

    finally:
        await close_db()
        logger.info("Database connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export OAuth seed data to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: merge/upsert (combine current DB with existing S3 data)
    uv run scripts/seed_export.py

    # Full overwrite (replace S3 file completely)
    uv run scripts/seed_export.py --overwrite

    # Custom filename
    uv run scripts/seed_export.py --filename backup-$(date +%Y-%m-%d).json
        """,
    )
    parser.add_argument(
        "--filename",
        default="oauth-seed-data.json",
        help="Output filename in S3 (default: oauth-seed-data.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite S3 file completely instead of merging (default: merge/upsert)",
    )
    args = parser.parse_args()

    # Default is merge=True, --overwrite sets merge=False
    exit_code = asyncio.run(main(args.filename, merge=not args.overwrite))
    sys.exit(exit_code)
