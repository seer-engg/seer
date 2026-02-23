#!/usr/bin/env python3
"""
Import OAuth seed data from S3.

Imports users, OAuth connections, integration resources, and secrets
from an S3 JSON file into the database using upsert patterns.

Usage:
    uv run scripts/seed_import.py
    uv run scripts/seed_import.py --filename backup-2025-02-22.json
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
from seer.database.seed.importer import SeedImporter
from seer.logger import get_logger

logger = get_logger("scripts.seed_import")


async def main(filename: str) -> int:
    """
    Import seed data from S3.

    Args:
        filename: Input filename in S3.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        await init_db()
        logger.info("Database initialized")

        importer = SeedImporter()
        result = await importer.import_all(filename=filename)

        print("Seed data imported successfully:")
        for table, count in result.items():
            print(f"  {table}: {count}")

        return 0

    except FileNotFoundError as e:
        print(f"Seed data not found: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        # Configuration errors (e.g., missing S3 bucket) or data errors
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: CLI needs to catch all errors for graceful exit
        logger.exception("Import failed: %s", e)
        print(f"Import failed: {e}", file=sys.stderr)
        return 1

    finally:
        await close_db()
        logger.info("Database connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import OAuth seed data from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run scripts/seed_import.py
    uv run scripts/seed_import.py --filename backup-2025-02-22.json
        """,
    )
    parser.add_argument(
        "--filename",
        default="oauth-seed-data.json",
        help="Input filename in S3 (default: oauth-seed-data.json)",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(args.filename))
    sys.exit(exit_code)
