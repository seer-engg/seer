#!/usr/bin/env python3
"""
Database Initialization Script for Seer

Initializes the SQLite database with Tortoise ORM and optionally migrates existing data.
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.database import init_db, get_db, close_db
from shared.models import Thread, Message, Event, AgentActivity, EvalSuite, TestResult, Subscriber


async def init_database(db_path: str = None):
    """Initialize database with schema"""
    print("🔧 Initializing Seer database with Tortoise ORM...")
    
    db = await init_db(db_path)
    print(f"✅ Database initialized at: {db.db_path}")
    
    # Test connection and list tables
    from tortoise import Tortoise
    conn = Tortoise.get_connection("default")
    
    # Get table names
    tables = [
        "threads", "messages", "events", "agent_activities",
        "eval_suites", "test_results", "subscribers"
    ]
    
    print(f"\n📊 Created {len(tables)} tables:")
    for table_name in tables:
        try:
            if table_name == "threads":
                count = await Thread.all().count()
            elif table_name == "messages":
                count = await Message.all().count()
            elif table_name == "events":
                count = await Event.all().count()
            elif table_name == "agent_activities":
                count = await AgentActivity.all().count()
            elif table_name == "eval_suites":
                count = await EvalSuite.all().count()
            elif table_name == "test_results":
                count = await TestResult.all().count()
            elif table_name == "subscribers":
                count = await Subscriber.all().count()
            else:
                count = 0
            print(f"   - {table_name}: {count} rows")
        except Exception as e:
            print(f"   - {table_name}: error ({e})")
    
    print("\n✅ Database ready!")


async def migrate_eval_files():
    """Migrate existing eval JSON files to database"""
    from pathlib import Path
    import json
    import uuid
    from datetime import datetime
    
    print("\n📦 Migrating existing eval files to database...")
    
    db = get_db()
    eval_agent_dir = Path(__file__).parent / "agents" / "eval_agent" / "generated_evals"
    
    if not eval_agent_dir.exists():
        print("⚠️  No generated_evals directory found, skipping migration")
        return
    
    # Find all eval suite files (not results files)
    eval_files = [f for f in eval_agent_dir.glob("eval_*.json") if not f.name.startswith("results_")]
    result_files = [f for f in eval_agent_dir.glob("results_*.json")]
    
    print(f"📋 Found {len(eval_files)} eval suite(s) and {len(result_files)} result file(s)")
    
    # Migrate eval suites
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            suite_id = eval_data.get('id')
            spec_name = eval_data.get('spec_name')
            spec_version = eval_data.get('spec_version')
            test_cases = eval_data.get('test_cases', [])
            
            # Check if already exists
            existing = await db.get_eval_suite(suite_id)
            if existing:
                print(f"   ⏭️  Skipping {suite_id} (already in database)")
                continue
            
            # Save to database
            await db.save_eval_suite(
                suite_id=suite_id,
                spec_name=spec_name,
                spec_version=spec_version,
                test_cases=test_cases,
                thread_id=None,  # No thread ID for migrated data
                target_agent_url="http://localhost:2024",  # Default URL
                target_agent_id="unknown"  # Unknown agent ID
            )
            
            print(f"   ✅ Migrated eval suite: {suite_id}")
        except Exception as e:
            print(f"   ❌ Failed to migrate {eval_file.name}: {e}")
    
    # Migrate test results
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            thread_id = result_data.get('thread_id')
            suite_id = result_data.get('eval_suite_id')
            results = result_data.get('results', [])
            
            if not suite_id or not results:
                print(f"   ⚠️  Skipping {result_file.name} (missing suite_id or results)")
                continue
            
            # Check if suite exists
            suite = await db.get_eval_suite(suite_id)
            if not suite:
                print(f"   ⚠️  Skipping {result_file.name} (suite {suite_id} not found)")
                continue
            
            # Save each result
            migrated = 0
            for result in results:
                try:
                    result_id = str(uuid.uuid4())
                    await db.save_test_result(
                        result_id=result_id,
                        suite_id=suite_id,
                        thread_id=thread_id or "migrated",
                        test_case_id=result.get('test_case_id'),
                        input_sent=result.get('input_sent'),
                        actual_output=result.get('actual_output'),
                        expected_behavior=result.get('expected_behavior'),
                        passed=result.get('passed'),
                        score=result.get('score'),
                        judge_reasoning=result.get('judge_reasoning')
                    )
                    migrated += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to migrate result: {e}")
            
            print(f"   ✅ Migrated {migrated} test results from: {result_file.name}")
        except Exception as e:
            print(f"   ❌ Failed to migrate {result_file.name}: {e}")
    
    print("\n✅ Migration complete!")


async def reset_database(db_path: str = None):
    """Reset database by deleting all data"""
    from tortoise import Tortoise
    
    print("🗑️  Resetting database...")
    
    # Initialize first
    db = await init_db(db_path)
    
    # Delete all data from tables
    await TestResult.all().delete()
    await EvalSuite.all().delete()
    await AgentActivity.all().delete()
    await Event.all().delete()
    await Message.all().delete()
    await Subscriber.all().delete()
    await Thread.all().delete()
    
    print("✅ Database reset complete!")


async def main_async(args):
    """Main async function"""
    try:
        # Handle reset
        if args.reset:
            if args.db_path:
                db_file = Path(args.db_path)
            else:
                db_file = Path(__file__).parent / "data" / "seer.db"
            
            if db_file.exists():
                print(f"🗑️  Deleting existing database: {db_file}")
                db_file.unlink()
        
        # Initialize database
        await init_database(args.db_path)
        
        # Migrate if requested
        if args.migrate:
            await migrate_eval_files()
        
        print("\n🎉 All done!")
        print("\n💡 Usage:")
        print("   - Database is automatically initialized when event bus starts")
        print("   - All new data will be persisted to the database")
        print("   - Use --migrate to import existing eval files")
        
    finally:
        # Close database connections
        await close_db()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Initialize Seer database with Tortoise ORM")
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database file (default: data/seer.db)"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate existing eval JSON files to database"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing database and create fresh"
    )
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
