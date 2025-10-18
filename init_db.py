#!/usr/bin/env python3
"""
Database Initialization Script for Seer

Initializes the SQLite database and optionally migrates existing data.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.database import init_db, get_db


def init_database(db_path: str = None):
    """Initialize database with schema"""
    print("🔧 Initializing Seer database...")
    
    db = init_db(db_path)
    print(f"✅ Database initialized at: {db.db_path}")
    
    # Test connection
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\n📊 Created {len(tables)} tables:")
        for table in tables:
            table_name = table['name']
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            count = cursor.fetchone()['count']
            print(f"   - {table_name}: {count} rows")
    
    print("\n✅ Database ready!")


def migrate_eval_files():
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
            existing = db.get_eval_suite(suite_id)
            if existing:
                print(f"   ⏭️  Skipping {suite_id} (already in database)")
                continue
            
            # Save to database
            db.save_eval_suite(
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
            suite = db.get_eval_suite(suite_id)
            if not suite:
                print(f"   ⚠️  Skipping {result_file.name} (suite {suite_id} not found)")
                continue
            
            # Save each result
            migrated = 0
            for result in results:
                try:
                    result_id = str(uuid.uuid4())
                    db.save_test_result(
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Initialize Seer database")
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
    init_database(args.db_path)
    
    # Migrate if requested
    if args.migrate:
        migrate_eval_files()
    
    print("\n🎉 All done!")
    print("\n💡 Usage:")
    print("   - Database is automatically initialized when event bus starts")
    print("   - All new data will be persisted to the database")
    print("   - Use --migrate to import existing eval files")


if __name__ == "__main__":
    main()

