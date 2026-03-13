"""Tests for the lint-migrations script."""
import sys
from pathlib import Path

import pytest


# Import the script as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
# pylint: disable=wrong-import-position,import-error  # Script import after sys.path modification
from importlib import import_module

lint_migrations = import_module("lint-migrations")


@pytest.mark.unit
class TestCheckMigrationOrder:
    """Tests for the check_migration_order function."""

    def test_sequential_migrations_pass(self, tmp_path: Path):
        """Migrations with increasing timestamps should pass."""
        # Create mock migration files with sequential timestamps
        (tmp_path / "1_20260101120000_init.py").touch()
        (tmp_path / "2_20260102120000_second.py").touch()
        (tmp_path / "3_20260103120000_third.py").touch()

        issues = lint_migrations.check_migration_order(tmp_path)
        assert issues == []

    def test_out_of_order_migration_detected(self, tmp_path: Path):
        """Migration with earlier timestamp than previous should be flagged."""
        # Migration #2 has timestamp AFTER #3 (wrong order)
        (tmp_path / "1_20260101120000_init.py").touch()
        (tmp_path / "2_20260103120000_second.py").touch()  # Jan 3
        (tmp_path / "3_20260102120000_third.py").touch()   # Jan 2 - EARLIER!

        issues = lint_migrations.check_migration_order(tmp_path)
        assert len(issues) == 1
        assert "OUT-OF-ORDER MIGRATION DETECTED" in issues[0]
        assert "#3" in issues[0]
        assert "3_20260102120000_third.py" in issues[0]

    def test_allowlisted_migrations_skipped(self, tmp_path: Path):
        """Migrations in the allowlist should not trigger issues."""
        # Create the specific out-of-order migration that's allowlisted
        (tmp_path / "47_20260310102817_previous.py").touch()
        (tmp_path / "48_20260309100000_global_variables.py").touch()  # Earlier timestamp but allowlisted

        issues = lint_migrations.check_migration_order(tmp_path)
        assert issues == []

    def test_non_standard_filenames_ignored(self, tmp_path: Path):
        """Files not matching migration pattern should be ignored."""
        (tmp_path / "__init__.py").touch()
        (tmp_path / "some_file.py").touch()
        (tmp_path / "1_20260101120000_valid.py").touch()

        # Should not crash and should find no issues
        issues = lint_migrations.check_migration_order(tmp_path)
        assert issues == []

    def test_multiple_out_of_order_migrations(self, tmp_path: Path):
        """Multiple out-of-order migrations should all be detected."""
        (tmp_path / "1_20260110120000_init.py").touch()      # Jan 10
        (tmp_path / "2_20260105120000_second.py").touch()    # Jan 5 - OUT OF ORDER
        (tmp_path / "3_20260115120000_third.py").touch()     # Jan 15
        (tmp_path / "4_20260108120000_fourth.py").touch()    # Jan 8 - OUT OF ORDER

        issues = lint_migrations.check_migration_order(tmp_path)
        assert len(issues) == 2

    def test_error_message_includes_fix_instructions(self, tmp_path: Path):
        """Error messages should include helpful fix instructions."""
        (tmp_path / "1_20260110120000_init.py").touch()
        (tmp_path / "2_20260105120000_second.py").touch()

        issues = lint_migrations.check_migration_order(tmp_path)
        assert len(issues) == 1
        assert "FIX:" in issues[0]
        assert "MODELS_STATE" in issues[0]

    def test_timestamps_formatted_readably(self, tmp_path: Path):
        """Error messages should have human-readable timestamps."""
        (tmp_path / "1_20260315143052_init.py").touch()  # 2026-03-15 14:30:52
        (tmp_path / "2_20260310091520_second.py").touch()  # Earlier

        issues = lint_migrations.check_migration_order(tmp_path)
        assert len(issues) == 1
        # Check formatted timestamp is present
        assert "2026-03-10 09:15:20" in issues[0]
        assert "2026-03-15 14:30:52" in issues[0]
