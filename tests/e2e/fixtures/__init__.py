# pylint: disable=import-outside-toplevel
# Reason: Fixtures use lazy imports to avoid import order issues
"""
E2E Real test fixtures package.

Provides fixtures for true end-to-end testing with real infrastructure:
- PostgreSQL with pgvector via Testcontainers
- Redis/Valkey via Testcontainers
- Taskiq in-process execution
- Authenticated API clients

NOTE: Fixtures are NOT imported at module level to prevent triggering
seer module imports before the test environment is configured.
Pytest discovers fixtures automatically from the .py files in this directory.
"""
# No imports here - pytest discovers fixtures from the files directly
