"""
Taskiq worker package for Seer.

Provides broker configuration and background tasks that mirror the API's
long-running operations (trigger polling, webhook dispatch, workflow runs).
"""

__all__ = ["broker"]
