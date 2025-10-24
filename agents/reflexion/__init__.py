"""
Reflexion Coding Agent

A self-improving coding agent using the reflexion pattern with test-driven evaluation:
- Actor (Coding Agent): Generates production-ready code using learned patterns
- Evaluator (Test Engineer): Creates unit tests and validates code quality
- Reflection (Senior Architect): Suggests coding paradigms to fix issues

The agent iterates until either:
1. All unit tests pass and code meets quality standards, OR
2. Maximum attempts are reached

Each reflection (coding patterns, best practices) is stored in persistent memory 
for continuous learning across threads and users.

Key Features:
- Test-driven code generation and refinement
- Persistent memory for coding patterns and lessons
- Natural conversation - user only sees code improvements
- Supports multiple programming languages
"""

from agents.reflexion.graph import graph

__all__ = ['graph']

