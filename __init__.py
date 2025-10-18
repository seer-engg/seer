"""
Seer - AI Agent Evaluation Platform

Event-driven multi-agent system for evaluating AI agents through blackbox testing.

Main Components:
- Event Bus: FastAPI-based message broker
- Customer Success Agent: User interaction handler
- Eval Agent: Spec generation, test generation, and A2A testing

Usage:
    # Launch everything
    python run.py
    
    # Then open:
    # - UI: http://localhost:8501
    # Use the "Agent Threads" tab to debug agent conversations
"""

__version__ = "1.0.0"
__all__ = []

