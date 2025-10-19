"""Orchestrator Agent - Simplified using modular design"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the simplified orchestrator
from simplified_graph import agent, graph, register_orchestrator

# Re-export for backward compatibility
__all__ = ['agent', 'graph', 'register_orchestrator']
