"""Validate all golden examples pass Pydantic schema validation."""
from __future__ import annotations

import pytest

from seer.agents.nexus.golden_examples import GOLDEN_EXAMPLES
from seer.core.compiler.parse import parse_workflow_spec


@pytest.mark.parametrize(
    "example",
    GOLDEN_EXAMPLES,
    ids=[f"example_{i+1}" for i in range(len(GOLDEN_EXAMPLES))],
)
def test_golden_example_parses(example):
    spec = parse_workflow_spec(example.spec)
    assert len(spec.nodes) > 0
    assert len(spec.triggers) > 0
