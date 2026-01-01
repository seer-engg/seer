from __future__ import annotations

import pytest

from workflow_compiler.errors import WorkflowCompilerError
from workflow_compiler.runtime.input_validation import coerce_inputs
from workflow_compiler.schema.models import InputDef, InputType, WorkflowSpec


def _spec_with_inputs(inputs: dict[str, InputDef]) -> WorkflowSpec:
    # Minimal workflow spec for validation tests.
    return WorkflowSpec(
        version="1",
        inputs=inputs,
        nodes=[],
    )


def test_coerce_inputs_casts_and_applies_defaults() -> None:
    spec = _spec_with_inputs(
        {
            "count": InputDef(type=InputType.integer, required=True),
            "ratio": InputDef(type=InputType.number, required=False, default=0.5),
            "feature": InputDef(type=InputType.boolean, required=False),
            "payload": InputDef(type=InputType.object, required=False),
            "items": InputDef(type=InputType.array, required=False),
            "message": InputDef(type=InputType.string, required=False),
        }
    )

    incoming = {
        "count": "42",
        "feature": "yes",
        "payload": '{"name": "seer"}',
        "items": "[1, 2, 3]",
        "message": 123,
        "extra": "preserve-me",
    }

    result = coerce_inputs(spec, incoming)

    assert result["count"] == 42
    assert result["ratio"] == pytest.approx(0.5)
    assert result["feature"] is True
    assert result["payload"] == {"name": "seer"}
    assert result["items"] == [1, 2, 3]
    assert result["message"] == "123"
    assert result["extra"] == "preserve-me"


def test_coerce_inputs_missing_required_raises() -> None:
    spec = _spec_with_inputs(
        {
            "required": InputDef(type=InputType.string, required=True),
        }
    )

    with pytest.raises(WorkflowCompilerError) as excinfo:
        coerce_inputs(spec, {})

    assert "required" in str(excinfo.value)


def test_coerce_inputs_invalid_bool_literal_raises() -> None:
    spec = _spec_with_inputs(
        {
            "flag": InputDef(type=InputType.boolean, required=True),
        }
    )

    with pytest.raises(WorkflowCompilerError) as excinfo:
        coerce_inputs(spec, {"flag": "maybe"})

    assert "boolean" in str(excinfo.value)

