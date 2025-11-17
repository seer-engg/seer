"""Test that angle-bracket placeholders are caught at validation time."""
import pytest
from shared.schema import DatasetExample, ExpectedOutput, ActionStep
from agents.eval_agent.nodes.plan.validate_generated_actions import _validate_generated_actions


def test_angle_bracket_placeholder_detected():
    """Angle-bracket placeholders should be rejected during validation."""
    example = DatasetExample(
        example_id="test-123",
        reasoning="Test",
        input_message="Test input",
        expected_output=ExpectedOutput(
            assert_actions=[
                ActionStep(
                    tool="GITHUB_GET_A_PULL_REQUEST",
                    params='{"owner": "<owner>", "repo": "<repo>", "pull_number": "<pr_number>"}',
                    assign_to_var="pr",
                    assert_field="",
                    assert_expected=""
                )
            ]
        ),
        status="active"
    )
    
    # Should raise ValueError about invalid placeholder
    with pytest.raises(ValueError, match="invalid placeholder.*<owner>.*Use \\[var:name\\] or \\[resource:name\\]"):
        _validate_generated_actions([example], {
            "github_get_a_pull_request": type('ToolEntry', (), {
                'name': 'GITHUB_GET_A_PULL_REQUEST',
                'pydantic_schema': {
                    'properties': {'owner': {}, 'repo': {}, 'pull_number': {}},
                    'required': ['owner', 'repo', 'pull_number']
                },
                'description': 'Test tool'
            })()
        })


def test_valid_variable_syntax_accepted():
    """Proper [var:...] syntax should pass validation."""
    example = DatasetExample(
        example_id="test-456",
        reasoning="Test",
        input_message="Test input",
        expected_output=ExpectedOutput(
            assert_actions=[
                ActionStep(
                    tool="GITHUB_GET_A_PULL_REQUEST",
                    params='{"owner": "[resource:github_owner]", "repo": "[resource:github_repo]", "pull_number": "[var:pr_data.number]"}',
                    assign_to_var="pr",
                    assert_field="",
                    assert_expected=""
                )
            ]
        ),
        status="active"
    )
    
    # Should NOT raise - valid syntax
    _validate_generated_actions([example], {
        "github_get_a_pull_request": type('ToolEntry', (), {
            'name': 'GITHUB_GET_A_PULL_REQUEST',
            'pydantic_schema': {
                'properties': {'owner': {}, 'repo': {}, 'pull_number': {}},
                'required': ['owner', 'repo', 'pull_number']
            },
            'description': 'Test tool'
        })()
    })


def test_runtime_injection_rejects_angle_brackets():
    """Runtime injection should fail fast on angle brackets."""
    from shared.test_runner.variable_injection import inject_variables
    
    params = {"owner": "<owner>", "repo": "my-repo"}
    
    with pytest.raises(ValueError, match="Invalid placeholder.*<owner>.*Use \\[var:name\\] or \\[resource:name\\]"):
        inject_variables(params, {}, {})

