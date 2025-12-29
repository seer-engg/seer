import pytest

from api.workflows import models as api_models
from api.workflows import services


@pytest.mark.asyncio
async def test_expression_helpers(db_user, workflow_spec):
    """Ensures expression suggestion and typechecking cover workflow inputs."""
    suggest_request = api_models.ExpressionSuggestRequest(
        spec=workflow_spec,
        cursor_context=api_models.ExpressionCursorContext(
            node_id="set_issue_title",
            field="prompt",
            prefix="${inputs.",
        ),
    )
    suggestions = services.suggest_expression(db_user, suggest_request)
    assert any(s.label == "user_id" for s in suggestions.suggestions)

    typecheck_request = api_models.ExpressionTypecheckRequest(
        spec=workflow_spec,
        expression="${inputs.user_id}",
    )
    type_resp = services.typecheck_expression(db_user, typecheck_request)
    assert type_resp.ok
    assert type_resp.type == {"type": "integer"}

