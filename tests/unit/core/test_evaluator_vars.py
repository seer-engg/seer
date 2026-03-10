"""Tests for ${vars.*} expression resolution in the evaluator."""
import pytest

from seer.core.expr.evaluator import (
    EvaluationContext,
    EvaluationError,
    render_template,
    resolve_reference,
)
from seer.core.expr.parser import parse_reference_string


@pytest.mark.unit
class TestVarsResolution:
    def _ctx(self, vars_dict: dict | None = None) -> EvaluationContext:
        return EvaluationContext(state={}, locals={}, vars=vars_dict)

    def test_resolve_vars_root(self):
        ctx = self._ctx({"api_key": "sk-123", "base_url": "https://example.com"})
        ref = parse_reference_string("vars")
        result = resolve_reference(ctx, ref)
        assert result == {"api_key": "sk-123", "base_url": "https://example.com"}

    def test_resolve_vars_property(self):
        ctx = self._ctx({"api_key": "sk-123"})
        result = render_template(ctx, "${vars.api_key}")
        assert result == "sk-123"

    def test_resolve_vars_in_template(self):
        ctx = self._ctx({"host": "example.com"})
        result = render_template(ctx, "https://${vars.host}/api")
        assert result == "https://example.com/api"

    def test_resolve_vars_missing_key_raises(self):
        ctx = self._ctx({"a": "1"})
        with pytest.raises(EvaluationError, match="Property 'b' not found"):
            render_template(ctx, "${vars.b}")

    def test_resolve_vars_none_raises_unknown_root(self):
        ctx = self._ctx(None)
        with pytest.raises(EvaluationError, match="Unknown reference root 'vars'"):
            render_template(ctx, "${vars.x}")

    def test_vars_does_not_shadow_state(self):
        """vars is resolved via the 'vars' root, not from state."""
        ctx = EvaluationContext(
            state={"node1": {"out": "from_state"}},
            locals={},
            vars={"my_var": "from_vars"},
        )
        assert render_template(ctx, "${node1.out}") == "from_state"
        assert render_template(ctx, "${vars.my_var}") == "from_vars"
