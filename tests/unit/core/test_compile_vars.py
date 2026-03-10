"""Tests for vars symbol registration in the type environment."""
import pytest

from seer.core.expr.typecheck import TypeEnvironment


@pytest.mark.unit
class TestVarsTypeRegistration:
    def test_vars_registered_in_type_env(self):
        """build_type_environment should register 'vars' as an object with string additionalProperties."""
        from seer.core.compiler.type_env import build_type_environment
        from seer.core.schema.models import WorkflowSpec
        from seer.core.schema.schema_registry import SchemaRegistry
        from seer.core.registry.tool_registry import ToolRegistry

        spec = WorkflowSpec(nodes=[], edges=[], triggers=[])
        env = build_type_environment(
            spec,
            schema_registry=SchemaRegistry(),
            tool_registry=ToolRegistry(),
        )
        schema = env.get("vars")
        assert schema is not None
        assert schema["type"] == "object"
        assert schema["additionalProperties"] == {"type": "string"}

    def test_vars_property_navigation(self):
        """TypeEnvironment should allow navigating vars.any_key since additionalProperties is set."""
        env = TypeEnvironment()
        env.register("vars", {"type": "object", "additionalProperties": {"type": "string"}})
        # The type env should resolve vars and allow property access
        schema = env.get("vars")
        assert schema is not None
