"""
Test type coercion in parameter completion.

Validates that parameter values are correctly coerced to match their schema types,
preventing errors like boolean True being passed to APIs expecting string "true".
"""
import pytest
from shared.parameter_population.completion import coerce_to_schema_type


class TestTypeCoercion:
    """Test suite for schema-based type coercion."""
    
    def test_boolean_to_string_true(self):
        """Boolean True should coerce to string "true"."""
        schema = {"type": "string", "description": "Enable recursive mode"}
        result = coerce_to_schema_type(True, schema)
        assert result == "true"
        assert isinstance(result, str)
    
    def test_boolean_to_string_false(self):
        """Boolean False should coerce to string "false"."""
        schema = {"type": "string"}
        result = coerce_to_schema_type(False, schema)
        assert result == "false"
        assert isinstance(result, str)
    
    def test_number_to_string(self):
        """Numbers should coerce to strings."""
        schema = {"type": "string"}
        assert coerce_to_schema_type(123, schema) == "123"
        assert coerce_to_schema_type(45.67, schema) == "45.67"
    
    def test_string_to_boolean(self):
        """String representations should coerce to boolean."""
        schema = {"type": "boolean"}
        assert coerce_to_schema_type("true", schema) is True
        assert coerce_to_schema_type("True", schema) is True
        assert coerce_to_schema_type("1", schema) is True
        assert coerce_to_schema_type("yes", schema) is True
        assert coerce_to_schema_type("false", schema) is False
        assert coerce_to_schema_type("no", schema) is False
    
    def test_string_to_integer(self):
        """String numbers should coerce to integers."""
        schema = {"type": "integer"}
        result = coerce_to_schema_type("123", schema)
        assert result == 123
        assert isinstance(result, int)
    
    def test_string_to_float(self):
        """String numbers should coerce to floats."""
        schema = {"type": "number"}
        result = coerce_to_schema_type("45.67", schema)
        assert result == 45.67
        assert isinstance(result, float)
    
    def test_no_coercion_when_type_matches(self):
        """Values matching their schema type should pass through unchanged."""
        assert coerce_to_schema_type("hello", {"type": "string"}) == "hello"
        assert coerce_to_schema_type(True, {"type": "boolean"}) is True
        assert coerce_to_schema_type(123, {"type": "integer"}) == 123
        assert coerce_to_schema_type(45.67, {"type": "number"}) == 45.67
    
    def test_none_values_preserved(self):
        """None values should be preserved."""
        schema = {"type": "string"}
        assert coerce_to_schema_type(None, schema) is None
    
    def test_no_schema_returns_original(self):
        """Values with no schema should pass through unchanged."""
        assert coerce_to_schema_type(True, None) is True
        assert coerce_to_schema_type(True, {}) is True
    
    def test_union_types(self):
        """Union types (array of types) should use first non-null type."""
        schema = {"type": ["string", "null"]}
        result = coerce_to_schema_type(True, schema)
        assert result == "true"
        assert isinstance(result, str)
    
    def test_invalid_coercion_returns_original(self):
        """Invalid coercions should return original value with warning."""
        schema = {"type": "integer"}
        result = coerce_to_schema_type("not_a_number", schema)
        # Should return original when coercion fails
        assert result == "not_a_number"
    
    def test_recursive_parameter_case(self):
        """Test the specific bug case: recursive parameter."""
        # This is the actual bug case from the logs
        schema = {
            "type": "string",
            "description": "Set to '1' or 'true' to fetch the ENTIRE repository file tree recursively"
        }
        result = coerce_to_schema_type(True, schema)
        assert result == "true"
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

