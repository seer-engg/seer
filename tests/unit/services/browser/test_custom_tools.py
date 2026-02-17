"""Tests for CustomBrowserTools - structured output via submit_result action."""
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from seer.services.browser.custom_tools import (
    CustomBrowserTools,
    _json_type_to_python,
    create_submit_result_model,
)


@pytest.mark.unit
class TestJsonTypeToPython:
    """Test _json_type_to_python helper function."""

    def test_string_type(self):
        assert _json_type_to_python({"type": "string"}) == str

    def test_number_type(self):
        assert _json_type_to_python({"type": "number"}) == float

    def test_integer_type(self):
        assert _json_type_to_python({"type": "integer"}) == int

    def test_boolean_type(self):
        assert _json_type_to_python({"type": "boolean"}) == bool

    def test_object_type(self):
        result = _json_type_to_python({"type": "object"})
        assert result == Dict[str, Any]

    def test_array_of_strings(self):
        schema = {"type": "array", "items": {"type": "string"}}
        result = _json_type_to_python(schema)
        assert result == List[str]

    def test_array_of_numbers(self):
        schema = {"type": "array", "items": {"type": "number"}}
        result = _json_type_to_python(schema)
        assert result == List[float]

    def test_unknown_type_defaults_to_str(self):
        result = _json_type_to_python({"type": "unknown"})
        assert result == str

    def test_missing_type_defaults_to_str(self):
        result = _json_type_to_python({})
        assert result == str


@pytest.mark.unit
class TestCreateSubmitResultModel:
    """Test create_submit_result_model function."""

    def test_creates_valid_pydantic_model(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["name"],
        }
        model = create_submit_result_model(schema)

        assert issubclass(model, BaseModel)
        assert "name" in model.model_fields
        assert "price" in model.model_fields

    def test_required_fields_are_required(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        }
        model = create_submit_result_model(schema)

        assert model.model_fields["name"].is_required()
        assert model.model_fields["email"].is_required()

    def test_optional_fields_have_default(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
        }
        model = create_submit_result_model(schema)

        assert model.model_fields["name"].is_required()
        assert not model.model_fields["nickname"].is_required()

    def test_can_instantiate_with_valid_data(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["name"],
        }
        model = create_submit_result_model(schema)

        instance = model(name="Widget", price=29.99)
        assert instance.name == "Widget"
        assert instance.price == 29.99

    def test_missing_required_field_raises_validation_error(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["name", "price"],
        }
        model = create_submit_result_model(schema)

        with pytest.raises(ValidationError):
            model(name="Widget")  # Missing required 'price'

    def test_array_field_type(self):
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        model = create_submit_result_model(schema)

        instance = model(tags=["tag1", "tag2"])
        assert instance.tags == ["tag1", "tag2"]

    def test_non_object_schema_wraps_in_data(self):
        schema = {"type": "string"}
        model = create_submit_result_model(schema)

        assert "data" in model.model_fields
        assert model.model_fields["data"].is_required()

    def test_custom_model_name(self):
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        model = create_submit_result_model(schema, model_name="CustomParams")

        assert model.__name__ == "CustomParams"

    def test_empty_properties(self):
        schema = {"type": "object", "properties": {}}
        model = create_submit_result_model(schema)

        assert issubclass(model, BaseModel)
        assert len(model.model_fields) == 0


@pytest.mark.unit
class TestCustomBrowserToolsInit:
    """Test CustomBrowserTools initialization."""

    def test_init_without_schema(self):
        """Tools should work without extraction schema (no submit_result action)."""
        tools = CustomBrowserTools()

        assert tools._extraction_schema is None
        assert tools._extracted_data == {}
        # submit_result should NOT be registered
        assert "submit_result" not in tools.registry.registry.actions

    def test_init_with_schema_registers_submit_result(self):
        """When schema is provided, submit_result action should be registered."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        tools = CustomBrowserTools(extraction_schema=schema)

        assert tools._extraction_schema == schema
        assert "submit_result" in tools.registry.registry.actions

    def test_submit_result_action_has_correct_param_model(self):
        """The submit_result action should have params matching the schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["name"],
        }
        tools = CustomBrowserTools(extraction_schema=schema)

        action = tools.registry.registry.actions["submit_result"]
        param_model = action.param_model

        # Check that param_model has the expected fields
        assert "name" in param_model.model_fields
        assert "price" in param_model.model_fields

    def test_exclude_actions_passed_to_parent(self):
        """Exclude actions should be passed to the parent Tools class."""
        tools = CustomBrowserTools(exclude_actions=["navigate"])

        # The excluded action should not be registered
        assert "navigate" not in tools.registry.registry.actions


@pytest.mark.asyncio
@pytest.mark.unit
class TestSubmitResultAction:
    """Test the submit_result action execution."""

    async def test_submit_result_stores_data(self):
        """Calling submit_result should store the extracted data."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["name"],
        }
        tools = CustomBrowserTools(extraction_schema=schema)

        # Get the action function
        action = tools.registry.registry.actions["submit_result"]
        param_model = action.param_model

        # Create params and execute - browser_use requires keyword arguments
        params = param_model(name="Widget", price=29.99)
        await action.function(params=params)

        # Verify data was stored
        assert tools.get_extracted_data() == {"name": "Widget", "price": 29.99}

    async def test_submit_result_returns_action_result(self):
        """submit_result should return an ActionResult with is_done=True."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        tools = CustomBrowserTools(extraction_schema=schema)

        action = tools.registry.registry.actions["submit_result"]
        param_model = action.param_model

        params = param_model(name="Test")
        result = await action.function(params=params)

        assert result.is_done is True
        assert result.success is True
        assert result.extracted_content is not None
        # Verify the extracted content is valid JSON
        extracted = json.loads(result.extracted_content)
        assert extracted == {"name": "Test"}

    async def test_submit_result_with_optional_fields(self):
        """submit_result should handle optional fields correctly."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["name"],
        }
        tools = CustomBrowserTools(extraction_schema=schema)

        action = tools.registry.registry.actions["submit_result"]
        param_model = action.param_model

        # Only provide required field - browser_use requires keyword arguments
        params = param_model(name="Widget")
        await action.function(params=params)

        assert tools.get_extracted_data()["name"] == "Widget"
        assert tools.get_extracted_data().get("optional_field") is None


@pytest.mark.unit
class TestGetExtractedData:
    """Test get_extracted_data and has_extracted_data methods."""

    def test_get_extracted_data_empty_initially(self):
        """get_extracted_data returns empty dict before submit_result is called."""
        tools = CustomBrowserTools(extraction_schema={"type": "object", "properties": {}})

        assert tools.get_extracted_data() == {}

    def test_has_extracted_data_false_initially(self):
        """has_extracted_data returns False before submit_result is called."""
        tools = CustomBrowserTools(extraction_schema={"type": "object", "properties": {}})

        assert tools.has_extracted_data() is False

    @pytest.mark.asyncio
    async def test_has_extracted_data_true_after_submit(self):
        """has_extracted_data returns True after submit_result is called."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        tools = CustomBrowserTools(extraction_schema=schema)

        # Execute submit_result - browser_use requires keyword arguments
        action = tools.registry.registry.actions["submit_result"]
        param_model = action.param_model
        params = param_model(name="Test")
        await action.function(params=params)

        assert tools.has_extracted_data() is True


@pytest.mark.unit
class TestSubmitResultActionDescription:
    """Test that submit_result has helpful description with field names."""

    def test_action_description_includes_field_names(self):
        """The action description should mention the expected fields."""
        schema = {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "product_price": {"type": "number"},
            },
        }
        tools = CustomBrowserTools(extraction_schema=schema)

        action = tools.registry.registry.actions["submit_result"]
        description = action.description

        # The description should mention the fields
        assert "product_name" in description
        assert "product_price" in description


@pytest.mark.unit
class TestRealWorldSchemas:
    """Test with schemas that match real-world extraction use cases."""

    def test_shop_extraction_schema(self):
        """Test with a typical shop/store extraction schema."""
        schema = {
            "type": "object",
            "properties": {
                "shops": {
                    "type": "array",
                    "items": {
                        "type": "object",
                    },
                },
                "total_count": {"type": "integer"},
            },
            "required": ["shops"],
        }
        tools = CustomBrowserTools(extraction_schema=schema)

        assert "submit_result" in tools.registry.registry.actions
        param_model = tools.registry.registry.actions["submit_result"].param_model

        # Verify we can create valid params
        instance = param_model(shops=[{"name": "Store 1"}], total_count=1)
        assert instance.shops == [{"name": "Store 1"}]
        assert instance.total_count == 1

    def test_product_details_schema(self):
        """Test with a typical product details extraction schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "price": {"type": "number"},
                "in_stock": {"type": "boolean"},
                "features": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "price"],
        }
        tools = CustomBrowserTools(extraction_schema=schema)
        param_model = tools.registry.registry.actions["submit_result"].param_model

        instance = param_model(
            name="Widget Pro",
            price=99.99,
            in_stock=True,
            features=["Feature A", "Feature B"],
        )
        assert instance.name == "Widget Pro"
        assert instance.price == 99.99
        assert instance.in_stock is True
        assert instance.features == ["Feature A", "Feature B"]
