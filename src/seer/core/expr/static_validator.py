"""
Static compile-time validation for condition expressions.

This module performs type checking on `${...}` expressions at compile time,
catching type errors before workflow execution. For example, comparing a string
to a number (`${name} > 100`) would be caught here rather than failing at runtime.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from seer.core.expr import parser
from seer.core.expr.parser import TemplateReference
from seer.core.expr.typecheck import (
    Scope,
    TypeCheckError,
    typecheck_reference,
)
from seer.core.schema.models import JsonSchema


# =============================================================================
# Type Information
# =============================================================================


@dataclass
class TypeInfo:
    """
    Represents inferred type information for a value or expression.

    This bridges JSON Schema types to our compile-time type system.
    """

    json_type: Union[str, List[str]]
    """The JSON type(s): "string", "number", "boolean", "array", "object", or list for unions."""

    nullable: bool = False
    """Whether the type includes null."""

    item_type: Optional["TypeInfo"] = None
    """For arrays, the type of items."""

    properties: Dict[str, "TypeInfo"] = field(default_factory=dict)
    """For objects, known property types."""

    additional_properties: bool = False
    """Whether the object allows additional properties."""

    def is_numeric(self) -> bool:
        """Check if this type is numeric (integer or number)."""
        types = self.normalize_types()
        return bool(types & {"number", "integer"})

    def is_string(self) -> bool:
        """Check if this type is a string."""
        types = self.normalize_types()
        return "string" in types

    def is_boolean(self) -> bool:
        """Check if this type is boolean."""
        types = self.normalize_types()
        return "boolean" in types

    def is_array(self) -> bool:
        """Check if this type is an array."""
        types = self.normalize_types()
        return "array" in types

    def is_object(self) -> bool:
        """Check if this type is an object."""
        types = self.normalize_types()
        return "object" in types

    def is_iterable(self) -> bool:
        """Check if this type can be iterated (array or string)."""
        return self.is_array() or self.is_string()

    def is_container(self) -> bool:
        """Check if this type supports `in` operator (array, string, or object)."""
        return self.is_array() or self.is_string() or self.is_object()

    def is_comparable(self) -> bool:
        """Check if this type supports ordering operators (<, >, <=, >=)."""
        return self.is_numeric() or self.is_string()

    def is_sliceable(self) -> bool:
        """Check if this type supports slice operations."""
        return self.is_array() or self.is_string()

    def normalize_types(self) -> Set[str]:
        """Get the set of non-null types."""
        if isinstance(self.json_type, str):
            types = {self.json_type}
        else:
            types = set(self.json_type)
        types.discard("null")
        return types

    def is_permissive_union(self) -> bool:
        """Check if a type is a permissive union (multiple non-null types)."""
        types = self.normalize_types()
        # If type includes both comparable and non-comparable types, it's permissive
        comparable_pairs = ({"string", "number"}, {"integer", "number"})
        return len(types) > 2 or (len(types) == 2 and types not in comparable_pairs)

    def is_definitely_numeric(self) -> bool:
        """Check if a type is definitely only numeric (no union with other types)."""
        types = self.normalize_types()
        return types.issubset({"number", "integer"}) and len(types) > 0

    def is_definitely_string(self) -> bool:
        """Check if a type is definitely only string (no union with other types)."""
        types = self.normalize_types()
        return types == {"string"}

    def type_description(self) -> str:
        """Get a human-readable description of this type."""
        types = self.normalize_types()
        if not types:
            return "null"
        if len(types) == 1:
            return next(iter(types))
        return "/".join(sorted(types))

    @classmethod
    def from_schema(cls, schema: JsonSchema) -> "TypeInfo":
        """Convert a JSON Schema to TypeInfo."""
        schema_type = schema.get("type")
        nullable = False

        # Handle type as list (union types)
        if isinstance(schema_type, list):
            nullable = "null" in schema_type
            types = [t for t in schema_type if t != "null"]
            json_type = types[0] if len(types) == 1 else types
        else:
            json_type = schema_type or "object"

        # Extract item type for arrays
        item_type = None
        if json_type == "array" or (isinstance(json_type, list) and "array" in json_type):
            items = schema.get("items")
            if isinstance(items, dict):
                item_type = cls.from_schema(items)

        # Extract properties for objects
        properties: Dict[str, TypeInfo] = {}
        additional_properties = False
        if json_type == "object" or (isinstance(json_type, list) and "object" in json_type):
            for key, prop_schema in schema.get("properties", {}).items():
                properties[key] = cls.from_schema(prop_schema)
            additional_properties = bool(schema.get("additionalProperties"))

        return cls(
            json_type=json_type,
            nullable=nullable,
            item_type=item_type,
            properties=properties,
            additional_properties=additional_properties,
        )

    @classmethod
    def from_python_value(cls, value: Any) -> "TypeInfo":
        """Infer TypeInfo from a Python value (used for literals in conditions)."""
        # pylint: disable=too-many-return-statements
        # Reason: Type inference requires checking each Python type separately
        if value is None:
            return cls(json_type="null", nullable=True)
        if isinstance(value, bool):
            return cls(json_type="boolean")
        if isinstance(value, int):
            return cls(json_type="integer")
        if isinstance(value, float):
            return cls(json_type="number")
        if isinstance(value, str):
            return cls(json_type="string")
        if isinstance(value, list):
            return cls(json_type="array")
        if isinstance(value, dict):
            return cls(json_type="object")
        return cls(json_type="object")  # Fallback


# =============================================================================
# Function Type Rules
# =============================================================================


@dataclass
class FunctionTypeRule:
    """Type rule for a safe function."""

    accepts: Set[str]
    """Set of accepted type categories: "iterable", "numeric", "any", "string"."""

    returns: str
    """Return type: "integer", "boolean", "string", "number"."""


FUNCTION_TYPE_RULES: Dict[str, FunctionTypeRule] = {
    "len": FunctionTypeRule(accepts={"iterable"}, returns="integer"),
    "any": FunctionTypeRule(accepts={"iterable"}, returns="boolean"),
    "all": FunctionTypeRule(accepts={"iterable"}, returns="boolean"),
    "sum": FunctionTypeRule(accepts={"iterable"}, returns="number"),
    "min": FunctionTypeRule(accepts={"iterable"}, returns="number"),
    "max": FunctionTypeRule(accepts={"iterable"}, returns="number"),
    "str": FunctionTypeRule(accepts={"any"}, returns="string"),
    "int": FunctionTypeRule(accepts={"numeric", "string"}, returns="integer"),
    "float": FunctionTypeRule(accepts={"numeric", "string"}, returns="number"),
}


# =============================================================================
# Common Typos and Hints
# =============================================================================


COMMON_TYPOS: Dict[str, str] = {
    "true": "True",
    "false": "False",
    "null": "None",
    "none": "None",
    "undefined": "None",
    "nil": "None",
}


# =============================================================================
# Static Condition Validator
# =============================================================================


# pylint: disable=invalid-name
# Reason: visit_* methods follow AST NodeVisitor naming convention
class StaticConditionValidator(ast.NodeVisitor):
    """
    AST visitor that performs static type checking on condition expressions.

    Validates type compatibility of operators and function arguments.
    """

    def __init__(self, placeholder_types: Dict[str, TypeInfo]) -> None:
        """
        Args:
            placeholder_types: Mapping from placeholder names to their types.
        """
        self.placeholder_types = placeholder_types
        self.errors: List[str] = []

    def _get_type(self, node: ast.expr) -> Optional[TypeInfo]:
        """Get the type of an AST expression node."""
        # pylint: disable=too-many-return-statements
        # Reason: Type resolution requires checking each AST node type separately
        if isinstance(node, ast.Name):
            return self._get_type_for_name(node)
        if isinstance(node, ast.Constant):
            return TypeInfo.from_python_value(node.value)
        if isinstance(node, ast.Call):
            return self._get_type_for_call(node)
        if isinstance(node, ast.Subscript):
            return self._get_type_for_subscript(node)
        if isinstance(node, ast.BinOp):
            return self._get_type_for_binop(node)
        if isinstance(node, ast.List):
            return TypeInfo(json_type="array")
        return None

    def _get_type_for_name(self, node: ast.Name) -> Optional[TypeInfo]:
        """Get type for a Name node (variable reference)."""
        if node.id in self.placeholder_types:
            return self.placeholder_types[node.id]
        if node.id in FUNCTION_TYPE_RULES:
            return None  # Functions are handled separately
        return None

    def _get_type_for_call(self, node: ast.Call) -> Optional[TypeInfo]:
        """Get type for a Call node (function call)."""
        if isinstance(node.func, ast.Name) and node.func.id in FUNCTION_TYPE_RULES:
            rule = FUNCTION_TYPE_RULES[node.func.id]
            return TypeInfo(json_type=rule.returns)
        return None

    def _get_type_for_subscript(self, node: ast.Subscript) -> Optional[TypeInfo]:
        """Get type for a Subscript node (array/object access)."""
        base_type = self._get_type(node.value)
        if base_type and base_type.is_array() and base_type.item_type:
            return base_type.item_type
        return None

    def _get_type_for_binop(self, node: ast.BinOp) -> Optional[TypeInfo]:
        """Get type for a BinOp node (binary operation)."""
        # Arithmetic operations return numeric
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            return TypeInfo(json_type="number")
        return None

    def visit_Name(self, node: ast.Name) -> None:
        """Check for undefined names and provide hints for common typos."""
        name = node.id

        # Check if it's a known placeholder or safe function
        if name in self.placeholder_types:
            return
        if name in FUNCTION_TYPE_RULES:
            return
        # Python built-in constants
        if name in ("True", "False", "None"):
            return

        # Check for common typos
        if name in COMMON_TYPOS:
            hint = COMMON_TYPOS[name]
            self.errors.append(f"Unknown variable '{name}'. Did you mean `{hint}`?")
            return

        self.errors.append(f"Unknown variable '{name}' in expression")

    def visit_Compare(self, node: ast.Compare) -> None:
        """Validate comparison operators."""
        # Get all operands for chained comparisons
        all_operands = [node.left] + list(node.comparators)

        for i, op in enumerate(node.ops):
            left = all_operands[i]
            right = all_operands[i + 1]

            left_type = self._get_type(left)
            right_type = self._get_type(right)

            # Handle `in` / `not in` operators
            if isinstance(op, (ast.In, ast.NotIn)):
                self._validate_in_operator(right_type)
                self.generic_visit(node)
                return

            # Handle ordering operators
            if isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)):
                self._validate_ordering_operator(left_type, right_type)
                self.generic_visit(node)
                return

        self.generic_visit(node)

    def _validate_in_operator(self, right_type: Optional[TypeInfo]) -> None:
        """Validate the 'in' operator's right operand."""
        # Skip check if right type is a permissive union (could be a container)
        if right_type and not right_type.is_container() and not right_type.is_permissive_union():
            self.errors.append(
                f"`in` requires container (array/string/object), got {right_type.type_description()}"
            )

    def _validate_ordering_operator(self, left_type: Optional[TypeInfo], right_type: Optional[TypeInfo]) -> None:
        """Validate ordering operators (<, >, <=, >=)."""
        # Skip strict checking for permissive union types (e.g., "any" types)
        # We can only catch definite type mismatches
        left_permissive = left_type and left_type.is_permissive_union()
        right_permissive = right_type and right_type.is_permissive_union()

        # If either side is a permissive union, skip strict validation
        if left_permissive or right_permissive:
            return

        # Both operands must be comparable
        if left_type and not left_type.is_comparable():
            self.errors.append(
                f"Ordering comparison not supported for type {left_type.type_description()}"
            )
        if right_type and not right_type.is_comparable():
            self.errors.append(
                f"Ordering comparison not supported for type {right_type.type_description()}"
            )

        # Check type compatibility - only flag when we're CERTAIN of incompatibility
        # (one is ONLY numeric, other is ONLY string)
        if left_type and right_type:
            if left_type.is_definitely_numeric() and right_type.is_definitely_string():
                self.errors.append(
                    f"Cannot compare {left_type.type_description()} with {right_type.type_description()} "
                    f"using ordering operators"
                )
            elif left_type.is_definitely_string() and right_type.is_definitely_numeric():
                self.errors.append(
                    f"Cannot compare {left_type.type_description()} with {right_type.type_description()} "
                    f"using ordering operators"
                )

    def visit_Call(self, node: ast.Call) -> None:
        """Validate function arguments against type rules."""
        if isinstance(node.func, ast.Attribute):
            # Method call like ${text}.lower()
            self.errors.append(
                f"Method calls like '.{node.func.attr}()' are not supported. "
                f"Use functions like `str(${{ref}})` instead"
            )
            return

        if not isinstance(node.func, ast.Name):
            self.errors.append("Only whitelisted helper functions can be used in conditions")
            return

        func_name = node.func.id
        if func_name not in FUNCTION_TYPE_RULES:
            self.errors.append(f"Function '{func_name}' is not allowed in conditions")
            return

        rule = FUNCTION_TYPE_RULES[func_name]

        # Check argument types
        if node.args:
            arg_type = self._get_type(node.args[0])
            if arg_type:
                if "any" in rule.accepts:
                    pass  # Accepts anything
                elif "iterable" in rule.accepts and not arg_type.is_iterable():
                    self.errors.append(
                        f"Function '{func_name}' requires iterable (array/string), "
                        f"got {arg_type.type_description()}"
                    )
                elif "numeric" in rule.accepts and not (arg_type.is_numeric() or arg_type.is_string()):
                    if "string" not in rule.accepts:
                        self.errors.append(
                            f"Function '{func_name}' requires numeric type, "
                            f"got {arg_type.type_description()}"
                        )

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Validate subscript/slice operations."""
        base_type = self._get_type(node.value)

        if isinstance(node.slice, ast.Slice):
            # Slice operations require sliceable types
            if base_type and not base_type.is_sliceable():
                self.errors.append(
                    f"Slice operation not supported for type {base_type.type_description()}"
                )
        else:
            # Numeric index requires array
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                if base_type and not base_type.is_array() and not base_type.is_string():
                    self.errors.append(
                        f"Numeric index requires array or string, got {base_type.type_description()}"
                    )

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Validate arithmetic operations."""
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
            left_type = self._get_type(node.left)
            right_type = self._get_type(node.right)

            # String concatenation is allowed for Add
            if isinstance(node.op, ast.Add):
                if left_type and left_type.is_string() and right_type and right_type.is_string():
                    self.generic_visit(node)
                    return

            # Otherwise both must be numeric
            if left_type and not left_type.is_numeric() and not left_type.is_string():
                self.errors.append(
                    f"Arithmetic operation requires numeric type, got {left_type.type_description()}"
                )
            if right_type and not right_type.is_numeric() and not right_type.is_string():
                self.errors.append(
                    f"Arithmetic operation requires numeric type, got {right_type.type_description()}"
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Catch method calls with helpful error."""
        self.errors.append(
            f"Method calls like '.{node.attr}()' are not supported. "
            f"Use functions like `str(${{ref}})` instead"
        )

    def visit_IfExp(self, node: ast.IfExp) -> None:  # pylint: disable=unused-argument
        """Catch ternary expressions with helpful error."""
        # Note: node parameter required by AST visitor interface but not used
        self.errors.append(
            "Ternary expressions (x if cond else y) are not supported. "
            "Use separate if/else nodes in the workflow instead"
        )

    def generic_visit(self, node: ast.AST) -> None:
        """Catch `is`/`is not` operators with suggestion."""
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, ast.Is):
                    self.errors.append("Use `== None` instead of `is None`")
                elif isinstance(op, ast.IsNot):
                    self.errors.append("Use `!= None` instead of `is not None`")

        super().generic_visit(node)


# =============================================================================
# Main Validation Function
# =============================================================================


def validate_condition_expression(expression: str, scope: Scope) -> List[str]:
    """
    Validate a condition expression for type errors.

    This performs compile-time static type checking on the expression,
    catching type mismatches before runtime.

    Args:
        expression: The condition expression string (e.g., "${count} > 10")
        scope: The type scope containing schemas for references

    Returns:
        List of error messages (empty if expression is valid)
    """
    errors: List[str] = []

    # Parse template to extract references
    try:
        tokens = parser.parse_template(expression)
    except ValueError as exc:
        errors.append(f"Invalid expression syntax: {exc}")
        return errors

    # Build placeholder types and rendered expression
    placeholder_types: Dict[str, TypeInfo] = {}
    rendered_parts: List[str] = []
    placeholder_counter = 0

    for token in tokens:
        if isinstance(token, TemplateReference):
            placeholder = f"__ref_{placeholder_counter}"
            placeholder_counter += 1

            # Try to resolve the reference type
            try:
                schema = typecheck_reference(token.reference, scope)
                placeholder_types[placeholder] = TypeInfo.from_schema(schema)
            except TypeCheckError:
                # Reference validation failed - skip type checking to avoid cascading errors
                # The reference error will be reported by _validate_value_references
                return []

            rendered_parts.append(placeholder)
        else:
            rendered_parts.append(token.text)

    rendered_expr = "".join(rendered_parts).strip()
    if not rendered_expr:
        return []  # Empty expression handled elsewhere

    # Parse as Python AST
    try:
        tree = ast.parse(rendered_expr, mode="eval")
    except SyntaxError as exc:
        errors.append(f"Invalid expression syntax: {exc.msg}")
        return errors

    # Run static type validator
    validator = StaticConditionValidator(placeholder_types)
    validator.visit(tree)

    return validator.errors
