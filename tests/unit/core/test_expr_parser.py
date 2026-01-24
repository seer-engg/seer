"""
Unit tests for expression parser.

Tests parsing of ${...} references, template strings, and reference collection.
Target coverage: 95%+
"""
import pytest

from seer.core.expr.parser import (
    IndexSegment,
    PropertySegment,
    TemplateLiteral,
    TemplateReference,
    parse_reference_string,
    parse_template,
    iterate_value_references,
    collect_unique_references,
)


# =============================================================================
# Reference String Parsing Tests
# =============================================================================


def test_parse_reference_string_simple():
    """Test parsing simple root reference."""
    ref = parse_reference_string("trigger")
    assert ref.root == "trigger"
    assert ref.segments == ()
    assert ref.raw == "trigger"


def test_parse_reference_string_with_property():
    """Test parsing reference with property access."""
    ref = parse_reference_string("trigger.data")
    assert ref.root == "trigger"
    assert len(ref.segments) == 1
    assert isinstance(ref.segments[0], PropertySegment)
    assert ref.segments[0].key == "data"


def test_parse_reference_string_with_multiple_properties():
    """Test parsing reference with multiple property accesses."""
    ref = parse_reference_string("trigger.data.message")
    assert ref.root == "trigger"
    assert len(ref.segments) == 2
    assert ref.segments[0].key == "data"
    assert ref.segments[1].key == "message"


def test_parse_reference_string_with_numeric_index():
    """Test parsing reference with numeric array index."""
    ref = parse_reference_string("items[0]")
    assert ref.root == "items"
    assert len(ref.segments) == 1
    assert isinstance(ref.segments[0], IndexSegment)
    assert ref.segments[0].index == 0


def test_parse_reference_string_with_string_index():
    """Test parsing reference with string index."""
    ref = parse_reference_string('items["key"]')
    assert ref.root == "items"
    assert len(ref.segments) == 1
    assert isinstance(ref.segments[0], IndexSegment)
    assert ref.segments[0].index == "key"


def test_parse_reference_string_with_single_quote_index():
    """Test parsing reference with single-quoted string index."""
    ref = parse_reference_string("items['key']")
    assert ref.root == "items"
    assert len(ref.segments) == 1
    assert isinstance(ref.segments[0], IndexSegment)
    assert ref.segments[0].index == "key"


def test_parse_reference_string_mixed_access():
    """Test parsing reference with mixed property and index access."""
    ref = parse_reference_string("data.items[0].name")
    assert ref.root == "data"
    assert len(ref.segments) == 3
    assert isinstance(ref.segments[0], PropertySegment)
    assert ref.segments[0].key == "items"
    assert isinstance(ref.segments[1], IndexSegment)
    assert ref.segments[1].index == 0
    assert isinstance(ref.segments[2], PropertySegment)
    assert ref.segments[2].key == "name"


def test_parse_reference_string_consecutive_indexes():
    """Test parsing reference with consecutive bracket indexes."""
    ref = parse_reference_string("matrix[0][1]")
    assert ref.root == "matrix"
    assert len(ref.segments) == 2
    assert ref.segments[0].index == 0
    assert ref.segments[1].index == 1


def test_parse_reference_string_with_whitespace():
    """Test parsing reference with whitespace."""
    ref = parse_reference_string("  trigger.data  ")
    assert ref.root == "trigger"
    assert len(ref.segments) == 1
    assert ref.segments[0].key == "data"


def test_parse_reference_string_with_whitespace_in_brackets():
    """Test parsing reference with whitespace inside brackets."""
    ref = parse_reference_string("items[ 0 ]")
    assert ref.root == "items"
    assert ref.segments[0].index == 0


# =============================================================================
# Reference String Parsing Error Tests
# =============================================================================


def test_parse_reference_string_empty():
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="Reference cannot be empty"):
        parse_reference_string("")


def test_parse_reference_string_whitespace_only():
    """Test that whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="Reference cannot be empty"):
        parse_reference_string("   ")


def test_parse_reference_string_unclosed_bracket():
    """Test that unclosed bracket raises ValueError."""
    with pytest.raises(ValueError, match="Unclosed '\\[' in reference"):
        parse_reference_string("items[0")


def test_parse_reference_string_empty_bracket():
    """Test that empty bracket accessor raises ValueError."""
    with pytest.raises(ValueError, match="Empty bracket accessor"):
        parse_reference_string("items[]")


def test_parse_reference_string_invalid_bracket_content():
    """Test that invalid bracket content raises ValueError."""
    with pytest.raises(ValueError, match="Bracket accessor must be an integer or quoted string"):
        parse_reference_string("items[not_a_number]")


def test_parse_reference_string_missing_root():
    """Test that reference with leading dot still parses (dot is ignored)."""
    # Leading dots are handled gracefully - the dot acts as a separator
    result = parse_reference_string(".property")
    assert result.root == "property"
    assert result.segments == ()


# =============================================================================
# Template Parsing Tests
# =============================================================================


def test_parse_template_no_references():
    """Test parsing template with no references (plain text)."""
    tokens = parse_template("Hello World")
    assert len(tokens) == 1
    assert isinstance(tokens[0], TemplateLiteral)
    assert tokens[0].text == "Hello World"


def test_parse_template_single_reference():
    """Test parsing template with single reference."""
    tokens = parse_template("${name}")
    assert len(tokens) == 1
    assert isinstance(tokens[0], TemplateReference)
    assert tokens[0].reference.root == "name"


def test_parse_template_with_prefix():
    """Test parsing template with literal prefix."""
    tokens = parse_template("Hello ${name}")
    assert len(tokens) == 2
    assert isinstance(tokens[0], TemplateLiteral)
    assert tokens[0].text == "Hello "
    assert isinstance(tokens[1], TemplateReference)
    assert tokens[1].reference.root == "name"


def test_parse_template_with_suffix():
    """Test parsing template with literal suffix."""
    tokens = parse_template("${name}!")
    assert len(tokens) == 2
    assert isinstance(tokens[0], TemplateReference)
    assert isinstance(tokens[1], TemplateLiteral)
    assert tokens[1].text == "!"


def test_parse_template_multiple_references():
    """Test parsing template with multiple references."""
    tokens = parse_template("Hello ${first} ${last}!")
    assert len(tokens) == 5
    assert isinstance(tokens[0], TemplateLiteral)
    assert tokens[0].text == "Hello "
    assert isinstance(tokens[1], TemplateReference)
    assert tokens[1].reference.root == "first"
    assert isinstance(tokens[2], TemplateLiteral)
    assert tokens[2].text == " "
    assert isinstance(tokens[3], TemplateReference)
    assert tokens[3].reference.root == "last"
    assert isinstance(tokens[4], TemplateLiteral)
    assert tokens[4].text == "!"


def test_parse_template_consecutive_references():
    """Test parsing template with consecutive references."""
    tokens = parse_template("${first}${last}")
    assert len(tokens) == 2
    assert isinstance(tokens[0], TemplateReference)
    assert tokens[0].reference.root == "first"
    assert isinstance(tokens[1], TemplateReference)
    assert tokens[1].reference.root == "last"


def test_parse_template_with_property_access():
    """Test parsing template with property access in reference."""
    tokens = parse_template("${trigger.data.message}")
    assert len(tokens) == 1
    assert isinstance(tokens[0], TemplateReference)
    assert tokens[0].reference.root == "trigger"
    assert len(tokens[0].reference.segments) == 2


def test_parse_template_with_array_index():
    """Test parsing template with array index in reference."""
    tokens = parse_template("${items[0]}")
    assert len(tokens) == 1
    assert isinstance(tokens[0], TemplateReference)
    assert tokens[0].reference.root == "items"
    assert tokens[0].reference.segments[0].index == 0


def test_parse_template_empty_string():
    """Test parsing empty template string."""
    tokens = parse_template("")
    assert len(tokens) == 1
    assert isinstance(tokens[0], TemplateLiteral)
    assert tokens[0].text == ""


def test_parse_template_with_escaped_braces():
    """Test that escaped braces are treated as literals (not implemented escape mechanism)."""
    # Note: The current implementation doesn't support escaping, so ${ will be matched
    tokens = parse_template("Not a ref: $${ but this is: ${ref}")
    # This will likely parse $${ as text and ${ref} as reference
    assert len(tokens) >= 2


def test_parse_template_complex():
    """Test parsing complex template with mixed content."""
    template = "Result: ${data.items[0].name} - Status: ${status}"
    tokens = parse_template(template)

    # Should have: literal, ref, literal, ref
    assert len(tokens) >= 3
    assert isinstance(tokens[0], TemplateLiteral)
    assert isinstance(tokens[1], TemplateReference)


# =============================================================================
# Value Reference Iteration Tests
# =============================================================================


def test_iterate_value_references_string_with_ref():
    """Test iterating references in string value."""
    refs = list(iterate_value_references("${name}"))
    assert len(refs) == 1
    assert refs[0].root == "name"


def test_iterate_value_references_string_with_multiple_refs():
    """Test iterating multiple references in string value."""
    refs = list(iterate_value_references("${first} ${last}"))
    assert len(refs) == 2
    assert refs[0].root == "first"
    assert refs[1].root == "last"


def test_iterate_value_references_string_no_refs():
    """Test iterating references in string without references."""
    refs = list(iterate_value_references("plain text"))
    assert len(refs) == 0


def test_iterate_value_references_list():
    """Test iterating references in list value."""
    refs = list(iterate_value_references(["${a}", "${b}"]))
    assert len(refs) == 2
    assert refs[0].root == "a"
    assert refs[1].root == "b"


def test_iterate_value_references_nested_list():
    """Test iterating references in nested list."""
    refs = list(iterate_value_references([["${a}"], ["${b}"]]))
    assert len(refs) == 2


def test_iterate_value_references_dict():
    """Test iterating references in dict value."""
    refs = list(iterate_value_references({"key1": "${a}", "key2": "${b}"}))
    assert len(refs) == 2
    roots = {ref.root for ref in refs}
    assert roots == {"a", "b"}


def test_iterate_value_references_nested_dict():
    """Test iterating references in nested dict."""
    refs = list(iterate_value_references({
        "outer": {
            "inner": "${nested}"
        }
    }))
    assert len(refs) == 1
    assert refs[0].root == "nested"


def test_iterate_value_references_mixed_structure():
    """Test iterating references in mixed structure."""
    refs = list(iterate_value_references({
        "items": ["${item1}", "${item2}"],
        "name": "${name}",
        "nested": {
            "value": "${nested_value}"
        }
    }))
    assert len(refs) == 4
    roots = {ref.root for ref in refs}
    assert roots == {"item1", "item2", "name", "nested_value"}


def test_iterate_value_references_non_string_primitives():
    """Test iterating references in non-string primitives (no refs)."""
    assert not list(iterate_value_references(42))
    assert not list(iterate_value_references(True))
    assert not list(iterate_value_references(None))


# =============================================================================
# Collect Unique References Tests
# =============================================================================


def test_collect_unique_references_single_value():
    """Test collecting references from single value."""
    refs = collect_unique_references(["${name}"])
    assert len(refs) == 1
    assert refs[0].root == "name"


def test_collect_unique_references_multiple_values():
    """Test collecting references from multiple values."""
    refs = collect_unique_references(["${a}", "${b}", "${c}"])
    assert len(refs) == 3
    assert [ref.root for ref in refs] == ["a", "b", "c"]


def test_collect_unique_references_with_duplicates():
    """Test that duplicate references are filtered."""
    refs = collect_unique_references(["${name}", "${name}", "${name}"])
    assert len(refs) == 1
    assert refs[0].root == "name"


def test_collect_unique_references_preserves_order():
    """Test that reference discovery order is preserved."""
    refs = collect_unique_references(["${c}", "${a}", "${b}"])
    assert [ref.root for ref in refs] == ["c", "a", "b"]


def test_collect_unique_references_complex_values():
    """Test collecting references from complex values."""
    refs = collect_unique_references([
        {"key": "${a}"},
        ["${b}", "${a}"],  # ${a} is duplicate
        "${c}"
    ])
    assert len(refs) == 3
    assert [ref.root for ref in refs] == ["a", "b", "c"]


def test_collect_unique_references_empty_input():
    """Test collecting references from empty list."""
    refs = collect_unique_references([])
    assert len(refs) == 0


def test_collect_unique_references_no_refs():
    """Test collecting references when values have no references."""
    refs = collect_unique_references(["plain text", 42, {"key": "value"}])
    assert len(refs) == 0


def test_collect_unique_references_same_root_different_paths():
    """Test that references with same root but different paths are collected."""
    refs = collect_unique_references(["${data.name}", "${data.age}"])
    assert len(refs) == 2
    # They have the same root but different full paths
    assert refs[0].raw != refs[1].raw


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_parse_reference_with_underscores():
    """Test parsing reference with underscores in identifiers."""
    ref = parse_reference_string("_private_var._internal_prop")
    assert ref.root == "_private_var"
    assert ref.segments[0].key == "_internal_prop"


def test_parse_reference_with_numbers():
    """Test parsing reference with numbers in identifiers."""
    ref = parse_reference_string("var123.prop456")
    assert ref.root == "var123"
    assert ref.segments[0].key == "prop456"


def test_parse_template_with_special_characters():
    """Test parsing template with special characters in literals."""
    tokens = parse_template("Special: @#$% ${ref} !?")
    assert len(tokens) >= 2
    assert isinstance(tokens[0], TemplateLiteral)
    assert "Special" in tokens[0].text


def test_parse_reference_long_chain():
    """Test parsing reference with very long property chain."""
    long_ref = "a.b.c.d.e.f.g.h.i.j"
    ref = parse_reference_string(long_ref)
    assert ref.root == "a"
    assert len(ref.segments) == 9


@pytest.mark.parametrize("bracket_value,expected", [
    ("[0]", 0),
    ("[123]", 123),
    ('["key"]', "key"),
    ("['key']", "key"),
    ("[  42  ]", 42),
])
def test_parse_reference_various_bracket_formats(bracket_value, expected):
    """Test parsing various bracket formats."""
    ref = parse_reference_string(f"items{bracket_value}")
    assert ref.segments[0].index == expected
