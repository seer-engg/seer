"""
Parsing utilities for `${...}` references embedded in strings as well as
recursive discovery of references inside arbitrary JSON values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

from seer.core.schema.models import JSONValue

REFERENCE_PATTERN = re.compile(r"\$\{([^{}]+)\}")


@dataclass(frozen=True)
class PathSegment:
    pass


@dataclass(frozen=True)
class PropertySegment(PathSegment):
    key: str


@dataclass(frozen=True)
class IndexSegment(PathSegment):
    index: int | str


@dataclass(frozen=True)
class ReferenceExpr:
    raw: str
    root: str
    segments: Sequence[PathSegment]


@dataclass(frozen=True)
class TemplateLiteral:
    text: str


@dataclass(frozen=True)
class TemplateReference:
    placeholder: str
    reference: ReferenceExpr


TemplateToken = TemplateLiteral | TemplateReference


def _parse_bracket_accessor(working: str, idx: int, expr: str) -> tuple[PathSegment, int]:
    """Parse bracket accessor and return segment and next index."""
    closing = working.find("]", idx + 1)
    if closing == -1:
        raise ValueError(f"Unclosed '[' in reference '{expr}'")

    token = working[idx + 1: closing].strip()
    if not token:
        raise ValueError(f"Empty bracket accessor in reference '{expr}'")

    if token[0] in {"'", '"'} and token[-1] == token[0]:
        return IndexSegment(token[1:-1]), closing + 1

    try:
        return IndexSegment(int(token)), closing + 1
    except ValueError as exc:
        raise ValueError(
            f"Bracket accessor must be an integer or quoted string in '{expr}'"
        ) from exc


def _flush_property_buffer(buffer: List[str], root: str | None, segments: List[PathSegment]) -> str | None:
    """Flush property buffer and update root/segments."""
    if not buffer:
        return root

    token = "".join(buffer).strip()
    buffer.clear()

    if not token:
        return root

    if root is None:
        return token

    segments.append(PropertySegment(token))
    return root


def parse_reference_string(expr: str) -> ReferenceExpr:
    working = expr.strip()
    if not working:
        raise ValueError("Reference cannot be empty")

    root: str | None = None
    segments: List[PathSegment] = []
    buffer: List[str] = []
    idx = 0

    while idx < len(working):
        char = working[idx]

        if char == ".":
            root = _flush_property_buffer(buffer, root, segments)
            idx += 1
        elif char == "[":
            root = _flush_property_buffer(buffer, root, segments)
            segment, next_idx = _parse_bracket_accessor(working, idx, expr)
            segments.append(segment)
            idx = next_idx
        else:
            buffer.append(char)
            idx += 1

    root = _flush_property_buffer(buffer, root, segments)

    if root is None:
        raise ValueError(f"Reference '{expr}' is missing a root symbol")

    return ReferenceExpr(raw=expr, root=root, segments=tuple(segments))


def parse_template(text: str) -> List[TemplateToken]:
    tokens: List[TemplateToken] = []
    cursor = 0
    for match in REFERENCE_PATTERN.finditer(text):
        start, end = match.span()
        if start > cursor:
            tokens.append(TemplateLiteral(text[cursor:start]))
        placeholder = match.group(0)
        expr = parse_reference_string(match.group(1))
        tokens.append(TemplateReference(placeholder=placeholder, reference=expr))
        cursor = end
    if cursor < len(text):
        tokens.append(TemplateLiteral(text[cursor:]))
    if not tokens:
        tokens.append(TemplateLiteral(text))
    return tokens


def iterate_value_references(value: JSONValue) -> Iterator[ReferenceExpr]:
    if isinstance(value, str):
        for token in parse_template(value):
            if isinstance(token, TemplateReference):
                yield token.reference
        return
    if isinstance(value, list):
        for item in value:
            yield from iterate_value_references(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iterate_value_references(item)


def collect_unique_references(values: Iterable[JSONValue]) -> List[ReferenceExpr]:
    """
    Collect references from a set of values while preserving discovery order and
    avoiding duplicates.
    """

    seen: set[str] = set()
    ordered: List[ReferenceExpr] = []
    for value in values:
        for ref in iterate_value_references(value):
            if ref.raw in seen:
                continue
            seen.add(ref.raw)
            ordered.append(ref)
    return ordered
