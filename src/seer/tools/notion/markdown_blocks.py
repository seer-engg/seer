"""
Markdown-to-Notion block converter.

Converts markdown text into Notion API block objects using a two-step pipeline:
markdown → HTML (via the ``markdown`` library) → Notion blocks (via BeautifulSoup tree walk).

This keeps Notion tools free of parsing logic and can be reused by any tool that
needs to write formatted content to Notion.
"""
from typing import Any, Dict, List, Optional

import markdown as md_lib
from bs4 import BeautifulSoup, NavigableString, Tag

from seer.logger import get_logger

logger = get_logger("shared.tools.notion.markdown_blocks")

_NOTION_MAX_TEXT_LENGTH = 2000


def _chunk_text(text: str) -> List[str]:
    """Split text into segments that respect the Notion 2000-char limit."""
    if not text:
        return [""]
    return [text[i:i + _NOTION_MAX_TEXT_LENGTH] for i in range(0, len(text), _NOTION_MAX_TEXT_LENGTH)]


def _plain_rich_text(text: str) -> List[Dict[str, Any]]:
    """Build a plain (no-annotation) rich_text array, chunked to the API limit."""
    return [{"type": "text", "text": {"content": chunk}} for chunk in _chunk_text(text)]


def _make_block(block_type: str, rich_text: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a single Notion block dict with a rich_text body."""
    return {
        "object": "block",
        "type": block_type,
        block_type: {"rich_text": rich_text},
    }


# ─── Inline HTML → rich_text annotations ────────────────────────────────────

_ANNOTATION_MAP: Dict[str, str] = {
    "strong": "bold",
    "b": "bold",
    "em": "italic",
    "i": "italic",
    "code": "code",
    "del": "strikethrough",
    "s": "strikethrough",
}


def _walk_inline(
    node: Any,
    annotations: Dict[str, Any],
    link: Optional[str],
    result: List[Dict[str, Any]],
) -> None:
    """Recursively walk an inline HTML node and append rich_text entries to *result*."""
    if isinstance(node, NavigableString):
        text = str(node)
        if not text:
            return
        for chunk in _chunk_text(text):
            entry: Dict[str, Any] = {"type": "text", "text": {"content": chunk}}
            if link:
                entry["text"]["link"] = {"url": link}
            clean = {k: v for k, v in annotations.items() if v}
            if clean:
                entry["annotations"] = clean
            result.append(entry)
        return

    if not isinstance(node, Tag):
        return

    new_annotations = dict(annotations)
    new_link = link

    annotation_key = _ANNOTATION_MAP.get(node.name)
    if annotation_key:
        new_annotations[annotation_key] = True
    elif node.name == "a":
        new_link = node.get("href", "")

    for child in node.children:
        _walk_inline(child, new_annotations, new_link, result)


def _html_inline_to_rich_text(element: Tag) -> List[Dict[str, Any]]:
    """Convert inline HTML children to a Notion rich_text array with annotations."""
    result: List[Dict[str, Any]] = []
    for child in element.children:
        _walk_inline(child, {}, None, result)
    return result if result else [{"type": "text", "text": {"content": ""}}]


# ─── Block-level HTML → Notion blocks ───────────────────────────────────────

_HEADING_MAP: Dict[str, str] = {
    "h1": "heading_1",
    "h2": "heading_2",
    "h3": "heading_3",
    "h4": "heading_3",  # Notion only has 3 heading levels
    "h5": "heading_3",
    "h6": "heading_3",
}


def _convert_heading(el: Tag) -> List[Dict[str, Any]]:
    notion_type = _HEADING_MAP[el.name]
    return [_make_block(notion_type, _html_inline_to_rich_text(el))]


def _convert_paragraph(el: Tag) -> List[Dict[str, Any]]:
    return [_make_block("paragraph", _html_inline_to_rich_text(el))]


def _convert_unordered_list(el: Tag) -> List[Dict[str, Any]]:
    return [
        _make_block("bulleted_list_item", _html_inline_to_rich_text(li))
        for li in el.find_all("li", recursive=False)
    ]


def _convert_ordered_list(el: Tag) -> List[Dict[str, Any]]:
    return [
        _make_block("numbered_list_item", _html_inline_to_rich_text(li))
        for li in el.find_all("li", recursive=False)
    ]


def _extract_code_language(code_el: Optional[Tag]) -> str:
    """Extract language identifier from ``class="language-xxx"``."""
    if code_el and code_el.get("class"):
        for cls in code_el["class"]:
            if cls.startswith("language-"):
                return cls[len("language-"):]
    return "plain text"


def _convert_pre(el: Tag) -> List[Dict[str, Any]]:
    code_el = el.find("code")
    code_text = (code_el.get_text() if code_el else el.get_text()).rstrip("\n")
    language = _extract_code_language(code_el)
    return [{
        "object": "block",
        "type": "code",
        "code": {"rich_text": _plain_rich_text(code_text), "language": language},
    }]


def _convert_blockquote(el: Tag) -> List[Dict[str, Any]]:
    inner_text = el.get_text(strip=True)
    if not inner_text:
        return []
    return [_make_block("quote", _plain_rich_text(inner_text))]


def _convert_hr(_el: Tag) -> List[Dict[str, Any]]:
    return [{"object": "block", "type": "divider", "divider": {}}]


def _convert_fallback(el: Tag) -> List[Dict[str, Any]]:
    text = el.get_text(strip=True)
    if not text:
        return []
    return [_make_block("paragraph", [{"type": "text", "text": {"content": text}}])]


# Dispatch table: HTML tag name → converter function
_BLOCK_CONVERTERS = {
    "h1": _convert_heading,
    "h2": _convert_heading,
    "h3": _convert_heading,
    "h4": _convert_heading,
    "h5": _convert_heading,
    "h6": _convert_heading,
    "p": _convert_paragraph,
    "ul": _convert_unordered_list,
    "ol": _convert_ordered_list,
    "pre": _convert_pre,
    "blockquote": _convert_blockquote,
    "hr": _convert_hr,
}


def _build_paragraph_blocks(text: str) -> List[Dict[str, Any]]:
    """Build plain Notion paragraph blocks, chunking at the 2000-char API limit.

    This is the simple fallback used when markdown parsing is unnecessary or fails.
    """
    chunks = [text[i:i + _NOTION_MAX_TEXT_LENGTH] for i in range(0, max(len(text), 1), _NOTION_MAX_TEXT_LENGTH)]
    return [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": chunk}}]
            },
        }
        for chunk in chunks
    ]


def markdown_to_notion_blocks(text: str) -> List[Dict[str, Any]]:
    """Convert markdown text to Notion API block objects.

    Uses the ``markdown`` library to parse markdown into HTML, then walks the
    HTML tree with BeautifulSoup and dispatches each element to a type-specific
    converter via ``_BLOCK_CONVERTERS``.

    Falls back to plain paragraph blocks if parsing fails or the input is empty.
    """
    if not text or not text.strip():
        return _build_paragraph_blocks(text or "")

    try:
        html = md_lib.markdown(text, extensions=["extra", "nl2br"])
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: markdown parsing must never block tool execution
        logger.debug("Markdown parsing failed, falling back to plain paragraphs", exc_info=True)
        return _build_paragraph_blocks(text)

    soup = BeautifulSoup(html, "html.parser")
    blocks: List[Dict[str, Any]] = []

    for el in soup.children:
        if isinstance(el, NavigableString):
            stripped = str(el).strip()
            if stripped:
                blocks.append(_make_block("paragraph", [{"type": "text", "text": {"content": stripped}}]))
            continue

        if not isinstance(el, Tag):
            continue

        converter = _BLOCK_CONVERTERS.get(el.name, _convert_fallback)
        blocks.extend(converter(el))

    return blocks if blocks else _build_paragraph_blocks(text)
