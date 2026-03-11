"""
Content → binary converters for agent artifact generation.

Provides public functions:
- markdown_to_html: converts Markdown string to a styled HTML document
- html_to_pdf: converts HTML string to PDF bytes using weasyprint
- html_to_docx: converts HTML string to DOCX bytes using htmldocx + python-docx
"""

from __future__ import annotations

import io

import markdown as md

# MIME type mapping for supported artifact formats
FORMAT_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


_HTML_WRAPPER = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
body {{ font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
h1, h2, h3, h4 {{ margin-top: 1.4em; }}
pre {{ background: #f5f5f5; padding: 12px 16px; border-radius: 6px; overflow-x: auto; }}
code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
pre code {{ background: none; padding: 0; }}
blockquote {{ border-left: 4px solid #ddd; margin: 1em 0; padding: 0.5em 1em; color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #f5f5f5; font-weight: 600; }}
img {{ max-width: 100%; }}
</style></head>
<body>{body}</body>
</html>"""


def markdown_to_html(text: str) -> str:
    """Convert a Markdown string to a full styled HTML document."""
    body = md.markdown(text, extensions=["tables", "fenced_code", "toc", "nl2br"])
    return _HTML_WRAPPER.format(body=body)


def html_to_pdf(html: str) -> bytes:
    """
    Convert an HTML string to PDF bytes.

    Uses weasyprint to render the HTML as a styled document (CSS-based layout).
    This produces high-fidelity output that preserves HTML/CSS formatting.

    Args:
        html: Full HTML document string.

    Returns:
        PDF file as bytes.

    Raises:
        ImportError: If weasyprint is not installed.
        Exception: If HTML rendering fails.
    """
    # pylint: disable=import-outside-toplevel  # Reason: Optional dependency, lazy-loaded to avoid startup cost
    from weasyprint import HTML

    buffer = io.BytesIO()
    HTML(string=html).write_pdf(buffer)
    return buffer.getvalue()


def html_to_docx(html: str) -> bytes:
    """
    Convert an HTML string to DOCX bytes.

    Uses htmldocx to map HTML tags to python-docx paragraph styles.
    The output is a Word-native editable document.

    Args:
        html: Full HTML document string.

    Returns:
        DOCX file as bytes.

    Raises:
        ImportError: If python-docx or htmldocx is not installed.
        Exception: If HTML parsing or DOCX generation fails.
    """
    # pylint: disable=import-outside-toplevel  # Reason: Optional dependency, lazy-loaded to avoid startup cost
    from docx import Document
    from htmldocx import HtmlToDocx

    doc = Document()
    parser = HtmlToDocx()
    parser.add_html_to_document(html, doc)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
