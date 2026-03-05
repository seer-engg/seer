"""
HTML → binary converters for agent artifact generation.

Provides two public functions:
- html_to_pdf: converts HTML string to PDF bytes using weasyprint
- html_to_docx: converts HTML string to DOCX bytes using htmldocx + python-docx
"""

from __future__ import annotations

import io

# MIME type mapping for supported artifact formats
FORMAT_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


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
