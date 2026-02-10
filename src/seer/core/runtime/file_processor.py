"""
Process file content for LLM consumption.

This module provides the FileContentProcessor class that transforms file data
into formats suitable for LLM input:
- Images are converted to base64 data URLs for vision models
- Documents (PDF, DOCX, TXT) have text extracted for inclusion in prompts
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Set

from seer.logger import get_logger

logger = get_logger("seer.core.runtime.file_processor")

# Supported image MIME types - these are sent as base64 content blocks
IMAGE_MIME_TYPES: Set[str] = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/webp",
}

# Document MIME types - text is extracted from these
DOCUMENT_MIME_TYPES: Set[str] = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


class FileContentProcessor:
    """
    Processes files for LLM input.

    Handles two categories of files:
    1. Images: Converted to base64 data URLs in the format expected by
       LangChain/OpenAI vision models
    2. Documents: Text extracted via DocumentProcessor and formatted for
       inclusion in prompts

    Usage:
        processor = FileContentProcessor()
        result = await processor.process_files([
            {"key": "image", "mime_type": "image/png", "filename": "photo.png", "content": b"..."},
            {"key": "doc", "mime_type": "application/pdf", "filename": "report.pdf", "content": b"..."},
        ])
        # result = {"image_blocks": [...], "extracted_text": "..."}
    """

    def __init__(self) -> None:
        """Initialize the file processor with lazy-loaded document processor."""
        self._doc_processor = None

    def _get_document_processor(self):
        """Get or create document processor instance (lazy loading)."""
        if self._doc_processor is None:
            # pylint: disable=import-outside-toplevel  # Avoid circular imports
            from seer.services.knowledge.document_processor import get_document_processor

            self._doc_processor = get_document_processor()
        return self._doc_processor

    async def process_files(
        self, file_contents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process files and return content suitable for LLM input.

        Args:
            file_contents: List of file info dicts, each containing:
                - key: The input key name (e.g., "image", "document")
                - mime_type: MIME type of the file
                - filename: Original filename
                - content: Raw file bytes

        Returns:
            Dict with:
                - image_blocks: List of content blocks for vision models
                - extracted_text: Combined text from documents
        """
        image_blocks: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        for file_info in file_contents:
            mime_type = file_info["mime_type"]
            content = file_info["content"]
            filename = file_info["filename"]

            if self._is_image(mime_type):
                block = self._create_image_block(mime_type, content, filename)
                image_blocks.append(block)
            elif self._is_document(mime_type):
                text = await self._extract_document_text(mime_type, content, filename)
                if text:
                    text_parts.append(f"--- Content from {filename} ---\n{text}")
            else:
                logger.warning(
                    "Unsupported file type for LLM: %s (%s)",
                    filename,
                    mime_type,
                )

        return {
            "image_blocks": image_blocks,
            "extracted_text": "\n\n".join(text_parts) if text_parts else "",
        }

    def _is_image(self, mime_type: str) -> bool:
        """Check if MIME type is a supported image type."""
        return mime_type.lower() in IMAGE_MIME_TYPES

    def _is_document(self, mime_type: str) -> bool:
        """Check if MIME type is a supported document type."""
        return mime_type.lower() in DOCUMENT_MIME_TYPES

    def _create_image_block(
        self, mime_type: str, content: bytes, filename: str
    ) -> Dict[str, Any]:
        """
        Create an image content block for LangChain/OpenAI format.

        The format follows the OpenAI vision API structure:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

        Args:
            mime_type: MIME type of the image
            content: Raw image bytes
            filename: Original filename (for logging)

        Returns:
            Image content block dict
        """
        b64_data = base64.b64encode(content).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"

        logger.debug(
            "Created image block: %s (%d bytes, %s)",
            filename,
            len(content),
            mime_type,
        )

        return {"type": "image_url", "image_url": {"url": data_url}}

    async def _extract_document_text(
        self, mime_type: str, content: bytes, filename: str
    ) -> str:
        """
        Extract text from a document.

        Uses the existing DocumentProcessor for PDF, DOCX, and TXT files.

        Args:
            mime_type: MIME type of the document
            content: Raw document bytes
            filename: Original filename (for logging/error messages)

        Returns:
            Extracted text, or empty string on failure
        """
        try:
            doc_processor = self._get_document_processor()
            text = await doc_processor.extract_text(content, mime_type)

            logger.debug(
                "Extracted text from %s: %d characters",
                filename,
                len(text),
            )

            return text
        except Exception as e:  # pylint: disable=broad-exception-caught  # Catch all extraction errors to not fail the LLM call
            logger.warning(
                "Failed to extract text from %s: %s",
                filename,
                str(e),
            )
            return f"[Error extracting text from {filename}: {e}]"


def is_supported_for_llm(mime_type: str) -> bool:
    """
    Check if a MIME type is supported for LLM file input.

    Args:
        mime_type: MIME type to check

    Returns:
        True if the file type can be processed for LLM input
    """
    mime_lower = mime_type.lower()
    return mime_lower in IMAGE_MIME_TYPES or mime_lower in DOCUMENT_MIME_TYPES


__all__ = [
    "FileContentProcessor",
    "IMAGE_MIME_TYPES",
    "DOCUMENT_MIME_TYPES",
    "is_supported_for_llm",
]
