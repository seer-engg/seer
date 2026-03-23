"""
Rich Text Converter - Unified format conversion for workflow tools.

Converts canonical Markdown to platform-specific formats:
- HTML (Gmail, Calendar)
- Slack mrkdwn
- Discord markdown

Usage:
    from seer.utils.rich_text import RichTextConverter, Platform

    converter = RichTextConverter()
    html = converter.convert(markdown_text, Platform.EMAIL)
    slack_text = converter.convert(markdown_text, Platform.SLACK)
"""

import re
from enum import Enum
from typing import Union


class Platform(Enum):
    """Supported platforms for rich text conversion."""

    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    CALENDAR = "calendar"


class RichTextConverter:
    """
    Converts canonical Markdown to platform-specific formats.

    Canonical format: Standard Markdown (CommonMark subset)
    - **bold** or __bold__
    - *italic* or _italic_
    - ~~strikethrough~~
    - `code`
    - ```code block```
    - > blockquote
    - [link text](url)
    - - bullet list
    - 1. numbered list
    """

    def convert(self, content: str, platform: Platform) -> Union[str, dict, list]:
        """
        Convert markdown to platform-specific format.

        Args:
            content: Markdown-formatted text
            platform: Target platform

        Returns:
            Converted text in platform format
        """
        if not content:
            return ""

        converters = {
            Platform.EMAIL: self.to_html,
            Platform.SLACK: self.to_slack_mrkdwn,
            Platform.DISCORD: self.to_discord,
            Platform.CALENDAR: self.to_html,
        }
        return converters[platform](content)

    def to_html(self, content: str) -> str:
        """
        Convert to HTML (Gmail, Calendar).

        Uses the markdown library with extensions for proper email rendering.
        """
        import markdown as md  # pylint: disable=import-outside-toplevel  # Reason: Lazy import to avoid startup cost

        body = md.markdown(content, extensions=["extra", "nl2br"])
        return (
            '<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;'
            'font-size:14px;line-height:1.6;color:#333;max-width:680px">'
            f"{body}</body></html>"
        )

    def to_slack_mrkdwn(self, content: str) -> str:
        """
        Convert to Slack mrkdwn format.

        Slack mrkdwn differences from standard Markdown:
        - *bold* instead of **bold**
        - _italic_ instead of *italic*
        - ~strikethrough~ instead of ~~strikethrough~~
        - <url|link text> instead of [link text](url)
        - No underline support
        """
        result = content

        # Convert strikethrough first: ~~text~~ -> ~text~
        result = re.sub(r"~~(.+?)~~", r"~\1~", result)

        # Convert links: [text](url) -> <url|text>
        result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", result)

        # Use a function to handle bold conversion with placeholders
        bold_placeholder = "\u0000SLACKBOLD\u0000"

        def replace_bold(match: re.Match) -> str:
            return f"{bold_placeholder}{match.group(1)}{bold_placeholder}"

        # Step 1: Convert **bold** to placeholder to avoid conflict with italic
        result = re.sub(r"\*\*(.+?)\*\*", replace_bold, result)

        # Step 2: Convert __bold__ to placeholder
        result = re.sub(r"__(.+?)__", replace_bold, result)

        # Step 3: Convert *italic* to _italic_
        result = re.sub(r"\*(.+?)\*", r"_\1_", result)

        # Step 4: Restore bold placeholders to Slack bold (*text*)
        result = result.replace(bold_placeholder, "*")

        # Blockquotes: > text stays as > text (Slack supports this)
        # Code blocks: ```code``` stays as ```code``` (Slack supports this)
        # Inline code: `code` stays as `code` (Slack supports this)

        return result

    def to_discord(self, content: str) -> str:
        """
        Convert to Discord markdown.

        Discord uses mostly standard markdown with some additions:
        - ||spoiler|| for spoiler text
        - >>> for multi-line quotes
        - Underline: __text__ (standard markdown uses this for bold)

        Since we use **bold** for bold, Discord handles most things natively.
        Main conversion needed: ensure underline is separate from bold.
        """
        result = content

        # Discord uses standard markdown for most things
        # **bold**, *italic*, ~~strikethrough~~, `code`, ```code blocks```, > quotes

        # Convert __text__ (markdown bold alternative) to Discord bold **text**
        # because Discord uses __text__ for underline
        result = re.sub(r"__(.+?)__", r"**\1**", result)

        # Links: Discord doesn't support [text](url) in regular messages
        # but shows URLs with auto-embed. Keep as-is for now.

        return result


# Convenience function for direct use
def convert_rich_text(content: str, platform: Union[Platform, str]) -> Union[str, dict, list]:
    """
    Convert markdown content to platform-specific format.

    Args:
        content: Markdown-formatted text
        platform: Target platform (Platform enum or string)

    Returns:
        Converted text in platform format
    """
    if isinstance(platform, str):
        platform = Platform(platform)

    converter = RichTextConverter()
    return converter.convert(content, platform)
