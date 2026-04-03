/**
 * Content Analysis Utilities
 *
 * Utilities for analyzing display values to determine appropriate rendering strategy.
 * Used by HITL display components to choose between compact and rich layouts.
 */

export interface ContentAnalysis {
  /** Content exceeds 100 chars or has multiple lines */
  isLong: boolean;
  /** Content contains markdown syntax */
  isMarkdown: boolean;
  /** Number of lines in content */
  lineCount: number;
  /** Content exceeds collapsible threshold (500 chars or 10 lines) */
  needsCollapse: boolean;
  /** Formatted text representation */
  text: string;
}

/**
 * Format a value for text display
 */
export function formatTextValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '-';
  }
  if (typeof value === 'object') {
    return JSON.stringify(value, null, 2);
  }
  return String(value);
}

/**
 * Detect if text contains markdown syntax
 */
export function isMarkdownContent(text: string): boolean {
  const markdownPatterns = [
    /^#{1,6}\s/m, // Headers
    /\*\*[^*]+\*\*/, // Bold
    /(?<!\*)\*[^*\n]+\*(?!\*)/, // Italic (not bold)
    /```/, // Code blocks
    /^\s*[-*+]\s/m, // Unordered lists
    /^\s*\d+\.\s/m, // Ordered lists
    /\[.+\]\(.+\)/, // Links
    /^\s*>/m, // Blockquotes
    /`[^`]+`/, // Inline code
    /\|.+\|.+\|/m, // Tables
  ];

  return markdownPatterns.some((pattern) => pattern.test(text));
}

/**
 * Analyze content to determine rendering strategy
 */
export function analyzeContent(value: unknown): ContentAnalysis {
  const text = formatTextValue(value);
  const lines = text.split('\n');
  const lineCount = lines.length;

  const isLong = text.length > 100 || lineCount > 1;
  const isMarkdown = isMarkdownContent(text);
  const needsCollapse = text.length > 500 || lineCount > 10;

  return {
    isLong,
    isMarkdown,
    lineCount,
    needsCollapse,
    text,
  };
}
