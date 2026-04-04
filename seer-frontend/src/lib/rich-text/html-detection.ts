/**
 * Check if HTML contains elements that Tiptap will strip.
 * Used to warn users when switching from Full HTML to Simple mode.
 *
 * Tiptap's StarterKit only supports: paragraphs, headings, bold/italic/underline,
 * links, lists, code blocks, and blockquotes. Everything else gets stripped.
 */
export function hasComplexHtml(html: string): boolean {
  if (!html) return false;

  const complexPatterns = [
    /<table[\s>]/i,       // Table-based layouts (email templates)
    /<div[\s>]/i,         // Div layouts (stripped to plain text)
    /<style[\s>]/i,       // Style blocks
    /style\s*=/i,         // Inline styles
    /<img[\s>]/i,         // Images
    /<svg[\s>]/i,         // SVG graphics
    /<!DOCTYPE/i,         // Full document declarations
    /<html[\s>]/i,        // HTML document root
    /<head[\s>]/i,        // Head section
    /<body[\s>]/i,        // Body section
    /<center[\s>]/i,      // Legacy center tags (common in emails)
    /<font[\s>]/i,        // Legacy font tags
    /class\s*=/i,         // Class attributes (styling)
    /bgcolor\s*=/i,       // Background color attributes
    /width\s*=/i,         // Width attributes
    /height\s*=/i,        // Height attributes
    /cellpadding/i,       // Table-specific attributes
    /cellspacing/i,
  ];

  return complexPatterns.some(pattern => pattern.test(html));
}
