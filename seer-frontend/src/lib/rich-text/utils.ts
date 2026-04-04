/**
 * Check if a string is HTML content (from Tiptap).
 */
export function isHtml(str: string): boolean {
  if (!str) return false;
  return /<(p|div|span|strong|em|br|a|ul|ol|li|h[1-6]|blockquote|code|pre)\b/i.test(str);
}

/**
 * Convert a value to HTML for editor - handles backward compatibility with markdown.
 */
export function toEditorContent(val: string, isHtml: (str: string) => boolean, marked: { parse: (str: string, opts: unknown) => string }): string {
  if (!val) return '';
  // If already HTML, use directly
  if (isHtml(val)) return val;
  // Otherwise, parse as markdown (backward compatibility)
  return marked.parse(val, { async: false, breaks: true }) as string;
}
