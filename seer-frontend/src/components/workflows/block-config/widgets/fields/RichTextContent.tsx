import { EditorContent, type Editor } from '@tiptap/react';
import { HtmlSourceEditor } from './HtmlSourceEditor';
import { HtmlPreviewIframe } from './HtmlPreviewIframe';
import type { ContentMode, EditorMode } from './types';

interface RichTextContentProps {
  editor: Editor | null;
  contentMode: ContentMode;
  editorMode: EditorMode;
  htmlSource: string;
  rawHtml: string;
  rows: number;
  onHtmlSourceChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onRawHtmlChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
}

/**
 * Renders the appropriate editor content based on content mode and editor mode.
 *
 * Content rendering matrix:
 * - simple + visual → Tiptap EditorContent
 * - simple + html-source → HtmlSourceEditor (Tiptap's HTML output)
 * - full-html + html-source → HtmlSourceEditor (raw HTML)
 * - full-html + html-preview → HtmlPreviewIframe (sandboxed render)
 */
export function RichTextContent({
  editor,
  contentMode,
  editorMode,
  htmlSource,
  rawHtml,
  rows,
  onHtmlSourceChange,
  onRawHtmlChange,
}: RichTextContentProps) {
  // Simple mode: visual editor
  if (contentMode === 'simple' && editorMode === 'visual') {
    return <EditorContent editor={editor} />;
  }

  // Simple mode: HTML source (Tiptap's output)
  if (contentMode === 'simple' && editorMode === 'html-source') {
    return <HtmlSourceEditor value={htmlSource} onChange={onHtmlSourceChange} rows={rows} />;
  }

  // Full HTML mode: source editor
  if (contentMode === 'full-html' && editorMode === 'html-source') {
    return <HtmlSourceEditor value={rawHtml} onChange={onRawHtmlChange} rows={rows} />;
  }

  // Full HTML mode: iframe preview
  if (contentMode === 'full-html' && editorMode === 'html-preview') {
    return <HtmlPreviewIframe html={rawHtml} rows={rows} />;
  }

  return null;
}
