import { useCallback, useEffect, useState } from 'react';
import type { Editor } from '@tiptap/react';
import type { ContentMode, EditorMode } from './types';
import { hasComplexHtml } from '@/lib/rich-text/html-detection';

interface UseRichTextModesOptions {
  value: string;
  onChange: (value: string) => void;
  editor: Editor | null;
  toEditorContent: (val: string) => string;
}

interface UseRichTextModesReturn {
  contentMode: ContentMode;
  editorMode: EditorMode;
  rawHtml: string;
  htmlSource: string;
  showSimplifyWarning: boolean;
  handleContentModeChange: (mode: ContentMode) => void;
  handleToggleSubMode: () => void;
  handleHtmlSourceChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  handleRawHtmlChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  handleConfirmSimplify: () => void;
  handleCancelSimplify: () => void;
}

/**
 * Custom hook for managing RichTextField content modes and editor modes.
 *
 * Content modes:
 * - simple: Uses Tiptap for WYSIWYG editing
 * - full-html: Raw HTML mode that preserves all HTML
 *
 * Editor modes (sub-modes within content mode):
 * - visual: Tiptap WYSIWYG (simple mode only)
 * - html-source: Raw HTML textarea
 * - html-preview: Sandboxed iframe preview (full-html mode only)
 */
export function useRichTextModes({
  value,
  onChange,
  editor,
  toEditorContent,
}: UseRichTextModesOptions): UseRichTextModesReturn {
  // Auto-detect complex HTML on mount to preserve formatting (tables, styles, etc.)
  const isComplex = hasComplexHtml(value);
  const [contentMode, setContentMode] = useState<ContentMode>(isComplex ? 'full-html' : 'simple');
  const [editorMode, setEditorMode] = useState<EditorMode>(isComplex ? 'html-source' : 'visual');
  const [rawHtml, setRawHtml] = useState(isComplex ? value : '');
  const [htmlSource, setHtmlSource] = useState('');
  // Warning dialog state
  const [showSimplifyWarning, setShowSimplifyWarning] = useState(false);
  const [pendingModeSwitch, setPendingModeSwitch] = useState<(() => void) | null>(null);

  // Handle content mode change (Simple ↔ Full HTML)
  const handleContentModeChange = useCallback((newMode: ContentMode) => {
    if (newMode === contentMode) return;

    if (newMode === 'full-html' && contentMode === 'simple') {
      // Simple → Full HTML: capture current Tiptap HTML as raw
      const currentHtml = editor?.getHTML() || '';
      setRawHtml(currentHtml);
      setEditorMode('html-source');
      setContentMode(newMode);
      onChange(currentHtml);
    } else if (newMode === 'simple' && contentMode === 'full-html') {
      // Full HTML → Simple: check if content has complex HTML
      if (hasComplexHtml(rawHtml)) {
        setPendingModeSwitch(() => () => {
          editor?.commands.setContent(rawHtml);
          setHtmlSource(rawHtml);
          setEditorMode('visual');
          setContentMode(newMode);
        });
        setShowSimplifyWarning(true);
        return;
      }
      editor?.commands.setContent(rawHtml);
      setEditorMode('visual');
      setContentMode(newMode);
    }
  }, [contentMode, editor, rawHtml, onChange]);

  // Handle warning dialog confirmation
  const handleConfirmSimplify = useCallback(() => {
    if (pendingModeSwitch) {
      pendingModeSwitch();
      setPendingModeSwitch(null);
    }
    setShowSimplifyWarning(false);
  }, [pendingModeSwitch]);

  const handleCancelSimplify = useCallback(() => {
    setPendingModeSwitch(null);
    setShowSimplifyWarning(false);
  }, []);

  // Handle sub-mode toggle
  const handleToggleSubMode = useCallback(() => {
    if (contentMode === 'simple') {
      if (editorMode === 'visual') {
        if (editor) setHtmlSource(editor.getHTML());
        setEditorMode('html-source');
      } else {
        if (editor) {
          editor.commands.setContent(htmlSource);
          onChange(htmlSource);
        }
        setEditorMode('visual');
      }
    } else {
      setEditorMode(editorMode === 'html-source' ? 'html-preview' : 'html-source');
    }
  }, [contentMode, editorMode, editor, htmlSource, onChange]);

  // Handle HTML source changes in simple mode
  const handleHtmlSourceChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setHtmlSource(e.target.value);
  }, []);

  // Handle raw HTML changes in full-html mode
  const handleRawHtmlChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newHtml = e.target.value;
    setRawHtml(newHtml);
    onChange(newHtml);
  }, [onChange]);

  // Sync external value changes — but skip when the editor is focused,
  // since the editor is the source of truth during active editing.
  // Without this guard, markdown output (which strips trailing whitespace)
  // triggers a round-trip that resets editor content and eats spaces.
  useEffect(() => {
    if (contentMode === 'simple' && editorMode === 'visual' && editor) {
      if (editor.isFocused) return;
      const currentHtml = editor.getHTML();
      if (value !== currentHtml && toEditorContent(value) !== currentHtml) {
        editor.commands.setContent(toEditorContent(value));
      }
    } else if (contentMode === 'full-html' && value !== rawHtml) {
      setRawHtml(value);
    }
  }, [value, editor, contentMode, editorMode, rawHtml, toEditorContent]);

  return {
    contentMode,
    editorMode,
    rawHtml,
    htmlSource,
    showSimplifyWarning,
    handleContentModeChange,
    handleToggleSubMode,
    handleHtmlSourceChange,
    handleRawHtmlChange,
    handleConfirmSimplify,
    handleCancelSimplify,
  };
}
