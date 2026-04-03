import { useCallback, useState, useRef } from 'react';
import { marked } from 'marked';
import type { Editor } from '@tiptap/react';
import { cn } from '@/lib/utils';
import { RichTextEditorHeader } from './RichTextEditorHeader';
import { RichTextContent } from './RichTextContent';
import { CharacterCounter } from './CharacterCounter';
import { useRichTextEditor } from './use-rich-text-editor';
import { useRichTextModes } from './useRichTextModes';
import { isHtml } from '@/lib/rich-text/utils';
import type { TemplateAutocompleteControls } from '../../types';
import { VariableAutocompleteDropdown } from '../VariableAutocompleteDropdown';
import { SimplifyWarningDialog } from './SimplifyWarningDialog';

export type RichTextOutputFormat = 'html' | 'markdown';

export interface RichTextFieldProps {
  id: string;
  value: string;
  onChange: (value: string) => void;
  outputFormat?: RichTextOutputFormat;
  /** Allowed formatting features from tool schema's x-rich-text-features */
  features?: string[];
  /** Character limit from tool schema's maxLength */
  charLimit?: number;
  placeholder?: string;
  showError?: boolean;
  templateAutocomplete: TemplateAutocompleteControls;
  rows?: number;
}

// Regex to match ${...} template variable expressions
const TEMPLATE_VAR_RE = /\$\{\s*([^}]+?)\s*\}/g;

// Convert ${path} text patterns to atom node HTML so TipTap parses them as TemplateVariable nodes
function injectTemplateVarNodes(html: string): string {
  return html.replace(
    TEMPLATE_VAR_RE,
    (_, path: string) =>
      `<span data-template-var data-path="${path.trim()}">\${${path.trim()}}</span>`,
  );
}

// Convert value to HTML for editor - handles backward compatibility
function toEditorContent(val: string): string {
  if (!val) return '';
  if (isHtml(val)) {
    return injectTemplateVarNodes(val);
  }
  // Markdown → HTML, then inject atom nodes for any {{...}} patterns
  const html = marked.parse(val, { async: false, breaks: true }) as string;
  return injectTemplateVarNodes(html);
}

/**
 * Insert a template variable node into the TipTap editor,
 * replacing any preceding "/" trigger character.
 */
function insertVariableIntoEditor(
  editor: Editor | null,
  variable: string,
) {
  if (!editor) return;
  const { state } = editor;
  const { from } = state.selection;
  const textBefore = state.doc.textBetween(Math.max(0, from - 50), from);
  const braceOffset = textBefore.lastIndexOf('/');
  const content = [
    { type: 'templateVariable', attrs: { path: variable } },
    { type: 'text', text: ' ' },
  ];
  if (braceOffset !== -1) {
    const deleteFrom = from - (textBefore.length - braceOffset);
    editor.chain().deleteRange({ from: deleteFrom, to: from }).insertContent(content).run();
  } else {
    editor.commands.insertContent(content);
  }
}

/**
 * Rich text editor field with schema-driven formatting support.
 *
 * Dual-mode system:
 * - Simple Editor: Uses Tiptap for WYSIWYG editing (HTML parsed through schema)
 * - Full HTML: Raw HTML mode that preserves all HTML (tables, styles, etc.)
 */
export function RichTextField({
  id,
  value,
  onChange,
  outputFormat = 'html',
  features,
  charLimit,
  placeholder,
  showError,
  templateAutocomplete,
  rows = 4,
}: RichTextFieldProps) {
  const [isFocused, setIsFocused] = useState(false);
  const editorContainerRef = useRef<HTMLDivElement>(null);
  const { autocompleteContext, closeAutocomplete, currentLevelItems, currentPath, drillInto, navigateTo, selectedIndex, showAutocomplete } = templateAutocomplete;

  const handleFocus = useCallback(() => setIsFocused(true), []);
  const handleBlur = useCallback(() => {
    setIsFocused(false);
    setTimeout(() => closeAutocomplete(), 200);
  }, [closeAutocomplete]);

  const editor = useRichTextEditor({
    initialContent: toEditorContent(value),
    placeholder,
    charLimit: charLimit ?? null,
    rows,
    id,
    isHtmlMode: false,
    outputFormat,
    onChange,
    onFocus: handleFocus,
    onBlur: handleBlur,
    templateAutocomplete,
  });

  const modes = useRichTextModes({ value, onChange, editor, toEditorContent });
  const { contentMode, editorMode, rawHtml, htmlSource, showSimplifyWarning } = modes;

  const handleInsertVariable = useCallback(
    (variable: string) => {
      if (contentMode === 'simple' && editorMode === 'visual' && editor) {
        insertVariableIntoEditor(editor, variable);
        closeAutocomplete();
      }
    },
    [editor, closeAutocomplete, contentMode, editorMode],
  );

  const isVisualMode = contentMode === 'simple' && editorMode === 'visual';
  const dropdownVisible = showAutocomplete && autocompleteContext?.inputId === id && isVisualMode;
  const charCount = editor?.storage.characterCount?.characters() ?? 0;

  return (
    <div
      ref={editorContainerRef}
      className={cn(
        'relative rounded-md border bg-background',
        isFocused && 'ring-2 ring-ring ring-offset-2',
        showError && 'border-destructive'
      )}
    >
      <RichTextEditorHeader
        editor={editor}
        features={features}
        outputFormat={outputFormat}
        contentMode={contentMode}
        editorMode={editorMode}
        onContentModeChange={modes.handleContentModeChange}
        onToggleSubMode={modes.handleToggleSubMode}
      />

      <RichTextContent
        editor={editor}
        contentMode={contentMode}
        editorMode={editorMode}
        htmlSource={htmlSource}
        rawHtml={rawHtml}
        rows={rows}
        onHtmlSourceChange={modes.handleHtmlSourceChange}
        onRawHtmlChange={modes.handleRawHtmlChange}
      />

      {charLimit && isVisualMode && (
        <CharacterCounter charCount={charCount} charLimit={charLimit} />
      )}

      <VariableAutocompleteDropdown
        visible={dropdownVisible}
        items={currentLevelItems}
        selectedIndex={selectedIndex}
        currentPath={currentPath}
        onSelect={handleInsertVariable}
        onDrillInto={drillInto}
        onNavigateTo={navigateTo}
      />

      <SimplifyWarningDialog
        open={showSimplifyWarning}
        onConfirm={modes.handleConfirmSimplify}
        onCancel={modes.handleCancelSimplify}
      />
    </div>
  );
}
