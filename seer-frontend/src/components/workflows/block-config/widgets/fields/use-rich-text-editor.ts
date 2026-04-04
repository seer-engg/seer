import { useEditor, type Editor, ReactNodeViewRenderer } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Link from '@tiptap/extension-link';
import Underline from '@tiptap/extension-underline';
import Placeholder from '@tiptap/extension-placeholder';
import CharacterCount from '@tiptap/extension-character-count';
import { Markdown, type MarkdownStorage } from 'tiptap-markdown';
import type { TemplateAutocompleteControls } from '../../types';
import type { RichTextOutputFormat } from './RichTextField';
import { TemplateVariable } from './extensions/template-variable';
import { TemplateVariableView } from './extensions/TemplateVariableView';

interface UseRichTextEditorProps {
  initialContent: string;
  placeholder?: string;
  charLimit: number | null;
  rows: number;
  id: string;
  isHtmlMode: boolean;
  outputFormat: RichTextOutputFormat;
  onChange: (content: string) => void;
  onFocus: () => void;
  onBlur: () => void;
  templateAutocomplete: TemplateAutocompleteControls;
}

export function useRichTextEditor({
  initialContent,
  placeholder,
  charLimit,
  rows,
  id,
  isHtmlMode,
  outputFormat,
  onChange,
  onFocus,
  onBlur,
  templateAutocomplete,
}: UseRichTextEditorProps): Editor | null {
  const { checkForAutocomplete } = templateAutocomplete;

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: { levels: [1, 2, 3] },
      }),
      Link.configure({
        openOnClick: false,
        HTMLAttributes: { class: 'text-primary underline' },
      }),
      Underline,
      Placeholder.configure({
        placeholder: placeholder || 'Start typing...',
      }),
      ...(charLimit ? [CharacterCount.configure({ limit: charLimit })] : []),
      TemplateVariable.extend({
        addNodeView() {
          return ReactNodeViewRenderer(TemplateVariableView);
        },
      }),
      Markdown,
    ],
    content: initialContent,
    onUpdate: ({ editor: ed }) => {
      if (isHtmlMode) return;
      const output = outputFormat === 'markdown'
        ? (ed.storage.markdown as MarkdownStorage).getMarkdown()
        : ed.getHTML();
      onChange(output);

      // Autocomplete check
      const text = ed.getText();
      const cursorPos = ed.state.selection.from;
      checkForAutocomplete(text, cursorPos, {
        inputId: id,
        ref: { current: null } as React.RefObject<HTMLTextAreaElement>,
        value: text,
        onChange: () => {},
      });
    },
    onFocus,
    onBlur,
    editorProps: {
      attributes: {
        class: 'focus:outline-none text-left min-h-[80px] px-3 py-2',
        style: `min-height: ${rows * 24}px`,
      },
    },
  });

  return editor;
}
