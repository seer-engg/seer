import { type Editor } from '@tiptap/react';
import {
  Bold,
  Italic,
  Underline,
  Strikethrough,
  Link,
  List,
  ListOrdered,
  Quote,
  Code,
  Code2,
  Heading2,
} from 'lucide-react';
// Extended chain type that includes commands from extensions
type ExtendedChain = {
  toggleBold: () => ExtendedChain;
  toggleItalic: () => ExtendedChain;
  toggleUnderline: () => ExtendedChain;
  toggleStrike: () => ExtendedChain;
  toggleBulletList: () => ExtendedChain;
  toggleOrderedList: () => ExtendedChain;
  toggleBlockquote: () => ExtendedChain;
  toggleCode: () => ExtendedChain;
  toggleCodeBlock: () => ExtendedChain;
  toggleHeading: (attrs: { level: number }) => ExtendedChain;
  setLink: (attrs: { href: string }) => ExtendedChain;
  unsetLink: () => ExtendedChain;
  extendMarkRange: (type: string) => ExtendedChain;
  run: () => boolean;
  focus: () => ExtendedChain;
};

export interface ToolbarFeature {
  feature: string;
  icon: React.ReactNode;
  tooltip: string;
  action: () => void;
  isActive: () => boolean;
}

export function createToolbarFeatures(editor: Editor): ToolbarFeature[] {
  const chain = () => editor.chain().focus() as unknown as ExtendedChain;

  const setLink = () => {
    const previousUrl = editor.getAttributes('link').href;
    const url = window.prompt('URL', previousUrl);

    if (url === null) return;

    if (url === '') {
      chain().extendMarkRange('link').unsetLink().run();
      return;
    }

    chain().extendMarkRange('link').setLink({ href: url }).run();
  };

  return [
    {
      feature: 'bold',
      icon: <Bold className="h-4 w-4" />,
      tooltip: 'Bold (Ctrl+B)',
      action: () => chain().toggleBold().run(),
      isActive: () => editor.isActive('bold'),
    },
    {
      feature: 'italic',
      icon: <Italic className="h-4 w-4" />,
      tooltip: 'Italic (Ctrl+I)',
      action: () => chain().toggleItalic().run(),
      isActive: () => editor.isActive('italic'),
    },
    {
      feature: 'underline',
      icon: <Underline className="h-4 w-4" />,
      tooltip: 'Underline (Ctrl+U)',
      action: () => chain().toggleUnderline().run(),
      isActive: () => editor.isActive('underline'),
    },
    {
      feature: 'strikethrough',
      icon: <Strikethrough className="h-4 w-4" />,
      tooltip: 'Strikethrough',
      action: () => chain().toggleStrike().run(),
      isActive: () => editor.isActive('strike'),
    },
    {
      feature: 'link',
      icon: <Link className="h-4 w-4" />,
      tooltip: 'Add Link',
      action: setLink,
      isActive: () => editor.isActive('link'),
    },
    {
      feature: 'bulletList',
      icon: <List className="h-4 w-4" />,
      tooltip: 'Bullet List',
      action: () => chain().toggleBulletList().run(),
      isActive: () => editor.isActive('bulletList'),
    },
    {
      feature: 'orderedList',
      icon: <ListOrdered className="h-4 w-4" />,
      tooltip: 'Numbered List',
      action: () => chain().toggleOrderedList().run(),
      isActive: () => editor.isActive('orderedList'),
    },
    {
      feature: 'blockquote',
      icon: <Quote className="h-4 w-4" />,
      tooltip: 'Quote',
      action: () => chain().toggleBlockquote().run(),
      isActive: () => editor.isActive('blockquote'),
    },
    {
      feature: 'code',
      icon: <Code className="h-4 w-4" />,
      tooltip: 'Inline Code',
      action: () => chain().toggleCode().run(),
      isActive: () => editor.isActive('code'),
    },
    {
      feature: 'codeBlock',
      icon: <Code2 className="h-4 w-4" />,
      tooltip: 'Code Block',
      action: () => chain().toggleCodeBlock().run(),
      isActive: () => editor.isActive('codeBlock'),
    },
    {
      feature: 'heading',
      icon: <Heading2 className="h-4 w-4" />,
      tooltip: 'Heading',
      action: () => chain().toggleHeading({ level: 2 }).run(),
      isActive: () => editor.isActive('heading', { level: 2 }),
    },
  ];
}
