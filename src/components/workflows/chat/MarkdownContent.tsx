import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Components } from 'react-markdown';

import { cn } from '@/lib/utils';

interface MarkdownContentProps {
  content: string;
  className?: string;
}

function createCodeComponent(isDark: boolean): Components['code'] {
  return function CodeComponent({ className: codeClassName, children, ...props }) {
    const match = /language-(\w+)/.exec(codeClassName || '');
    const language = match ? match[1] : '';
    const codeString = String(children).replace(/\n$/, '');
    const isInline = !match && !codeString.includes('\n');

    if (isInline) {
      return (
        <code className="px-1.5 py-0.5 rounded bg-muted font-mono text-[0.85em]" {...props}>
          {children}
        </code>
      );
    }

    return (
      <SyntaxHighlighter
        style={isDark ? oneDark : oneLight}
        language={language || 'text'}
        PreTag="div"
        customStyle={{ margin: 0, borderRadius: '0.375rem', fontSize: '0.8125rem' }}
      >
        {codeString}
      </SyntaxHighlighter>
    );
  };
}

const markdownComponents: Partial<Components> = {
  p: ({ children }) => <p className="mb-2 break-words last:mb-0">{children}</p>,
  h1: ({ children }) => <h1 className="text-lg font-semibold mt-3 mb-2 first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-semibold mt-3 mb-2 first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mt-2 mb-1 first:mt-0">{children}</h3>,
  h4: ({ children }) => <h4 className="text-sm font-medium mt-2 mb-1 first:mt-0">{children}</h4>,
  ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
  li: ({ children }) => <li className="ml-1">{children}</li>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-seer-500 hover:text-seer-600 dark:text-seer-400 dark:hover:text-seer-300 underline underline-offset-2"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-muted-foreground/30 pl-3 italic text-muted-foreground my-2">
      {children}
    </blockquote>
  ),
  pre: ({ children }) => <div className="my-2 overflow-x-auto">{children}</div>,
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="min-w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-muted/50">{children}</thead>,
  th: ({ children }) => (
    <th className="border border-border px-2 py-1 text-left font-medium">{children}</th>
  ),
  td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
  hr: () => <hr className="my-3 border-border" />,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
};

export function MarkdownContent({ content, className }: MarkdownContentProps) {
  const isDark = document.documentElement.classList.contains('dark');

  const components: Components = {
    ...markdownComponents,
    code: createCodeComponent(isDark),
  };

  return (
    <div className={cn('min-w-0 max-w-full text-left text-sm break-words [&_pre]:max-w-full [&_table]:max-w-full [&_td]:break-words [&_th]:break-words [&_code]:break-all', className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
