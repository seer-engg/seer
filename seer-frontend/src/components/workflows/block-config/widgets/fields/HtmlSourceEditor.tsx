import { Textarea } from '@/components/ui/textarea';

interface HtmlSourceEditorProps {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  rows: number;
}

export function HtmlSourceEditor({ value, onChange, rows }: HtmlSourceEditorProps) {
  return (
    <Textarea
      value={value}
      onChange={onChange}
      placeholder="<p>Enter HTML here...</p>"
      className="min-h-[120px] border-0 rounded-none focus-visible:ring-0 font-mono text-xs resize-none"
      style={{ minHeight: `${rows * 24 + 40}px` }}
    />
  );
}
