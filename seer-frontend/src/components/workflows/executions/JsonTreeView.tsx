import { useState, useRef, useEffect } from 'react';
import { ChevronRight, ChevronDown, Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface JsonTreeViewProps {
  data: unknown;
  className?: string;
}

// Copy button component
function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); }, []);

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    await navigator.clipboard.writeText(content);
    setCopied(true);
    timeoutRef.current = setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-5 w-5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
      onClick={handleCopy}
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
    </Button>
  );
}

// Render primitive values with appropriate styling
function PrimitiveValue({ value }: { value: unknown }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (value === null) {
    return <span className="text-muted-foreground italic">null</span>;
  }
  if (value === undefined) {
    return <span className="text-muted-foreground italic">undefined</span>;
  }
  if (typeof value === 'boolean') {
    return (
      <span className="text-amber-600 dark:text-amber-400">
        {value ? 'true' : 'false'}
      </span>
    );
  }
  if (typeof value === 'number') {
    return <span className="text-blue-600 dark:text-blue-400">{value}</span>;
  }
  if (typeof value === 'string') {
    const maxLength = 100;
    const truncated = value.length > maxLength;
    const displayValue = isExpanded || !truncated ? value : value.slice(0, maxLength) + '...';

    return (
      <span className="text-emerald-600 dark:text-emerald-400">
        "{displayValue}"
        {truncated && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
            className="text-muted-foreground text-[10px] ml-1 hover:text-foreground hover:underline cursor-pointer"
          >
            {isExpanded ? 'show less' : `(${value.length} chars)`}
          </button>
        )}
      </span>
    );
  }
  return <span>{String(value)}</span>;
}

// Helper to get object entries
function getEntries(value: unknown): Array<[string | number, unknown]> {
  if (typeof value !== 'object' || value === null) return [];
  if (Array.isArray(value)) return value.map((v, i) => [i, v]);
  return Object.entries(value as Record<string, unknown>);
}

// Helper to render the value summary
function ValueSummary({ value, entries, isExpanded }: {
  value: unknown;
  entries: Array<[string | number, unknown]>;
  isExpanded: boolean;
}) {
  const isArray = Array.isArray(value);
  const isEmpty = entries.length === 0;

  if (isEmpty) {
    return <span className="text-muted-foreground">{isArray ? '[]' : '{}'}</span>;
  }
  if (isExpanded) {
    return (
      <span className="text-muted-foreground">
        {isArray ? `[${entries.length}]` : `{${entries.length}}`}
      </span>
    );
  }
  return (
    <span className="text-muted-foreground">
      {isArray ? `[${entries.length} items]` : `{${entries.length} keys}`}
    </span>
  );
}

// Tree node for objects and arrays
interface TreeNodeProps {
  name: string | number;
  value: unknown;
  depth: number;
  isLast: boolean;
}

function TreeNode({ name, value, depth, isLast }: TreeNodeProps) {
  const isObject = typeof value === 'object' && value !== null;
  const [isExpanded, setIsExpanded] = useState(depth < 2);
  const entries = getEntries(value);
  const isEmpty = entries.length === 0;
  const prefix = isLast ? '└─' : '├─';
  const isExpandable = isObject && !isEmpty;

  return (
    <div className="group">
      <div
        className={cn(
          'flex items-start gap-1 py-0.5 hover:bg-muted/50 rounded -mx-1 px-1',
          isExpandable && 'cursor-pointer'
        )}
        onClick={() => isExpandable && setIsExpanded(!isExpanded)}
      >
        <span className="text-muted-foreground/50 select-none shrink-0 font-mono">
          {depth > 0 && prefix}
        </span>

        {isExpandable ? (
          <span className="shrink-0 text-muted-foreground w-4">
            {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          </span>
        ) : (
          <span className="w-4 shrink-0" />
        )}

        <span className="text-foreground font-medium shrink-0">{name}</span>
        <span className="text-muted-foreground shrink-0">: </span>

        {isObject ? (
          <ValueSummary value={value} entries={entries} isExpanded={isExpanded} />
        ) : (
          <span className="min-w-0 break-all">
            <PrimitiveValue value={value} />
          </span>
        )}

        <CopyButton content={JSON.stringify(value, null, 2)} />
      </div>

      {isExpandable && isExpanded && (
        <div className={cn('ml-4', depth > 0 && 'border-l border-border/30')}>
          {entries.map(([key, val], idx: number) => (
            <TreeNode
              key={String(key)}
              name={key}
              value={val}
              depth={depth + 1}
              isLast={idx === entries.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// Main component
export function JsonTreeView({ data, className }: JsonTreeViewProps) {
  if (data === null || data === undefined) {
    return (
      <div className={cn('font-mono text-xs', className)}>
        <PrimitiveValue value={data} />
      </div>
    );
  }

  const isObject = typeof data === 'object';
  const isArray = Array.isArray(data);

  if (!isObject) {
    return (
      <div className={cn('font-mono text-xs', className)}>
        <PrimitiveValue value={data} />
      </div>
    );
  }

  const entries = isArray
    ? data.map((v, i) => [i, v] as [number, unknown])
    : Object.entries(data as Record<string, unknown>);

  if (entries.length === 0) {
    return (
      <div className={cn('font-mono text-xs text-muted-foreground', className)}>
        {isArray ? '[]' : '{}'}
      </div>
    );
  }

  return (
    <div className={cn('font-mono text-xs', className)}>
      {entries.map(([key, value], idx: number) => (
        <TreeNode
          key={String(key)}
          name={key}
          value={value}
          depth={0}
          isLast={idx === entries.length - 1}
        />
      ))}
    </div>
  );
}

// Raw JSON view with syntax highlighting
interface JsonRawViewProps {
  data: unknown;
  className?: string;
}

export function JsonRawView({ data, className }: JsonRawViewProps) {
  const jsonString = typeof data === 'string' ? data : JSON.stringify(data, null, 2);

  return (
    <pre
      className={cn(
        'font-mono text-xs bg-muted/50 p-3 rounded-md',
        'overflow-x-auto whitespace-pre-wrap break-words',
        className
      )}
    >
      {jsonString}
    </pre>
  );
}
