import { useEffect, useRef } from 'react';
import type { KeyboardEvent } from 'react';

import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { ArrowUp, Square } from 'lucide-react';
import { cn } from '@/lib/utils';

import type { ModelInfo } from '../buildtypes';
import { useChatStore } from '@/stores';

interface ChatInputProps {
  onSend: (overrideInput?: string) => void;
  onStop?: () => void;
  models: ModelInfo[];
  isLoadingModels: boolean;
  isInterruptPending?: boolean;
}

export function ChatInput({onSend, onStop, models, isLoadingModels,
  isInterruptPending = false,
}: ChatInputProps) {
  const input = useChatStore((state) => state.input);
  const isLoading = useChatStore((state) => state.isLoading);
  const selectedModel = useChatStore((state) => state.selectedModel);
  const setInput = useChatStore((state) => state.setInput);
  const setSelectedModel = useChatStore((state) => state.setSelectedModel);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    const raf = requestAnimationFrame(() => {
      el.style.height = 'auto';
      el.style.height = `${Math.max(60, Math.min(el.scrollHeight, 200))}px`;
    });
    return () => cancelAnimationFrame(raf);
  }, [input]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const canSend = !isLoading && !!input.trim();

  return (
    <div className="px-3 pb-3 pt-1 flex-shrink-0">
      <div
        className={cn(
          'flex flex-col rounded-2xl border bg-background shadow-sm transition-all duration-150',
          'focus-within:shadow-md focus-within:border-[hsl(var(--seer)/0.4)]',
          isLoading ? 'opacity-70' : 'border-border',
        )}
      >
        <Textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isInterruptPending ? 'Reply to the clarification in your own words...' : 'Describe your workflow or pick a template above...'}
          disabled={isLoading}
          className={cn(
            'min-h-[60px] resize-none w-full border-0 bg-transparent shadow-none',
            'focus-visible:ring-0 focus-visible:ring-offset-0',
            'px-4 pt-3 pb-1 text-sm placeholder:text-muted-foreground/60',
          )}
          style={{ maxHeight: '200px' }}
        />
        <div className="flex items-center justify-between px-3 pb-2.5 pt-1">
          <Select
            value={selectedModel}
            onValueChange={setSelectedModel}
            disabled={isLoadingModels || isLoading || isInterruptPending}
          >
            <SelectTrigger className="h-6 text-xs w-auto max-w-[130px] border-0 bg-transparent p-0 shadow-none gap-1 focus:ring-0 text-muted-foreground hover:text-foreground transition-colors">
              <SelectValue placeholder={isLoadingModels ? 'Loading...' : 'Model'} />
            </SelectTrigger>
            <SelectContent>
              {models
                .filter((model) => model.available)
                .map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.name}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
          {isLoading && onStop ? (
            <Button
              onClick={onStop}
              size="icon"
              className="h-7 w-7 rounded-lg bg-destructive hover:bg-destructive/90 text-destructive-foreground shadow-sm transition-all duration-150"
            >
              <Square className="w-3 h-3" />
            </Button>
          ) : (
            <Button
              onClick={() => onSend()}
              disabled={!canSend}
              size="icon"
              className={cn(
                'h-7 w-7 rounded-lg transition-all duration-150',
                canSend
                  ? 'bg-[hsl(var(--seer))] hover:bg-[hsl(var(--seer)/0.85)] text-white shadow-sm'
                  : 'bg-muted text-muted-foreground',
              )}
            >
              <ArrowUp className="w-3.5 h-3.5" />
            </Button>
          )}
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground/50 text-center mt-1.5">
        {isInterruptPending
          ? 'Resume mode: reply here or use the clarification card above. Start a new session for a different request.'
          : 'Enter to send · Shift+Enter for new line'}
      </p>
    </div>
  );
}
