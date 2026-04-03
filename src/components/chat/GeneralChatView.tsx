import { useEffect, useRef, useState } from 'react';
import { Send, Square, ImageIcon, Sparkles, Paperclip, X, Copy, Check } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { useGeneralChatStore } from '@/stores/generalChatStore';
import { MarkdownContent } from '@/components/workflows/chat/MarkdownContent';
import { backendApiClient } from '@/lib/api-client';
import { modelKeys } from '@/lib/query-keys';
import type { ModelInfo } from '@/components/workflows/buildtypes';

interface GeneralChatViewProps {
  onSend: () => void;
  onStop?: () => void;
}

interface ChatMessage {
  id: string | number;
  role: 'user' | 'assistant';
  content: string;
  image_urls?: string[];
}

const SUGGESTION_CHIPS = [
  'Explain how workflows work',
  'Help me debug an issue',
  'Write a Python script',
  'Summarize a document',
];

function EmptyState({ modelName, onChipClick }: { modelName: string; onChipClick: (t: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-seer-500 to-seer-700 flex items-center justify-center mb-5">
        <Sparkles className="w-7 h-7 text-white" />
      </div>
      <h2 className="text-2xl font-semibold mb-1">How can I help?</h2>
      <p className="text-sm text-muted-foreground mb-8">
        Using <span className="font-medium text-foreground">{modelName}</span>
      </p>
      <div className="flex flex-wrap justify-center gap-2 max-w-lg">
        {SUGGESTION_CHIPS.map((chip) => (
          <button
            key={chip}
            onClick={() => onChipClick(chip)}
            className="px-3 py-1.5 text-sm rounded-full border border-border bg-background hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
          >
            {chip}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className={cn('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
      <div className={cn(
        'max-w-[85%] min-w-0 group relative rounded-2xl px-4 py-2.5',
        msg.role === 'user'
          ? 'bg-muted rounded-br-md'
          : 'bg-muted/50 rounded-bl-md'
      )}>
        {msg.role === 'assistant' && (
          <button onClick={handleCopy} className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-muted">
            {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5 text-muted-foreground" />}
          </button>
        )}
        <div className={cn('prose prose-sm dark:prose-invert max-w-none', msg.role === 'user' && 'prose-p:my-0')}>
          <MarkdownContent content={msg.content} />
        </div>
        {msg.image_urls && msg.image_urls.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {msg.image_urls.map((url, i) => (
              <img key={i} src={url} alt={`Generated image ${i + 1}`} className="max-w-sm rounded-lg border" />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface AttachedFile {
  name: string;
  content: string;
}

interface ChatInputBarProps {
  input: string;
  setInput: (v: string) => void;
  isLoading: boolean;
  selectedModel: string;
  setSelectedModel: (v: string) => void;
  isLoadingModels: boolean;
  models: ModelInfo[];
  generateImage: boolean;
  setGenerateImage: (v: boolean) => void;
  attachedFiles: AttachedFile[];
  onAttachFiles: (files: FileList) => void;
  onRemoveFile: (name: string) => void;
  onSend: () => void;
  onStop?: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}

function ChatInputBar({ input, setInput, isLoading, selectedModel, setSelectedModel, isLoadingModels, models, generateImage, setGenerateImage, attachedFiles, onAttachFiles, onRemoveFile, onSend, onStop, onKeyDown, textareaRef }: ChatInputBarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="border-t bg-background/80 backdrop-blur-sm">
      <div className="max-w-3xl mx-auto p-4">
        <div className="rounded-xl border bg-card shadow-sm">
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept=".txt,.md,.csv,.json,.xml,.log"
            multiple
            onChange={(e) => { if (e.target.files) { onAttachFiles(e.target.files); e.target.value = ''; } }}
          />
          {attachedFiles.length > 0 && (
            <div className="flex flex-wrap gap-1.5 px-4 pt-3">
              {attachedFiles.map((f) => (
                <span key={f.name} className="inline-flex items-center gap-1 text-xs bg-muted rounded-md px-2 py-1">
                  {f.name}
                  <button onClick={() => onRemoveFile(f.name)} className="hover:text-destructive"><X className="w-3 h-3" /></button>
                </span>
              ))}
            </div>
          )}
          <textarea
            ref={textareaRef}
            className="w-full resize-none bg-transparent px-4 pt-3 pb-1 text-sm placeholder:text-muted-foreground focus:outline-none min-h-[44px] max-h-[200px]"
            placeholder="Message Seer..."
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = e.target.scrollHeight + 'px';
            }}
            onKeyDown={onKeyDown}
            disabled={isLoading}
          />
          <div className="flex items-center justify-between px-3 pb-2">
            <div className="flex items-center gap-1">
              <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isLoadingModels}>
                <SelectTrigger className="h-7 text-xs border-0 bg-transparent shadow-none hover:bg-muted px-2 gap-1 w-auto max-w-[160px]">
                  <SelectValue placeholder={isLoadingModels ? 'Loading...' : 'Model'} />
                </SelectTrigger>
                <SelectContent>
                  {models.filter((m) => m.available).map((m) => (
                    <SelectItem key={m.id} value={m.id}>{m.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button variant="ghost" size="sm" className="h-7 text-xs px-2" onClick={() => fileInputRef.current?.click()}>
                <Paperclip className="w-3.5 h-3.5 mr-1" />
                Attach
              </Button>
              <Button variant={generateImage ? 'secondary' : 'ghost'} size="sm" className="h-7 text-xs px-2" onClick={() => setGenerateImage(!generateImage)}>
                <ImageIcon className="w-3.5 h-3.5 mr-1" />
                Image
              </Button>
            </div>
            {isLoading && onStop ? (
              <Button onClick={onStop} size="icon" className="h-7 w-7 rounded-lg bg-destructive hover:bg-destructive/90 text-destructive-foreground">
                <Square className="w-3 h-3" />
              </Button>
            ) : (
              <Button onClick={onSend} disabled={isLoading || (!input.trim() && attachedFiles.length === 0)} size="icon" className="h-7 w-7 rounded-lg">
                <Send className="w-3.5 h-3.5" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="bg-muted/50 rounded-2xl rounded-bl-md px-4 py-2.5">
        <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <Sparkles className="w-3.5 h-3.5 animate-pulse" />
          <span className="animate-pulse">Thinking…</span>
        </div>
      </div>
    </div>
  );
}

function useGeneralChat(onSend: () => void) {
  const messages = useGeneralChatStore((s) => s.messages);
  const input = useGeneralChatStore((s) => s.input);
  const setInput = useGeneralChatStore((s) => s.setInput);
  const isLoading = useGeneralChatStore((s) => s.isLoading);
  const selectedModel = useGeneralChatStore((s) => s.selectedModel);
  const setSelectedModel = useGeneralChatStore((s) => s.setSelectedModel);
  const generateImage = useGeneralChatStore((s) => s.generateImage);
  const setGenerateImage = useGeneralChatStore((s) => s.setGenerateImage);
  const selectedImageModel = useGeneralChatStore((s) => s.selectedImageModel);
  const setSelectedImageModel = useGeneralChatStore((s) => s.setSelectedImageModel);
  const streamingContent = useGeneralChatStore((s) => s.streamingContent);
  const isStreaming = useGeneralChatStore((s) => s.isStreaming);
  const attachedFiles = useGeneralChatStore((s) => s.attachedFiles);
  const addAttachedFile = useGeneralChatStore((s) => s.addAttachedFile);
  const removeAttachedFile = useGeneralChatStore((s) => s.removeAttachedFile);

  const handleAttachFiles = (files: FileList) => {
    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          addAttachedFile({ name: file.name, content: reader.result });
        }
      };
      reader.readAsText(file);
    });
  };
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { data: models = [], isLoading: isLoadingModels } = useQuery<ModelInfo[]>({
    queryKey: modelKeys.available(),
    queryFn: () => backendApiClient.request<ModelInfo[]>('/api/models', { method: 'GET' }),
  });

  const { data: imageModels = [], isLoading: isLoadingImageModels } = useQuery<ModelInfo[]>({
    queryKey: modelKeys.availableImages(),
    queryFn: () => backendApiClient.request<ModelInfo[]>('/api/models/image', { method: 'GET' }),
  });

  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      const preferred = models.find((m) => m.id === 'qwen/qwen3-235b-a22b-2507') || models[0];
      setSelectedModel(preferred.id);
    }
  }, [models, selectedModel, setSelectedModel]);

  useEffect(() => {
    if (imageModels.length > 0 && !selectedImageModel) {
      setSelectedImageModel(imageModels[0].id);
    }
  }, [imageModels, selectedImageModel, setSelectedImageModel]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const handleSuggestionClick = (text: string) => {
    setInput(text);
    setTimeout(() => textareaRef.current?.focus(), 0);
  };

  const selectedModelName = models.find((m) => m.id === selectedModel)?.name || 'AI';

  return {
    messages, input, setInput, isLoading, selectedModel, setSelectedModel,
    generateImage, setGenerateImage, selectedImageModel, setSelectedImageModel,
    streamingContent, isStreaming, attachedFiles, removeAttachedFile,
    handleAttachFiles, messagesEndRef, textareaRef,
    models, isLoadingModels, imageModels, isLoadingImageModels,
    handleKeyDown, handleSuggestionClick, selectedModelName,
  };
}

export function GeneralChatView({ onSend, onStop }: GeneralChatViewProps) {
  const {
    messages, input, setInput, isLoading, selectedModel, setSelectedModel,
    generateImage, setGenerateImage, selectedImageModel, setSelectedImageModel,
    streamingContent, isStreaming, attachedFiles, removeAttachedFile,
    handleAttachFiles, messagesEndRef, textareaRef,
    models, isLoadingModels, imageModels, isLoadingImageModels,
    handleKeyDown, handleSuggestionClick, selectedModelName,
  } = useGeneralChat(onSend);

  return (
    <div className="flex flex-col h-full bg-white dark:bg-background">
      {/* Messages */}
      <ScrollArea className="flex-1">
        <div className="max-w-3xl mx-auto px-4 py-6">
          {messages.length === 0 && (
            <EmptyState modelName={selectedModelName} onChipClick={handleSuggestionClick} />
          )}

          <div className="space-y-6">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}
            {isStreaming && streamingContent && (
              <MessageBubble msg={{ id: 'streaming', role: 'assistant', content: streamingContent }} />
            )}
            {isLoading && (!isStreaming || !streamingContent) && <TypingIndicator />}
          </div>
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      <ChatInputBar
        input={input}
        setInput={setInput}
        isLoading={isLoading}
        selectedModel={generateImage ? selectedImageModel : selectedModel}
        setSelectedModel={generateImage ? setSelectedImageModel : setSelectedModel}
        isLoadingModels={generateImage ? isLoadingImageModels : isLoadingModels}
        models={generateImage ? imageModels : models}
        generateImage={generateImage}
        setGenerateImage={setGenerateImage}
        attachedFiles={attachedFiles}
        onAttachFiles={handleAttachFiles}
        onRemoveFile={removeAttachedFile}
        onSend={onSend}
        onStop={onStop}
        onKeyDown={handleKeyDown}
        textareaRef={textareaRef}
      />
    </div>
  );
}
