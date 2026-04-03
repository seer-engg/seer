import { useState, useEffect, useCallback, useMemo } from 'react';
import { Link2, HardDrive, X, File } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AutocompleteInput } from '../AutocompleteInput';
import { FilePickerDialog } from '../FilePickerDialog';
import { FileTypeBadge } from '@/components/data-storage/FileTypeBadge';
import { filesApi, type UserFile } from '@/lib/files-api';
import { fileKeys } from '@/lib/query-keys';
import { useTemplateAutocomplete } from '@/hooks/useTemplateAutocomplete';
import {
  type FileInputValue,
  isStaticFileRef,
  parseExpressionToWorkflowRef,
  createStaticFileRef,
} from '../../helpers/fileInputSchema';
import type { FileVariable } from '../../helpers/discoverFileVariables';

// ============================================================================
// Types
// ============================================================================

type FileInputMode = 'expression' | 'static';

export interface FileInputFieldProps {
  id: string;
  value: FileInputValue;
  onChange: (value: FileInputValue) => void;
  placeholder?: string;
  showError: boolean;
  availableFileVariables?: FileVariable[];
  mimeTypeFilter?: string;
}

// ============================================================================
// Helper Components
// ============================================================================

function SelectedFileDisplay({ file, onClear }: { file: UserFile; onClear: () => void }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-md border">
      <File className="h-4 w-4 text-muted-foreground flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{file.filename}</p>
        <p className="text-xs text-muted-foreground">{file.size_human}</p>
      </div>
      <FileTypeBadge mimeType={file.mime_type} className="flex-shrink-0" />
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onClear}
        className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive flex-shrink-0"
      >
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

function FileNotFoundDisplay({ fileId, onClear }: { fileId: string; onClear: () => void }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-destructive/10 rounded-md border border-destructive/30">
      <File className="h-4 w-4 text-destructive flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-destructive">File not found</p>
        <p className="text-xs text-muted-foreground truncate">ID: {fileId}</p>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onClear}
        className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive flex-shrink-0"
      >
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-md border">
      <div className="h-4 w-4 bg-muted rounded animate-pulse" />
      <div className="flex-1 space-y-1">
        <div className="h-4 bg-muted rounded w-1/3 animate-pulse" />
        <div className="h-3 bg-muted rounded w-1/4 animate-pulse" />
      </div>
    </div>
  );
}

function ModeToggle({ mode, onModeChange }: { mode: FileInputMode; onModeChange: (mode: string) => void }) {
  return (
    <Tabs value={mode} onValueChange={onModeChange}>
      <TabsList className="h-8 w-full grid grid-cols-2">
        <TabsTrigger value="expression" className="text-xs h-6 gap-1.5">
          <Link2 className="h-3 w-3" />
          From workflow
        </TabsTrigger>
        <TabsTrigger value="static" className="text-xs h-6 gap-1.5">
          <HardDrive className="h-3 w-3" />
          From storage
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );
}

// ============================================================================
// Custom Hook
// ============================================================================

function useFileInputState(value: FileInputValue, onChange: (v: FileInputValue) => void) {
  const getInitialMode = useCallback((): FileInputMode => {
    if (isStaticFileRef(value)) return 'static';
    return 'expression'; // string or null → expression mode
  }, [value]);

  const [mode, setMode] = useState<FileInputMode>(getInitialMode);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [expressionValue, setExpressionValue] = useState<string>(() => {
    // Template strings are stored directly now
    if (typeof value === 'string') return value;
    return '';
  });

  useEffect(() => {
    if (typeof value === 'string') {
      setExpressionValue(value);
      setMode('expression');
    } else if (isStaticFileRef(value)) {
      setMode('static');
    }
  }, [value]);

  const handleExpressionChange = useCallback((newExpression: string) => {
    setExpressionValue(newExpression);
    if (!newExpression.trim()) {
      onChange(null);
      return;
    }
    // Validate format, but store the STRING (not parsed object)
    const isValid = parseExpressionToWorkflowRef(newExpression);
    if (isValid) {
      onChange(newExpression); // Store template string directly
    }
  }, [onChange]);

  const handleModeChange = useCallback((newMode: string) => {
    setMode(newMode as FileInputMode);
    onChange(null);
    if (newMode === 'expression') setExpressionValue('');
  }, [onChange]);

  const handleFileSelect = useCallback((file: UserFile) => {
    onChange(createStaticFileRef(file.file_id));
  }, [onChange]);

  const handleClear = useCallback(() => {
    onChange(null);
    if (mode === 'expression') setExpressionValue('');
  }, [onChange, mode]);

  return {
    mode, pickerOpen, setPickerOpen, expressionValue,
    handleExpressionChange, handleModeChange, handleFileSelect, handleClear,
  };
}

// ============================================================================
// Main Component
// ============================================================================

export function FileInputField({
  id, value, onChange, placeholder,
  showError, availableFileVariables = [], mimeTypeFilter,
}: FileInputFieldProps) {
  const {
    mode, pickerOpen, setPickerOpen, expressionValue,
    handleExpressionChange, handleModeChange, handleFileSelect, handleClear,
  } = useFileInputState(value, onChange);

  // Create file-specific autocomplete using available file variable paths
  const fileVariablePaths = useMemo(
    () => availableFileVariables.map((v) => v.path),
    [availableFileVariables],
  );
  const fileAutocomplete = useTemplateAutocomplete(fileVariablePaths);

  const staticFileId = isStaticFileRef(value) ? value.file_id : null;
  const { data: fileInfo, isLoading } = useQuery({
    queryKey: fileKeys.detail(staticFileId),
    queryFn: () => filesApi.getFile(staticFileId!),
    enabled: !!staticFileId,
    retry: 1,
  });

  return (
    <div className="space-y-2">
      <ModeToggle mode={mode} onModeChange={handleModeChange} />

      {mode === 'expression' ? (
        <div>
          <AutocompleteInput
            id={id}
            value={expressionValue}
            onChange={handleExpressionChange}
            placeholder={placeholder || '${nodeId.file}'}
            templateAutocomplete={fileAutocomplete}
            className={cn('text-xs', showError && 'border-destructive')}
          />
          {availableFileVariables.length === 0 && (
            <p className="text-xs text-muted-foreground mt-1.5">No file outputs found from upstream nodes</p>
          )}
        </div>
      ) : (
        <div>
          {isStaticFileRef(value) ? (
            isLoading ? <LoadingSkeleton /> :
            fileInfo ? <SelectedFileDisplay file={fileInfo} onClear={handleClear} /> :
            <FileNotFoundDisplay fileId={value.file_id} onClear={handleClear} />
          ) : (
            <Button
              type="button"
              variant="outline"
              className={cn('w-full justify-start text-muted-foreground', showError && 'border-destructive')}
              onClick={() => setPickerOpen(true)}
            >
              <File className="h-4 w-4 mr-2" />
              Browse files...
            </Button>
          )}
        </div>
      )}

      <FilePickerDialog
        open={pickerOpen}
        onOpenChange={setPickerOpen}
        onSelect={handleFileSelect}
        selectedFileId={staticFileId || undefined}
        mimeTypeFilter={mimeTypeFilter}
      />
    </div>
  );
}
