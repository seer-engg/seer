import { useState, useCallback } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { Upload, FileJson, AlertCircle, ClipboardPaste } from 'lucide-react';

interface WorkflowImportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImport: (file: File, options: ImportOptions) => Promise<void>;
}

interface ImportOptions {
  name?: string;
  importTriggers: boolean;
}

type WorkflowExportPreview = {
  version?: string;
  workflow: { name: string; description?: string; spec: { nodes?: unknown[] } };
  triggers?: unknown[];
  metadata?: { exported_at?: string };
};

type ImportMethod = 'file' | 'paste';

// ─────────────────────────────────────────────────────────────────────────────
// Parsing utilities
// ─────────────────────────────────────────────────────────────────────────────

function validateWorkflowFile(file: File): string | null {
  if (!file.name.endsWith('.json') && !file.name.endsWith('.seer.json')) {
    return 'Please select a valid JSON file';
  }
  return null;
}

function parseWorkflowJson(json: unknown): WorkflowExportPreview {
  const data = json as Record<string, unknown>;
  if (!data.version || !data.workflow || !(data.workflow as Record<string, unknown>).spec) {
    throw new Error('Invalid workflow export format. Expected { version, workflow: { spec, ... } }');
  }
  return data as unknown as WorkflowExportPreview;
}

async function parseWorkflowFile(file: File): Promise<WorkflowExportPreview> {
  const text = await file.text();
  const json = JSON.parse(text);
  return parseWorkflowJson(json);
}

function parseWorkflowText(text: string): WorkflowExportPreview {
  const json = JSON.parse(text);
  return parseWorkflowJson(json);
}

function textToFile(text: string, name: string): File {
  const blob = new Blob([text], { type: 'application/json' });
  return new File([blob], name, { type: 'application/json' });
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom hook for import dialog state
// ─────────────────────────────────────────────────────────────────────────────

function useImportDialogState(onImport: WorkflowImportDialogProps['onImport'], onOpenChange: (open: boolean) => void) {
  const [method, setMethod] = useState<ImportMethod>('file');
  const [file, setFile] = useState<File | null>(null);
  const [pastedJson, setPastedJson] = useState('');
  const [workflowName, setWorkflowName] = useState('');
  const [importTriggers, setImportTriggers] = useState(true);
  const [isImporting, setIsImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<WorkflowExportPreview | null>(null);

  const resetState = useCallback(() => {
    setFile(null);
    setPastedJson('');
    setPreview(null);
    setWorkflowName('');
    setError(null);
  }, []);

  const handleMethodChange = useCallback((newMethod: string) => {
    setMethod(newMethod as ImportMethod);
    resetState();
  }, [resetState]);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    const validationError = validateWorkflowFile(selectedFile);
    if (validationError) {
      setError(validationError);
      return;
    }

    setFile(selectedFile);
    setError(null);

    try {
      const parsedPreview = await parseWorkflowFile(selectedFile);
      setPreview(parsedPreview);
      setWorkflowName(parsedPreview.workflow.name || '');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse JSON file');
      setFile(null);
      setPreview(null);
    }
  }, []);

  const handleParseJson = useCallback(() => {
    if (!pastedJson.trim()) {
      setError('Please paste some JSON first');
      return;
    }

    setError(null);

    try {
      const parsedPreview = parseWorkflowText(pastedJson);
      setPreview(parsedPreview);
      setWorkflowName(parsedPreview.workflow.name || '');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse JSON');
      setPreview(null);
    }
  }, [pastedJson]);

  const handlePastedJsonChange = useCallback((val: string) => {
    setPastedJson(val);
    if (preview) setPreview(null);
  }, [preview]);

  const handleImport = useCallback(async () => {
    if (!preview) return;

    const importFile = method === 'file'
      ? file
      : textToFile(pastedJson, `${preview.workflow.name || 'workflow'}.seer.json`);

    if (!importFile) return;

    setIsImporting(true);
    setError(null);

    try {
      await onImport(importFile, { name: workflowName || undefined, importTriggers });
      resetState();
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to import workflow');
    } finally {
      setIsImporting(false);
    }
  }, [preview, method, file, pastedJson, workflowName, importTriggers, onImport, resetState, onOpenChange]);

  const canImport = preview && (method === 'file' ? file : pastedJson.trim());

  return {
    method,
    file,
    pastedJson,
    workflowName,
    importTriggers,
    isImporting,
    error,
    preview,
    canImport,
    setWorkflowName,
    setImportTriggers,
    handleMethodChange,
    handleFileChange,
    handleParseJson,
    handlePastedJsonChange,
    handleImport,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function FileUploadArea({ file, onChange }: { file: File | null; onChange: (e: React.ChangeEvent<HTMLInputElement>) => void }) {
  return (
    <div>
      <Label htmlFor="file-upload">Select Workflow File</Label>
      <div className="mt-2">
        <label
          htmlFor="file-upload"
          className="flex items-center justify-center w-full px-4 py-6 border-2 border-dashed rounded-lg cursor-pointer hover:border-primary transition-colors"
        >
          {file ? (
            <div className="flex items-center gap-2">
              <FileJson className="w-5 h-5 text-primary" />
              <span className="text-sm font-medium">{file.name}</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-8 h-8 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Click to select a .seer.json file</span>
            </div>
          )}
        </label>
        <input id="file-upload" type="file" accept=".json,.seer.json" onChange={onChange} className="hidden" />
      </div>
    </div>
  );
}

interface JsonPasteAreaProps {
  value: string;
  onChange: (value: string) => void;
  onParse: () => void;
  hasPreview: boolean;
}

function JsonPasteArea({ value, onChange, onParse, hasPreview }: JsonPasteAreaProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="json-paste">Paste Workflow JSON</Label>
      <Textarea
        id="json-paste"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder='{"version": "1.0", "workflow": { "name": "...", "spec": {...} }}'
        className="font-mono text-xs min-h-[150px] resize-y"
      />
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">Paste the exported workflow JSON directly</p>
        {!hasPreview && value.trim() && (
          <Button variant="secondary" size="sm" onClick={onParse}>
            <ClipboardPaste className="w-3 h-3 mr-1" />
            Parse JSON
          </Button>
        )}
      </div>
    </div>
  );
}

interface WorkflowPreviewProps {
  preview: WorkflowExportPreview;
  workflowName: string;
  importTriggers: boolean;
  onNameChange: (name: string) => void;
  onImportTriggersChange: (checked: boolean) => void;
}

function WorkflowPreview({ preview, workflowName, importTriggers, onNameChange, onImportTriggersChange }: WorkflowPreviewProps) {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border p-4 space-y-2">
        <h4 className="text-sm font-semibold">Preview</h4>
        <div className="text-sm space-y-1">
          <p><span className="text-muted-foreground">Name:</span> {preview.workflow.name}</p>
          <p><span className="text-muted-foreground">Nodes:</span> {preview.workflow.spec.nodes?.length || 0}</p>
          <p><span className="text-muted-foreground">Triggers:</span> {preview.triggers?.length || 0}</p>
          {preview.metadata?.exported_at && (
            <p className="text-xs text-muted-foreground">
              Exported: {new Date(preview.metadata.exported_at).toLocaleString()}
            </p>
          )}
        </div>
      </div>

      <div>
        <Label htmlFor="workflow-name">Workflow Name (optional)</Label>
        <Input
          id="workflow-name"
          value={workflowName}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder={preview.workflow.name}
          className="mt-2"
        />
        <p className="text-xs text-muted-foreground mt-1">
          Leave empty to use original name. Conflicts will be auto-renamed.
        </p>
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox id="import-triggers" checked={importTriggers} onCheckedChange={(checked) => onImportTriggersChange(checked === true)} />
        <label htmlFor="import-triggers" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          Import triggers ({preview.triggers?.length || 0})
        </label>
      </div>

      {preview.triggers && preview.triggers.length > 0 && importTriggers && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="text-xs">
            Imported triggers will be disabled. You'll need to configure OAuth connections and enable them manually.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export function WorkflowImportDialog({ open, onOpenChange, onImport }: WorkflowImportDialogProps) {
  const state = useImportDialogState(onImport, onOpenChange);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Import Workflow</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <Tabs value={state.method} onValueChange={state.handleMethodChange}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="file" className="gap-2">
                <Upload className="w-4 h-4" />
                Upload File
              </TabsTrigger>
              <TabsTrigger value="paste" className="gap-2">
                <ClipboardPaste className="w-4 h-4" />
                Paste JSON
              </TabsTrigger>
            </TabsList>

            <TabsContent value="file" className="mt-4">
              <FileUploadArea file={state.file} onChange={state.handleFileChange} />
            </TabsContent>

            <TabsContent value="paste" className="mt-4">
              <JsonPasteArea
                value={state.pastedJson}
                onChange={state.handlePastedJsonChange}
                onParse={state.handleParseJson}
                hasPreview={!!state.preview}
              />
            </TabsContent>
          </Tabs>

          {state.preview && (
            <WorkflowPreview
              preview={state.preview}
              workflowName={state.workflowName}
              importTriggers={state.importTriggers}
              onNameChange={state.setWorkflowName}
              onImportTriggersChange={state.setImportTriggers}
            />
          )}

          {state.error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{state.error}</AlertDescription>
            </Alert>
          )}

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button onClick={state.handleImport} disabled={!state.canImport || state.isImporting}>
              {state.isImporting ? 'Importing...' : 'Import Workflow'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
