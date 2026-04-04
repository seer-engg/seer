import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Globe, Plus, Trash2, Loader2, CheckCircle2, XCircle, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { backendApiClient, BackendAPIError } from '@/lib/api-client';

interface SavedMcpServer {
  id: number;
  name: string;
  server_url: string;
  server_type: string;
  has_auth: boolean;
  created_at?: string | null;
  updated_at?: string | null;
}

interface TestResult {
  id: number;
  ok: boolean;
  message: string;
}

function KVEditor({
  entries,
  onChange,
  keyPlaceholder = 'Key',
  valuePlaceholder = 'Value',
}: {
  entries: Record<string, string>;
  onChange: (entries: Record<string, string>) => void;
  keyPlaceholder?: string;
  valuePlaceholder?: string;
}) {
  const pairs = Object.entries(entries);

  const updateKey = (oldKey: string, newKey: string) => {
    const result: Record<string, string> = {};
    for (const [k, v] of Object.entries(entries)) {
      result[k === oldKey ? newKey : k] = v;
    }
    onChange(result);
  };

  const updateValue = (key: string, value: string) => {
    onChange({ ...entries, [key]: value });
  };

  const addPair = () => {
    let key = '';
    let i = 0;
    while (key in entries) { i++; key = `key_${i}`; }
    onChange({ ...entries, [key]: '' });
  };

  const removePair = (key: string) => {
    const copy = { ...entries };
    delete copy[key];
    onChange(copy);
  };

  return (
    <div className="space-y-2">
      {pairs.map(([key, value], idx) => (
        <div key={idx} className="flex gap-1.5 items-center">
          <Input className="h-8 text-sm flex-1" placeholder={keyPlaceholder} value={key} onChange={e => updateKey(key, e.target.value)} />
          <Input className="h-8 text-sm flex-1" placeholder={valuePlaceholder} value={value} onChange={e => updateValue(key, e.target.value)} />
          <Button type="button" variant="ghost" size="icon" className="h-8 w-8 flex-shrink-0" onClick={() => removePair(key)}>
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" size="sm" className="text-xs" onClick={addPair}>
        <Plus className="h-3 w-3 mr-1" /> Add
      </Button>
    </div>
  );
}

function ArgsEditor({
  items,
  onChange,
}: {
  items: string[];
  onChange: (items: string[]) => void;
}) {
  const addItem = () => onChange([...items, '']);
  const updateItem = (idx: number, value: string) => { const c = [...items]; c[idx] = value; onChange(c); };
  const removeItem = (idx: number) => onChange(items.filter((_, i) => i !== idx));

  return (
    <div className="space-y-2">
      {items.map((item, idx) => (
        <div key={idx} className="flex gap-1.5 items-center">
          <Input className="h-8 text-sm flex-1" placeholder="e.g., postgres-mcp" value={item} onChange={e => updateItem(idx, e.target.value)} />
          <Button type="button" variant="ghost" size="icon" className="h-8 w-8 flex-shrink-0" onClick={() => removeItem(idx)}>
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" size="sm" className="text-xs" onClick={addItem}>
        <Plus className="h-3 w-3 mr-1" /> Add Argument
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// API helpers (outside hook to reduce line count)
// ---------------------------------------------------------------------------

function buildFormPayload(form: {
  name: string; serverType: 'http' | 'stdio';
  url: string; headers: Record<string, string>;
  command: string; args: string[]; env: Record<string, string>;
}): { name: string; server_url: string; server_type: string; auth?: Record<string, unknown> } | null {
  if (!form.name.trim()) return null;
  if (form.serverType === 'stdio') {
    if (!form.command.trim()) return null;
    const parts = [form.command.trim(), ...form.args.filter(Boolean)];
    return {
      name: form.name.trim(), server_url: parts.join(' '), server_type: 'stdio',
      auth: Object.keys(form.env).length > 0 ? { env: form.env } : undefined,
    };
  }
  if (!form.url.trim()) return null;
  return {
    name: form.name.trim(), server_url: form.url.trim(), server_type: 'http',
    auth: Object.keys(form.headers).length > 0 ? { headers: form.headers } : undefined,
  };
}

async function testMcpServer(server: SavedMcpServer): Promise<TestResult> {
  try {
    const response = await backendApiClient.request<{ status: string; connected: boolean; tool_count?: number; error?: string }>(
      '/api/integrations/mcp/servers/test',
      { method: 'POST', body: { name: server.name, server_url: server.server_url, server_type: server.server_type } },
    );
    return {
      id: server.id,
      ok: response.connected,
      message: response.connected
        ? `Connected - ${response.tool_count ?? 0} tools available`
        : response.error || 'Connection failed',
    };
  } catch (err) {
    return {
      id: server.id, ok: false,
      message: err instanceof BackendAPIError ? err.message : 'Connection test failed',
    };
  }
}

// ---------------------------------------------------------------------------
// Hook: encapsulates all server CRUD + form state
// ---------------------------------------------------------------------------

function useMcpServers() {
  const [servers, setServers] = useState<SavedMcpServer[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [testingId, setTestingId] = useState<number | null>(null);
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [formName, setFormName] = useState('');
  const [formServerType, setFormServerType] = useState<'http' | 'stdio'>('http');
  const [formUrl, setFormUrl] = useState('');
  const [formHeaders, setFormHeaders] = useState<Record<string, string>>({});
  const [formCommand, setFormCommand] = useState('');
  const [formArgs, setFormArgs] = useState<string[]>([]);
  const [formEnv, setFormEnv] = useState<Record<string, string>>({});

  const resetForm = useCallback(() => {
    setFormName(''); setFormServerType('http'); setFormUrl(''); setFormHeaders({});
    setFormCommand(''); setFormArgs([]); setFormEnv({}); setError(null);
  }, []);

  const fetchServers = useCallback(async () => {
    try {
      const response = await backendApiClient.request<{ items: SavedMcpServer[] }>('/api/integrations/mcp/servers', { method: 'GET' });
      setServers(response.items);
    } catch { setError('Failed to load MCP servers'); }
    finally { setIsLoading(false); }
  }, []);

  useEffect(() => { fetchServers(); }, [fetchServers]);

  const handleSave = useCallback(async () => {
    const payload = buildFormPayload({
      name: formName, serverType: formServerType,
      url: formUrl, headers: formHeaders, command: formCommand, args: formArgs, env: formEnv,
    });
    if (!payload) return;
    setIsSaving(true); setError(null);
    try {
      await backendApiClient.request('/api/integrations/mcp/servers/bind', { method: 'POST', body: payload });
      resetForm(); setIsDialogOpen(false); await fetchServers();
    } catch (err) {
      setError(err instanceof BackendAPIError ? err.message : 'Failed to save server');
    } finally { setIsSaving(false); }
  }, [formName, formServerType, formCommand, formArgs, formEnv, formUrl, formHeaders, resetForm, fetchServers]);

  const handleDelete = useCallback(async (id: number) => {
    setDeletingId(id);
    try {
      await backendApiClient.request(`/api/integrations/resources/${id}`, { method: 'DELETE' });
      setServers(prev => prev.filter(s => s.id !== id));
    } catch (err) {
      setError(err instanceof BackendAPIError ? err.message : 'Failed to delete server');
    } finally { setDeletingId(null); }
  }, []);

  const handleTest = useCallback(async (server: SavedMcpServer) => {
    setTestingId(server.id); setTestResult(null);
    const result = await testMcpServer(server);
    setTestResult(result); setTestingId(null);
  }, []);

  const canSave = formName.trim() && (formServerType === 'http' ? formUrl.trim() : formCommand.trim());

  return {
    servers, isLoading, isDialogOpen, setIsDialogOpen, isSaving, deletingId, testingId, testResult, error,
    formName, setFormName, formServerType, setFormServerType,
    formUrl, setFormUrl, formHeaders, setFormHeaders,
    formCommand, setFormCommand, formArgs, setFormArgs, formEnv, setFormEnv,
    resetForm, handleSave, handleDelete, handleTest, canSave,
  };
}

// ---------------------------------------------------------------------------
// Dialog form for adding a server
// ---------------------------------------------------------------------------

function AddServerDialogContent({
  formName, setFormName, formServerType, setFormServerType,
  formUrl, setFormUrl, formHeaders, setFormHeaders,
  formCommand, setFormCommand, formArgs, setFormArgs, formEnv, setFormEnv,
  error, isSaving, canSave, onSave, onCancel,
}: {
  formName: string; setFormName: (v: string) => void;
  formServerType: 'http' | 'stdio'; setFormServerType: (v: 'http' | 'stdio') => void;
  formUrl: string; setFormUrl: (v: string) => void;
  formHeaders: Record<string, string>; setFormHeaders: (v: Record<string, string>) => void;
  formCommand: string; setFormCommand: (v: string) => void;
  formArgs: string[]; setFormArgs: (v: string[]) => void;
  formEnv: Record<string, string>; setFormEnv: (v: Record<string, string>) => void;
  error: string | null; isSaving: boolean; canSave: string | boolean;
  onSave: () => void; onCancel: () => void;
}) {
  return (
    <DialogContent className="max-w-lg">
      <DialogHeader>
        <DialogTitle>Add MCP Server</DialogTitle>
        <DialogDescription>
          Save an MCP server configuration for reuse across workflows.
        </DialogDescription>
      </DialogHeader>
      <div className="space-y-4 py-4">
        <div className="space-y-2">
          <Label htmlFor="name">Name</Label>
          <Input id="name" placeholder="My MCP Server" value={formName} onChange={e => setFormName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label>Server Type</Label>
          <div className="flex gap-1 p-1 bg-muted rounded-md">
            <button
              type="button"
              className={cn(
                'flex-1 text-sm py-1.5 px-3 rounded-sm transition-colors',
                formServerType === 'http' ? 'bg-background shadow-sm font-medium' : 'text-muted-foreground hover:text-foreground',
              )}
              onClick={() => setFormServerType('http')}
            >
              HTTP
            </button>
            <button
              type="button"
              className={cn(
                'flex-1 text-sm py-1.5 px-3 rounded-sm transition-colors',
                formServerType === 'stdio' ? 'bg-background shadow-sm font-medium' : 'text-muted-foreground hover:text-foreground',
              )}
              onClick={() => setFormServerType('stdio')}
            >
              Stdio
            </button>
          </div>
        </div>
        {formServerType === 'http' ? (
          <>
            <div className="space-y-2">
              <Label htmlFor="url">Server URL</Label>
              <Input id="url" placeholder="http://localhost:8080/mcp" value={formUrl} onChange={e => setFormUrl(e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Headers (optional)</Label>
              <KVEditor entries={formHeaders} onChange={setFormHeaders} keyPlaceholder="Header name" valuePlaceholder="Header value" />
            </div>
          </>
        ) : (
          <>
            <div className="space-y-2">
              <Label htmlFor="command">Command</Label>
              <Input id="command" placeholder="uvx, npx, node, python..." value={formCommand} onChange={e => setFormCommand(e.target.value)} />
              <p className="text-xs text-muted-foreground">The executable to run the MCP server</p>
            </div>
            <div className="space-y-2">
              <Label>Arguments</Label>
              <ArgsEditor items={formArgs} onChange={setFormArgs} />
            </div>
            <div className="space-y-2">
              <Label>Environment Variables (optional)</Label>
              <KVEditor entries={formEnv} onChange={setFormEnv} keyPlaceholder="Variable name" valuePlaceholder="Value" />
              <p className="text-xs text-muted-foreground">Sensitive values are stored encrypted.</p>
            </div>
          </>
        )}
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}
      <DialogFooter>
        <Button variant="outline" onClick={onCancel}>Cancel</Button>
        <Button onClick={onSave} disabled={!canSave || isSaving}>
          {isSaving ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
          Save Server
        </Button>
      </DialogFooter>
    </DialogContent>
  );
}

// ---------------------------------------------------------------------------
// Single server row in the list
// ---------------------------------------------------------------------------

function ServerListItem({
  server, testResult, testingId, deletingId, onTest, onDelete,
}: {
  server: SavedMcpServer;
  testResult: TestResult | null;
  testingId: number | null;
  deletingId: number | null;
  onTest: (server: SavedMcpServer) => void;
  onDelete: (id: number) => void;
}) {
  return (
    <div className="flex items-center justify-between p-3 rounded-lg border bg-card">
      <div className="flex items-center gap-3 min-w-0">
        <Globe className="h-4 w-4 text-cyan-500 flex-shrink-0" />
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">{server.name}</p>
          <p className="text-xs text-muted-foreground truncate">
            {server.server_url} ({server.server_type})
            {server.has_auth && ' - Auth configured'}
          </p>
          {testResult?.id === server.id && (
            <p className={`text-xs mt-1 ${testResult.ok ? 'text-green-600' : 'text-destructive'}`}>
              {testResult.ok ? <CheckCircle2 className="inline h-3 w-3 mr-1" /> : <XCircle className="inline h-3 w-3 mr-1" />}
              {testResult.message}
            </p>
          )}
        </div>
      </div>
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="sm" className="text-xs" onClick={() => onTest(server)} disabled={testingId === server.id}>
          {testingId === server.id ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Test'}
        </Button>
        <Button
          variant="ghost" size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-destructive"
          onClick={() => onDelete(server.id)} disabled={deletingId === server.id}
        >
          {deletingId === server.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main card (orchestrator only)
// ---------------------------------------------------------------------------

export function McpServersCard() {
  const state = useMcpServers();

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-base">MCP Servers</CardTitle>
          <CardDescription>Saved MCP server configurations for use in workflow nodes.</CardDescription>
        </div>
        <Dialog open={state.isDialogOpen} onOpenChange={open => { state.setIsDialogOpen(open); if (!open) state.resetForm(); }}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm"><Plus className="h-4 w-4 mr-1" />Add Server</Button>
          </DialogTrigger>
          <AddServerDialogContent
            formName={state.formName} setFormName={state.setFormName}
            formServerType={state.formServerType} setFormServerType={state.setFormServerType}
            formUrl={state.formUrl} setFormUrl={state.setFormUrl}
            formHeaders={state.formHeaders} setFormHeaders={state.setFormHeaders}
            formCommand={state.formCommand} setFormCommand={state.setFormCommand}
            formArgs={state.formArgs} setFormArgs={state.setFormArgs}
            formEnv={state.formEnv} setFormEnv={state.setFormEnv}
            error={state.error} isSaving={state.isSaving} canSave={state.canSave}
            onSave={state.handleSave} onCancel={() => state.setIsDialogOpen(false)}
          />
        </Dialog>
      </CardHeader>
      <CardContent>
        {state.isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : state.servers.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-6">No MCP servers saved yet. Add one to get started.</p>
        ) : (
          <div className="space-y-3">
            {state.servers.map(server => (
              <ServerListItem
                key={server.id} server={server}
                testResult={state.testResult} testingId={state.testingId} deletingId={state.deletingId}
                onTest={state.handleTest} onDelete={state.handleDelete}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
