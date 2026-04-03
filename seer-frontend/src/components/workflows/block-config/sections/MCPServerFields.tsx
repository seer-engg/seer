import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Loader2, Save, Plus, X } from 'lucide-react';
import { cn } from '@/lib/utils';

import { McpBlockConfig, BlockSectionProps } from '../types';

type MCPBlockSectionProps = BlockSectionProps<McpBlockConfig>;

// ---------------------------------------------------------------------------
// Shared inline editors
// ---------------------------------------------------------------------------

export function KeyValueEditor({
  label,
  description,
  entries,
  onChange,
  keyPlaceholder = 'Key',
  valuePlaceholder = 'Value',
}: {
  label: string;
  description?: string;
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
    while (key in entries) {
      i++;
      key = `key_${i}`;
    }
    onChange({ ...entries, [key]: '' });
  };

  const removePair = (key: string) => {
    const copy = { ...entries };
    delete copy[key];
    onChange(copy);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">{label}</label>
        <Button type="button" variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={addPair}>
          <Plus className="h-3 w-3 mr-1" /> Add
        </Button>
      </div>
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
      {pairs.length === 0 && (
        <p className="text-xs text-muted-foreground italic">No entries. Click Add to add one.</p>
      )}
      {pairs.map(([key, value], idx) => (
        <div key={idx} className="flex gap-1.5 items-center">
          <Input
            className="h-7 text-xs flex-1"
            placeholder={keyPlaceholder}
            value={key}
            onChange={e => updateKey(key, e.target.value)}
          />
          <Input
            className="h-7 text-xs flex-1"
            placeholder={valuePlaceholder}
            value={value}
            onChange={e => updateValue(key, e.target.value)}
          />
          <Button type="button" variant="ghost" size="icon" className="h-7 w-7 flex-shrink-0" onClick={() => removePair(key)}>
            <X className="h-3 w-3" />
          </Button>
        </div>
      ))}
    </div>
  );
}

export function StringListEditor({
  label,
  description,
  items,
  onChange,
  placeholder = 'Value',
}: {
  label: string;
  description?: string;
  items: string[];
  onChange: (items: string[]) => void;
  placeholder?: string;
}) {
  const addItem = () => onChange([...items, '']);
  const updateItem = (idx: number, value: string) => {
    const copy = [...items];
    copy[idx] = value;
    onChange(copy);
  };
  const removeItem = (idx: number) => onChange(items.filter((_, i) => i !== idx));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">{label}</label>
        <Button type="button" variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={addItem}>
          <Plus className="h-3 w-3 mr-1" /> Add
        </Button>
      </div>
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
      {items.length === 0 && (
        <p className="text-xs text-muted-foreground italic">No arguments. Click Add to add one.</p>
      )}
      {items.map((item, idx) => (
        <div key={idx} className="flex gap-1.5 items-center">
          <Input
            className="h-7 text-xs flex-1"
            placeholder={placeholder}
            value={item}
            onChange={e => updateItem(idx, e.target.value)}
          />
          <Button type="button" variant="ghost" size="icon" className="h-7 w-7 flex-shrink-0" onClick={() => removeItem(idx)}>
            <X className="h-3 w-3" />
          </Button>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// HTTP fields sub-section
// ---------------------------------------------------------------------------

function HttpFields({
  config,
  setConfig,
  validationErrors,
}: {
  config: McpBlockConfig;
  setConfig: MCPBlockSectionProps['setConfig'];
  validationErrors: Record<string, string>;
}) {
  return (
    <>
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">Server URL</label>
        <Input
          className="h-8 text-xs"
          placeholder="http://localhost:8080/mcp"
          value={config.server || ''}
          onChange={e => setConfig(prev => ({ ...prev, server: e.target.value }))}
        />
        {validationErrors['server'] && (
          <p className="text-xs text-destructive">{validationErrors['server']}</p>
        )}
      </div>
      <KeyValueEditor
        label="Headers"
        description="HTTP headers sent with every request (e.g., Authorization)"
        entries={config.http_headers || {}}
        onChange={headers => setConfig(prev => ({
          ...prev,
          http_headers: headers,
          auth: Object.keys(headers).length > 0 ? { headers } : undefined,
        }))}
        keyPlaceholder="Header name"
        valuePlaceholder="Header value"
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// Stdio fields sub-section
// ---------------------------------------------------------------------------

function StdioFields({
  config,
  setConfig,
}: {
  config: McpBlockConfig;
  setConfig: MCPBlockSectionProps['setConfig'];
}) {
  return (
    <>
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">Command</label>
        <Input
          className="h-8 text-xs"
          placeholder="uvx, npx, node, python..."
          value={config.stdio_command || ''}
          onChange={e => setConfig(prev => ({ ...prev, stdio_command: e.target.value }))}
        />
        <p className="text-xs text-muted-foreground">The executable to run the MCP server</p>
      </div>
      <StringListEditor
        label="Arguments"
        description="Command-line arguments passed to the command"
        items={config.stdio_args || []}
        onChange={args => setConfig(prev => ({ ...prev, stdio_args: args }))}
        placeholder="e.g., postgres-mcp or --access-mode=unrestricted"
      />
      <KeyValueEditor
        label="Environment Variables"
        description="Environment variables passed to the process"
        entries={config.stdio_env || {}}
        onChange={env => setConfig(prev => ({
          ...prev,
          stdio_env: env,
          auth: Object.keys(env).length > 0 ? { env } : undefined,
        }))}
        keyPlaceholder="Variable name"
        valuePlaceholder="Value"
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// Type-aware server fields (main export)
// ---------------------------------------------------------------------------

export function McpServerFields({
  config,
  setConfig,
  validationErrors,
  isFetching,
  fetchError,
  onFetchTools,
  onSaveServer,
  isSaving,
}: {
  config: McpBlockConfig;
  setConfig: MCPBlockSectionProps['setConfig'];
  validationErrors: Record<string, string>;
  isFetching: boolean;
  fetchError: string | null;
  onFetchTools: () => void;
  onSaveServer: () => void;
  isSaving: boolean;
}) {
  const serverType = config.server_type || 'http';

  const hasServerConfig = serverType === 'stdio'
    ? !!(config.stdio_command || '').trim()
    : !!(config.server || '').trim();

  return (
    <>
      {/* Server type toggle */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">Server Type</label>
        <div className="flex gap-1 p-1 bg-muted rounded-md">
          <button
            type="button"
            className={cn(
              'flex-1 text-xs py-1 px-2 rounded-sm transition-colors',
              serverType === 'http'
                ? 'bg-background shadow-sm font-medium'
                : 'text-muted-foreground hover:text-foreground',
            )}
            onClick={() => setConfig(prev => ({
              ...prev,
              server_type: 'http',
              stdio_command: undefined, stdio_args: undefined, stdio_env: undefined,
              fetchedTools: undefined, tool: '', inputs: {},
            }))}
          >
            HTTP
          </button>
          <button
            type="button"
            className={cn(
              'flex-1 text-xs py-1 px-2 rounded-sm transition-colors',
              serverType === 'stdio'
                ? 'bg-background shadow-sm font-medium'
                : 'text-muted-foreground hover:text-foreground',
            )}
            onClick={() => setConfig(prev => ({
              ...prev,
              server_type: 'stdio',
              server: '', http_headers: undefined,
              fetchedTools: undefined, tool: '', inputs: {},
            }))}
          >
            Stdio
          </button>
        </div>
      </div>

      {serverType === 'http' ? (
        <HttpFields config={config} setConfig={setConfig} validationErrors={validationErrors} />
      ) : (
        <StdioFields config={config} setConfig={setConfig} />
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        <Button
          variant="outline" size="sm" className="flex-1 text-xs"
          disabled={!hasServerConfig || isFetching} onClick={onFetchTools}
        >
          {isFetching ? (
            <><Loader2 className="mr-2 h-3 w-3 animate-spin" />Fetching tools...</>
          ) : (
            'Fetch Tools'
          )}
        </Button>
        <Button
          variant="outline" size="sm" className="text-xs"
          disabled={!hasServerConfig || isSaving} onClick={onSaveServer} title="Save this server for reuse"
        >
          {isSaving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
        </Button>
      </div>
      {fetchError && <p className="text-xs text-destructive">{fetchError}</p>}
    </>
  );
}

