import { useState, useCallback, useMemo, useEffect } from 'react';

import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Check, ChevronsUpDown, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

import { McpBlockConfig, McpToolInfo, SavedMcpServer, BlockSectionProps } from '../types';
import { DynamicFormField } from '../widgets/DynamicFormField';
import { backendApiClient, BackendAPIError } from '@/lib/api-client';
import { McpServerFields } from './MCPServerFields';
import { buildFlatServerConfig } from './mcp-utils';

type MCPBlockSectionProps = BlockSectionProps<McpBlockConfig>;

type ServerMode = 'saved' | 'manual';

// ---------------------------------------------------------------------------
// Tool selector
// ---------------------------------------------------------------------------

function McpToolSelector({
  tools,
  selectedTool,
  onChange,
}: {
  tools: McpToolInfo[];
  selectedTool?: string;
  onChange: (toolName: string) => void;
}) {
  const [open, setOpen] = useState(false);

  if (tools.length === 0) return null;

  return (
    <div className="space-y-2">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal text-xs"
          >
            {selectedTool
              ? tools.find(t => t.name === selectedTool)?.name ?? selectedTool
              : 'Select a tool...'}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search tools..." />
            <CommandList>
              <CommandEmpty>No tool found.</CommandEmpty>
              <CommandGroup>
                {tools.map(tool => (
                  <CommandItem
                    key={tool.name}
                    value={tool.name}
                    onSelect={value => {
                      onChange(value);
                      setOpen(false);
                    }}
                    className="flex flex-col items-start gap-1 py-2"
                  >
                    <div className="flex items-center gap-2 w-full">
                      <Check
                        className={cn(
                          'h-4 w-4 flex-shrink-0',
                          selectedTool === tool.name ? 'opacity-100' : 'opacity-0',
                        )}
                      />
                      <span className="font-medium">{tool.name}</span>
                    </div>
                    {tool.description && (
                      <span className="text-xs text-muted-foreground ml-6 line-clamp-2">
                        {tool.description}
                      </span>
                    )}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Schema-driven tool inputs
// ---------------------------------------------------------------------------

function SchemaInputs({
  selectedTool,
  tools,
  inputs,
  updateInputs,
  templateAutocomplete,
}: {
  selectedTool: string;
  tools: McpToolInfo[];
  inputs: Record<string, unknown>;
  updateInputs: (updater: (prev: Record<string, unknown>) => Record<string, unknown>) => void;
  templateAutocomplete: MCPBlockSectionProps['templateAutocomplete'];
}) {
  const tool = tools.find(t => t.name === selectedTool);
  const schema = tool?.input_schema as Record<string, unknown> | undefined;
  const properties = (schema?.properties as Record<string, Record<string, unknown>>) ?? {};
  const required = (schema?.required as string[]) ?? [];
  const hasProperties = Object.keys(properties).length > 0;

  if (!hasProperties) {
    return (
      <DynamicFormField
        name="inputs"
        label="Inputs"
        description="Arguments passed to the tool (JSON). Use ${variable} to reference upstream outputs."
        value={inputs}
        onChange={value => updateInputs(() => value as Record<string, unknown>)}
        def={{ type: 'object' }}
        templateAutocomplete={templateAutocomplete}
        rows={4}
      />
    );
  }

  return (
    <div className="space-y-3">
      {Object.entries(properties).map(([paramName, paramDef]) => (
        <DynamicFormField
          key={paramName}
          name={paramName}
          label={paramName}
          description={paramDef.description as string | undefined}
          required={required.includes(paramName)}
          value={inputs[paramName]}
          onChange={val => updateInputs(prev => ({ ...prev, [paramName]: val }))}
          def={paramDef}
          templateAutocomplete={templateAutocomplete}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Saved server selector
// ---------------------------------------------------------------------------

function SavedServerSelector({
  savedServers,
  selectedId,
  onChange,
  isLoading,
}: {
  savedServers: SavedMcpServer[];
  selectedId?: number | null;
  onChange: (server: SavedMcpServer | null) => void;
  isLoading: boolean;
}) {
  const [open, setOpen] = useState(false);
  const selected = savedServers.find(s => s.id === selectedId);

  return (
    <div className="space-y-2">
      <label className="text-xs font-medium text-muted-foreground">Saved Server</label>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal text-xs"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                Loading servers...
              </>
            ) : selected ? (
              <span className="truncate">{selected.name} ({selected.server_url})</span>
            ) : (
              'Select a saved server...'
            )}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search servers..." />
            <CommandList>
              <CommandEmpty>No saved servers found.</CommandEmpty>
              <CommandGroup>
                {savedServers.map(server => (
                  <CommandItem
                    key={server.id}
                    value={`${server.name} ${server.server_url}`}
                    onSelect={() => {
                      onChange(server.id === selectedId ? null : server);
                      setOpen(false);
                    }}
                    className="flex flex-col items-start gap-1 py-2"
                  >
                    <div className="flex items-center gap-2 w-full">
                      <Check
                        className={cn(
                          'h-4 w-4 flex-shrink-0',
                          selectedId === server.id ? 'opacity-100' : 'opacity-0',
                        )}
                      />
                      <span className="font-medium">{server.name}</span>
                    </div>
                    <span className="text-xs text-muted-foreground ml-6 truncate max-w-full">
                      {server.server_url} ({server.server_type})
                      {server.has_auth && ' - Auth configured'}
                    </span>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}

// ---------------------------------------------------------------------------
// API helper (outside hook to reduce function line count)
// ---------------------------------------------------------------------------

async function saveServerToBackend(
  config: McpBlockConfig,
): Promise<{ resource: { id: number }; servers: SavedMcpServer[] }> {
  const flat = buildFlatServerConfig(config);
  if (!flat.server) throw new Error('No server configured');
  const name = flat.server.length > 50 ? flat.server.substring(0, 50) + '...' : flat.server;
  const response = await backendApiClient.request<{ resource: { id: number } }>(
    '/api/integrations/mcp/servers/bind',
    { method: 'POST', body: { name, server_url: flat.server, server_type: flat.server_type, auth: flat.auth } },
  );
  const listResponse = await backendApiClient.request<{ items: SavedMcpServer[] }>(
    '/api/integrations/mcp/servers', { method: 'GET' },
  );
  return { resource: response.resource, servers: listResponse.items };
}

// ---------------------------------------------------------------------------
// Hook: MCP section state
// ---------------------------------------------------------------------------

function useMcpBlockState(config: McpBlockConfig, setConfig: MCPBlockSectionProps['setConfig']) {
  const [isFetching, setIsFetching] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [savedServers, setSavedServers] = useState<SavedMcpServer[]>([]);
  const [isLoadingServers, setIsLoadingServers] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const [serverMode, setServerMode] = useState<ServerMode>(() => {
    if (config.mcp_server_config_id) return 'saved';
    if (config.server || config.stdio_command) return 'manual';
    return 'saved';
  });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setIsLoadingServers(true);
      try {
        const r = await backendApiClient.request<{ items: SavedMcpServer[] }>('/api/integrations/mcp/servers', { method: 'GET' });
        if (!cancelled) setSavedServers(r.items);
      } catch { /* user can switch to manual */ }
      finally { if (!cancelled) setIsLoadingServers(false); }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!isLoadingServers && savedServers.length === 0 && serverMode === 'saved' && !config.mcp_server_config_id) {
      setServerMode('manual');
    }
  }, [isLoadingServers, savedServers.length, serverMode, config.mcp_server_config_id]);

  const handleFetchTools = useCallback(async () => {
    setIsFetching(true); setFetchError(null);
    try {
      const body: Record<string, unknown> = serverMode === 'saved'
        ? (() => { if (!config.mcp_server_config_id) throw new Error('skip'); return { mcp_server_config_id: config.mcp_server_config_id }; })()
        : (() => { const f = buildFlatServerConfig(config); if (!f.server) throw new Error('skip'); return { server: f.server, server_type: f.server_type, auth: f.auth }; })();
      const response = await backendApiClient.request<{ tools: McpToolInfo[] }>('/api/v1/mcp/tools', { method: 'POST', body });
      setConfig(prev => ({ ...prev, fetchedTools: response.tools }));
    } catch (err) {
      if (err instanceof Error && err.message === 'skip') return;
      setFetchError(err instanceof BackendAPIError ? err.message : 'Failed to fetch tools');
    } finally { setIsFetching(false); }
  }, [serverMode, config, setConfig]);

  useEffect(() => {
    if ((config.server || config.mcp_server_config_id || config.stdio_command) && config.tool && !config.fetchedTools) {
      handleFetchTools();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSaveServer = useCallback(async () => {
    setIsSaving(true);
    try {
      const { resource, servers } = await saveServerToBackend(config);
      setSavedServers(servers);
      const newServer = servers.find(s => s.id === resource.id);
      if (newServer) {
        setConfig(prev => ({
          ...prev, mcp_server_config_id: newServer.id, server: '', auth: undefined,
          stdio_command: undefined, stdio_args: undefined, stdio_env: undefined, http_headers: undefined,
        }));
        setServerMode('saved');
      }
    } catch (err) {
      setFetchError(err instanceof BackendAPIError ? err.message : 'Failed to save server');
    } finally { setIsSaving(false); }
  }, [config, setConfig]);

  const handleModeSwitch = useCallback((mode: ServerMode) => {
    setServerMode(mode);
    const resetFields = mode === 'manual'
      ? { mcp_server_config_id: null }
      : { server: '', auth: undefined, stdio_command: undefined, stdio_args: undefined, stdio_env: undefined, http_headers: undefined };
    setConfig(prev => ({ ...prev, ...resetFields, fetchedTools: undefined, tool: '', inputs: {} }));
    setFetchError(null);
  }, [setConfig]);

  return {
    isFetching, fetchError, savedServers, isLoadingServers, isSaving,
    serverMode, handleFetchTools, handleSaveServer, handleModeSwitch,
  };
}

// ---------------------------------------------------------------------------
// Server mode toggle
// ---------------------------------------------------------------------------

function ModeToggle({ serverMode, onSwitch }: { serverMode: ServerMode; onSwitch: (m: ServerMode) => void }) {
  return (
    <div className="flex gap-1 p-1 bg-muted rounded-md">
      <button
        type="button"
        className={cn(
          'flex-1 text-xs py-1.5 px-3 rounded-sm transition-colors',
          serverMode === 'saved' ? 'bg-background shadow-sm font-medium' : 'text-muted-foreground hover:text-foreground',
        )}
        onClick={() => onSwitch('saved')}
      >
        Saved Server
      </button>
      <button
        type="button"
        className={cn(
          'flex-1 text-xs py-1.5 px-3 rounded-sm transition-colors',
          serverMode === 'manual' ? 'bg-background shadow-sm font-medium' : 'text-muted-foreground hover:text-foreground',
        )}
        onClick={() => onSwitch('manual')}
      >
        Manual
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tool inputs section
// ---------------------------------------------------------------------------

function ToolInputsSection({
  config, fetchedTools, inputs, updateInputs, templateAutocomplete, validationErrors, onToolChange,
}: {
  config: McpBlockConfig; fetchedTools: McpToolInfo[] | undefined;
  inputs: Record<string, unknown>;
  updateInputs: (updater: (prev: Record<string, unknown>) => Record<string, unknown>) => void;
  templateAutocomplete: MCPBlockSectionProps['templateAutocomplete'];
  validationErrors: Record<string, string>;
  onToolChange: (toolName: string) => void;
}) {
  return (
    <>
      {fetchedTools && fetchedTools.length > 0 && (
        <McpToolSelector tools={fetchedTools} selectedTool={config.tool} onChange={onToolChange} />
      )}
      {config.tool && fetchedTools && fetchedTools.length > 0 ? (
        <SchemaInputs
          selectedTool={config.tool} tools={fetchedTools}
          inputs={inputs} updateInputs={updateInputs} templateAutocomplete={templateAutocomplete}
        />
      ) : (
        <DynamicFormField
          name="inputs" label="Inputs"
          description="Arguments passed to the tool. Use ${variable} to reference upstream outputs."
          value={inputs}
          onChange={value => updateInputs(() => value as Record<string, unknown>)}
          def={{ type: 'object' }} templateAutocomplete={templateAutocomplete}
          rows={4} error={validationErrors['inputs']}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Main section
// ---------------------------------------------------------------------------

export function MCPBlockSection({
  config, setConfig, templateAutocomplete, validationErrors = {},
}: MCPBlockSectionProps) {
  const state = useMcpBlockState(config, setConfig);

  const handleToolChange = useCallback(
    (toolName: string) => setConfig(prev => ({ ...prev, tool: toolName, inputs: {} })),
    [setConfig],
  );

  const updateInputs = useCallback(
    (updater: (prev: Record<string, unknown>) => Record<string, unknown>) => {
      setConfig(prev => ({ ...prev, inputs: updater(prev.inputs || {}) }));
    },
    [setConfig],
  );

  const handleSavedServerChange = useCallback(
    (server: SavedMcpServer | null) => {
      if (server) {
        setConfig(prev => ({
          ...prev, mcp_server_config_id: server.id,
          server: '', server_type: server.server_type, auth: undefined,
          stdio_command: undefined, stdio_args: undefined, stdio_env: undefined, http_headers: undefined,
          tool: '', inputs: {}, fetchedTools: undefined,
        }));
      } else {
        setConfig(prev => ({ ...prev, mcp_server_config_id: null, tool: '', inputs: {}, fetchedTools: undefined }));
      }
    },
    [setConfig],
  );

  const inputs = useMemo(() => config.inputs || {}, [config.inputs]);

  return (
    <div className="space-y-4">
      <ModeToggle serverMode={state.serverMode} onSwitch={state.handleModeSwitch} />

      {state.serverMode === 'saved' ? (
        <>
          <SavedServerSelector
            savedServers={state.savedServers} selectedId={config.mcp_server_config_id}
            onChange={handleSavedServerChange} isLoading={state.isLoadingServers}
          />
          {config.mcp_server_config_id && (
            <Button variant="outline" size="sm" className="w-full text-xs" disabled={state.isFetching} onClick={state.handleFetchTools}>
              {state.isFetching ? <><Loader2 className="mr-2 h-3 w-3 animate-spin" />Fetching tools...</> : 'Fetch Tools'}
            </Button>
          )}
          {state.fetchError && <p className="text-xs text-destructive">{state.fetchError}</p>}
        </>
      ) : (
        <McpServerFields
          config={config} setConfig={setConfig} validationErrors={validationErrors}
          isFetching={state.isFetching} fetchError={state.fetchError}
          onFetchTools={state.handleFetchTools} onSaveServer={state.handleSaveServer} isSaving={state.isSaving}
        />
      )}

      <ToolInputsSection
        config={config} fetchedTools={config.fetchedTools} inputs={inputs}
        updateInputs={updateInputs} templateAutocomplete={templateAutocomplete}
        validationErrors={validationErrors} onToolChange={handleToolChange}
      />
    </div>
  );
}
