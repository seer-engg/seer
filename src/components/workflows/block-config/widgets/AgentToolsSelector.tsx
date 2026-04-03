import { useState, useMemo, useCallback } from 'react';
import { ChevronDown, ChevronRight, Search, Link2 } from 'lucide-react';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { useConnectIntegration } from '@/hooks/useConnectIntegration';
import { useToolingData } from '@/hooks/useToolingData';
import { cn } from '@/lib/utils';
import type { IntegrationGroup, ToolMetadata } from '@/stores/toolsStore';
import { AccountSelector } from './AccountSelector';
import { ResourcePicker } from '@/components/workflows/dialogs/ResourcePicker';
import type { ResourcePickerConfig } from '@/components/workflows/dialogs/ResourcePicker/types';
import type { IntegrationType } from '@/lib/integrations/client';

/**
 * Detects if a tool uses credential-based resource binding (Postgres, Supabase)
 * rather than OAuth. These tools have an `integration_resource_id` parameter
 * with an `x-resource-picker` annotation in their schema.
 */
function getBindingResourcePicker(tool: ToolMetadata): { config: ResourcePickerConfig; provider: string } | null {
  const props = tool.parameters?.properties;
  if (!props) return null;
  const resourceIdProp = props['integration_resource_id'];
  if (!resourceIdProp) return null;
  const rp = (resourceIdProp as Record<string, unknown>)['x-resource-picker'] as ResourcePickerConfig | undefined;
  if (!rp) return null;
  return { config: rp, provider: tool.provider || tool.integration_type || '' };
}

interface AgentToolsSelectorProps {
  selectedTools: string[];
  toolConnections: Record<string, number | null>;
  toolResourceIds: Record<string, number | null>;
  onToolsChange: (tools: string[]) => void;
  onConnectionChange: (toolName: string, connectionId: number | null) => void;
  onResourceIdChange: (toolName: string, resourceId: number | null) => void;
}

interface ToolItemProps {
  tool: ToolMetadata;
  isSelected: boolean;
  connectionId?: number | null;
  resourceId?: number | null;
  onToggle: (toolName: string, checked: boolean) => void;
  onConnectionChange: (toolName: string, connectionId: number | null) => void;
  onResourceIdChange: (toolName: string, resourceId: number | null) => void;
  onConnectAccount?: () => void;
}

function ToolItem({
  tool,
  isSelected,
  connectionId,
  resourceId,
  onToggle,
  onConnectionChange,
  onResourceIdChange,
  onConnectAccount,
}: ToolItemProps) {
  const requiresAuth = tool.integration_type && tool.integration_type !== 'sandbox';
  const bindingPicker = useMemo(() => getBindingResourcePicker(tool), [tool]);

  return (
    <div className="space-y-2">
      <div className="flex items-start gap-2 py-1">
        <Checkbox
          id={`tool-${tool.name}`}
          checked={isSelected}
          onCheckedChange={(checked) => onToggle(tool.name, checked === true)}
          className="mt-0.5"
        />
        <div className="flex-1 min-w-0">
          <Label
            htmlFor={`tool-${tool.name}`}
            className="text-sm font-normal cursor-pointer leading-tight"
          >
            {tool.name}
          </Label>
          {tool.description && (
            <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
              {tool.description}
            </p>
          )}
        </div>
      </div>
      {isSelected && bindingPicker && (
        <div className="ml-6">
          <Label className="text-xs font-medium">Database</Label>
          <ResourcePicker
            config={bindingPicker.config}
            provider={bindingPicker.provider}
            value={resourceId != null ? String(resourceId) : undefined}
            onChange={(value) => {
              const parsed = parseInt(value, 10);
              onResourceIdChange(tool.name, Number.isNaN(parsed) ? null : parsed);
            }}
            placeholder="Select a database..."
            className="text-xs mt-1"
          />
        </div>
      )}
      {isSelected && !bindingPicker && requiresAuth && (
        <div className="ml-6">
          <AccountSelector
            toolName={tool.name}
            selectedConnectionId={connectionId}
            onSelect={(id) => onConnectionChange(tool.name, id)}
            onConnectAccount={onConnectAccount}
          />
        </div>
      )}
    </div>
  );
}

interface IntegrationGroupSectionProps {
  group: IntegrationGroup;
  selectedTools: string[];
  toolConnections: Record<string, number | null>;
  toolResourceIds: Record<string, number | null>;
  onToolToggle: (toolName: string, checked: boolean) => void;
  onConnectionChange: (toolName: string, connectionId: number | null) => void;
  onResourceIdChange: (toolName: string, resourceId: number | null) => void;
  onConnectIntegration: (type: IntegrationType, toolNames: string[]) => void;
}

function IntegrationGroupSection({
  group,
  selectedTools,
  toolConnections,
  toolResourceIds,
  onToolToggle,
  onConnectionChange,
  onResourceIdChange,
  onConnectIntegration,
}: IntegrationGroupSectionProps) {
  const [isOpen, setIsOpen] = useState(false);
  const selectedCount = group.tools.filter((t) => selectedTools.includes(t.name)).length;

  // If all tools in this group use resource binding, hide OAuth-related UI
  const allToolsUseBinding = useMemo(
    () => group.tools.every((t) => getBindingResourcePicker(t) !== null),
    [group.tools]
  );

  const handleConnectAccount = useCallback(() => {
    const toolNames = group.tools.map((t) => t.name);
    onConnectIntegration(group.integration_type as IntegrationType, toolNames);
  }, [group, onConnectIntegration]);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className={cn(
            'flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm hover:bg-muted/50 transition-colors',
            isOpen && 'bg-muted/30'
          )}
        >
          <div className="flex items-center gap-2">
            {isOpen ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
            <span className="font-medium">{group.display_name}</span>
            {selectedCount > 0 && (
              <Badge variant="secondary" className="h-5 px-1.5 text-[10px]">
                {selectedCount}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            {!allToolsUseBinding && (
              group.isConnected ? (
                <Badge variant="outline" className="h-5 px-1.5 text-[10px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20">
                  Connected
                </Badge>
              ) : (
                <Badge variant="outline" className="h-5 px-1.5 text-[10px] bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20">
                  Not connected
                </Badge>
              )
            )}
            <span className="text-xs text-muted-foreground">{group.tools.length}</span>
          </div>
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent className="pl-6 pr-2 pb-2 space-y-1">
        {!allToolsUseBinding && !group.isConnected && (
          <Button
            variant="link"
            size="sm"
            className="h-auto p-0 text-xs text-seer mb-2"
            onClick={handleConnectAccount}
          >
            <Link2 className="h-3 w-3 mr-1" />
            Connect {group.display_name}
          </Button>
        )}
        {group.tools.map((tool) => (
          <ToolItem
            key={tool.name}
            tool={tool}
            isSelected={selectedTools.includes(tool.name)}
            connectionId={toolConnections[tool.name]}
            resourceId={toolResourceIds[tool.name]}
            onToggle={onToolToggle}
            onConnectionChange={onConnectionChange}
            onResourceIdChange={onResourceIdChange}
            onConnectAccount={handleConnectAccount}
          />
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}

export function AgentToolsSelector({
  selectedTools,
  toolConnections,
  toolResourceIds,
  onToolsChange,
  onConnectionChange,
  onResourceIdChange,
}: AgentToolsSelectorProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const { integrationGroups } = useToolingData();
  const connectIntegration = useConnectIntegration();

  const handleToolToggle = useCallback(
    (toolName: string, checked: boolean) => {
      if (checked) {
        onToolsChange([...selectedTools, toolName]);
      } else {
        onToolsChange(selectedTools.filter((t) => t !== toolName));
        // Clear connection/resource when tool is deselected
        if (toolConnections[toolName] !== undefined) {
          onConnectionChange(toolName, null);
        }
        if (toolResourceIds[toolName] !== undefined) {
          onResourceIdChange(toolName, null);
        }
      }
    },
    [selectedTools, toolConnections, toolResourceIds, onToolsChange, onConnectionChange, onResourceIdChange]
  );

  const handleConnectIntegration = useCallback(
    async (type: IntegrationType, toolNames: string[]) => {
      const redirectUrl = await connectIntegration(type, { toolNames });
      if (redirectUrl) {
        window.location.href = redirectUrl;
      }
    },
    [connectIntegration]
  );

  const filteredGroups = useMemo(() => {
    if (!searchQuery.trim()) return integrationGroups;

    const query = searchQuery.toLowerCase();
    return integrationGroups
      .map((group) => ({
        ...group,
        tools: group.tools.filter(
          (tool) =>
            tool.name.toLowerCase().includes(query) ||
            tool.description?.toLowerCase().includes(query)
        ),
      }))
      .filter((group) => group.tools.length > 0);
  }, [integrationGroups, searchQuery]);

  const totalSelectedCount = selectedTools.length;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">Tools</Label>
        {totalSelectedCount > 0 && (
          <Badge variant="secondary" className="h-5 px-1.5 text-[10px]">
            {totalSelectedCount} selected
          </Badge>
        )}
      </div>

      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search tools..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-8 h-8 text-sm"
        />
      </div>

      <div className="max-h-64 overflow-y-auto border rounded-md divide-y">
        {filteredGroups.length === 0 ? (
          <div className="p-4 text-center text-sm text-muted-foreground">
            {searchQuery ? 'No tools found matching your search' : 'No tools available'}
          </div>
        ) : (
          filteredGroups.map((group) => (
            <IntegrationGroupSection
              key={group.integration_type}
              group={group}
              selectedTools={selectedTools}
              toolConnections={toolConnections}
              toolResourceIds={toolResourceIds}
              onToolToggle={handleToolToggle}
              onConnectionChange={onConnectionChange}
              onResourceIdChange={onResourceIdChange}
              onConnectIntegration={handleConnectIntegration}
            />
          ))
        )}
      </div>
    </div>
  );
}
