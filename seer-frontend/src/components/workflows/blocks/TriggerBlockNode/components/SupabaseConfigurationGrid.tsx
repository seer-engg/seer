import { useMemo, useState, useEffect } from 'react';
import { Database, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ResourcePicker } from '../../../dialogs/ResourcePicker';
import { ProjectDropdown } from './ProjectDropdown';
import {
  SUPABASE_EVENT_TYPES,
  type SupabaseEventType,
  type SupabaseConfigState,
  validateSupabaseConfig,
} from '../../../triggers/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useCheckResourceValid } from '@/hooks/useCheckResourceValid';
import { useWorkflowSave } from '@/hooks/useWorkflowSave';
import type { BindingState } from '../../../triggers/utils';
import { SupabaseOutputsDisplay } from './SupabaseOutputsDisplay';

const SUPABASE_PICKER_CONFIG = {
  resource_type: 'supabase_binding',
  endpoint: '/api/integrations/supabase/resources/bindings',
  display_field: 'display_name',
  value_field: 'id',
  search_enabled: true,
} as const;

const SUPABASE_SCHEMA_PICKER_CONFIG = {
  resource_type: 'supabase_schema',
  endpoint: '/api/integrations/supabase/resources/schemas',
  display_field: 'display_name',
  value_field: 'id',
  search_enabled: true,
  depends_on: 'integration_resource_id',
} as const;

const SUPABASE_TABLE_PICKER_CONFIG = {
  resource_type: 'supabase_table',
  endpoint: '/api/integrations/supabase/resources/tables',
  display_field: 'display_name',
  value_field: 'id',
  search_enabled: true,
  depends_on: 'integration_resource_id',
} as const;

interface SupabaseConfigurationGridProps {
  supabaseConfig: SupabaseConfigState;
  validation: ReturnType<typeof validateSupabaseConfig> | null;
  setConfigValue: (key: string, value: unknown) => void;
  supabaseConnected: boolean;
  isConnecting: boolean;
  onConnect: () => Promise<void>;
  onOpenBinding: () => void;
  bindingState: BindingState;
}

function useResourceValidity(integrationResourceId: string | null) {
  const checkResourceValid = useCheckResourceValid();
  const [resourceIsValid, setResourceIsValid] = useState<boolean>(true);
  const [isCheckingResource, setIsCheckingResource] = useState<boolean>(false);

  useEffect(() => {
    const checkResource = async () => {
      if (!integrationResourceId) {
        setResourceIsValid(true);
        return;
      }

      setIsCheckingResource(true);
      try {
        const isValid = await checkResourceValid(integrationResourceId);
        setResourceIsValid(isValid);
      } catch (error) {
        console.error('Error checking resource validity:', error);
        setResourceIsValid(false);
      } finally {
        setIsCheckingResource(false);
      }
    };

    checkResource();
  }, [integrationResourceId, checkResourceValid]);

  return { resourceIsValid, isCheckingResource };
}

interface GridRowProps {
  label: string;
  children: React.ReactNode;
}

function GridRow({ label, children }: GridRowProps) {
  return (
    <div className="space-y-2">
      <span className="text-sm font-medium">{label}</span>
      <div>{children}</div>
    </div>
  );
}

function ConnectionRow({
  supabaseConnected,
  isConnecting,
  onConnect,
}: {
  supabaseConnected: boolean;
  isConnecting: boolean;
  onConnect: () => Promise<void>;
}) {
  return (
    <GridRow label="Connection">
      {supabaseConnected ? (
        <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/20">
          Connected
        </Badge>
      ) : (
        <Button
          size="sm"
          onClick={onConnect}
          disabled={isConnecting}
          className="bg-amber-500 hover:bg-amber-600 text-white"
        >
          {isConnecting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Connect Supabase
        </Button>
      )}
    </GridRow>
  );
}

function ProjectRow({
  supabaseConfig,
  validation,
  resourceIsValid,
  isCheckingResource,
  onResourceChange,
  onOpenBinding,
}: {
  supabaseConfig: SupabaseConfigState;
  validation: ReturnType<typeof validateSupabaseConfig> | null;
  resourceIsValid: boolean;
  isCheckingResource: boolean;
  onResourceChange: (value: string | number, label?: string) => void;
  onOpenBinding: () => void;
}) {
  return (
    <div className="space-y-2">
      <span className="text-sm font-medium">Project</span>
      <div className="space-y-2">
        <div className="flex gap-2 items-center">
          <div className="flex-1">
            <ProjectDropdown
              value={supabaseConfig.integrationResourceId || ''}
              onChange={(value, label) => onResourceChange(value, label)}
              placeholder="Select project"
            />
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="px-2 h-10 text-xs text-muted-foreground hover:text-foreground whitespace-nowrap"
            onClick={onOpenBinding}
          >
            Bind new
          </Button>
        </div>
        {supabaseConfig.integrationResourceLabel && !isCheckingResource && !resourceIsValid && (
          <Alert variant="destructive" className="py-2">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-xs">
              Connection removed. Please re-select or bind a project.
            </AlertDescription>
          </Alert>
        )}
        {validation && !validation.valid && validation.errors.resource && (
          <p className="text-xs text-destructive">{validation.errors.resource}</p>
        )}
      </div>
    </div>
  );
}

function EventsRow({
  supabaseConfig,
  validation,
  onToggleEvent,
}: {
  supabaseConfig: SupabaseConfigState;
  validation: ReturnType<typeof validateSupabaseConfig> | null;
  onToggleEvent: (eventType: SupabaseEventType, checked: boolean) => void;
}) {
  return (
    <div className="space-y-2">
      <span className="text-sm font-medium">Events</span>
      <div className="space-y-2">
        <div className="flex gap-4">
          {SUPABASE_EVENT_TYPES.map((eventType) => (
            <label key={eventType} className="flex items-center gap-2">
              <Checkbox
                checked={supabaseConfig.events.includes(eventType)}
                onCheckedChange={(checked) => onToggleEvent(eventType, checked === true)}
              />
              <span className="text-xs font-medium">{eventType}</span>
            </label>
          ))}
        </div>
        {validation && !validation.valid && validation.errors.events && (
          <p className="text-xs text-destructive">{validation.errors.events}</p>
        )}
      </div>
    </div>
  );
}

export function SupabaseConfigurationGrid({
  supabaseConfig,
  validation,
  setConfigValue,
  supabaseConnected,
  isConnecting,
  onConnect,
  onOpenBinding,
  bindingState,
}: SupabaseConfigurationGridProps) {
  const { resourceIsValid, isCheckingResource } = useResourceValidity(
    supabaseConfig.integrationResourceId || null
  );

  const handleResourceChange = (value: string | number, label?: string) => {
    setConfigValue('integration_resource_id', value);
    if (label) {
      setConfigValue('integration_resource_label', label);
    }
    setConfigValue('schema', 'public');
    setConfigValue('table', '');
  };

  const handleEventToggle = (eventType: SupabaseEventType, checked: boolean) => {
    const nextEvents = checked
      ? Array.from(new Set([...supabaseConfig.events, eventType]))
      : supabaseConfig.events.filter((event) => event !== eventType);
    setConfigValue('events', nextEvents);
  };

  return (
    <div className="space-y-4">
      <ConnectionRow
        supabaseConnected={supabaseConnected}
        isConnecting={isConnecting}
        onConnect={onConnect}
      />

      <ProjectRow
        supabaseConfig={supabaseConfig}
        validation={validation}
        resourceIsValid={resourceIsValid}
        isCheckingResource={isCheckingResource}
        onResourceChange={handleResourceChange}
        onOpenBinding={onOpenBinding}
      />

      <GridRow label="Schema">
        <ResourcePicker
          config={SUPABASE_SCHEMA_PICKER_CONFIG}
          value={supabaseConfig.schema}
          dependsOnValues={{ integration_resource_id: supabaseConfig.integrationResourceId }}
          onChange={(value) => {
            setConfigValue('schema', String(value || ''));
            setConfigValue('table', '');
          }}
          placeholder="Select schema"
        />
      </GridRow>

      <GridRow label="Table">
        <div className="space-y-1">
          <ResourcePicker
            config={SUPABASE_TABLE_PICKER_CONFIG}
            value={supabaseConfig.table}
            dependsOnValues={{
              integration_resource_id: supabaseConfig.integrationResourceId,
              schema: supabaseConfig.schema || 'public',
            }}
            disabled={!supabaseConfig.integrationResourceId || !supabaseConfig.schema}
            onChange={(value) => setConfigValue('table', String(value || ''))}
            placeholder={supabaseConfig.schema ? 'Select table' : 'Select schema first'}
          />
          {validation && !validation.valid && validation.errors.table && (
            <p className="text-xs text-destructive">{validation.errors.table}</p>
          )}
        </div>
      </GridRow>

      <EventsRow
        supabaseConfig={supabaseConfig}
        validation={validation}
        onToggleEvent={handleEventToggle}
      />

      <GridRow label="Outputs">
        <SupabaseOutputsDisplay
          isConfigured={!!(supabaseConfig.integrationResourceId && supabaseConfig.table)}
        />
      </GridRow>
    </div>
  );
}
