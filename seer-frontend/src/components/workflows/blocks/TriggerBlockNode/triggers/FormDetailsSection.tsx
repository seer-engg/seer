import { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { useActiveWorkflowId } from '@/hooks/useActiveWorkflowId';
import { Copy, Link2, ExternalLink, Play } from 'lucide-react';
import type { WorkflowNodeData } from '../../../types';
import { copyToClipboard } from '../components/handlers';
import { startListening, getPendingEvents } from '@/lib/api-client';
import type { PendingEventItem } from '@/lib/api-client';
import { useTriggersStore } from '@/stores/triggersStore';
import { useCanvasStore } from '@/stores/canvasStore';
import type { JsonObject } from '@/types/workflow-spec';
import { ListeningSection } from './WebhookTriggerConfig';
import { inferSchemaFromPayload } from './utils';

export interface FormDetailsSectionProps {
  subscription: WorkflowNodeData['triggerMeta']['subscription'];
  setDataSchema?: (schema: JsonObject) => void;
}

function useFormPolling(
  workflowId: string | null,
  triggerId: string | undefined,
  subscription: FormDetailsSectionProps['subscription'],
) {
  const [isStarting, setIsStarting] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [events, setEvents] = useState<PendingEventItem[]>([]);
  const [localFormUrl, setLocalFormUrl] = useState<string | null>(null);
  const latestEventIdRef = useRef<number | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startPolling = useCallback(() => {
    if (!workflowId || !triggerId) return;
    if (pollIntervalRef.current) return;

    const poll = async () => {
      try {
        const response = await getPendingEvents(workflowId, triggerId, latestEventIdRef.current ?? undefined);
        if (response.events.length > 0) {
          setEvents((prev) => {
            const existingIds = new Set(prev.map((e) => e.event_id));
            const newEvents = response.events.filter((e) => !existingIds.has(e.event_id));
            return [...prev, ...newEvents];
          });
        }
        if (response.latest_event_id != null) {
          latestEventIdRef.current = response.latest_event_id;
        }
      } catch {
        // Silently ignore poll errors
      }
    };

    poll();
    pollIntervalRef.current = setInterval(poll, 2000);
  }, [workflowId, triggerId]);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const formUrl = localFormUrl || (subscription?.ui_meta as Record<string, unknown>)?.form_url as string | undefined || null;

  useEffect(() => {
    if (formUrl) {
      setIsListening(true);
      startPolling();
    }
    return () => stopPolling();
  }, [formUrl, startPolling, stopPolling]);

  const handleStartListening = async () => {
    if (!workflowId || !triggerId) return;
    setIsStarting(true);
    try {
      const response = await startListening(workflowId, triggerId);
      if (response.form_url) {
        setLocalFormUrl(response.form_url);
        useTriggersStore.getState().updateTrigger(workflowId, triggerId, {
          ui_meta: {
            ...(subscription?.ui_meta as Record<string, unknown> || {}),
            form_url: response.form_url,
          },
        });
        useCanvasStore.getState().markDirty();
        setIsListening(true);
        startPolling();
      }
    } catch (error) {
      console.error('Failed to start listening for form', error);
    } finally {
      setIsStarting(false);
    }
  };

  const handleStopListening = () => {
    stopPolling();
    setIsListening(false);
  };

  return {
    isStarting,
    isListening,
    events,
    formUrl,
    handleStartListening,
    handleStopListening,
  };
}

interface FormUrlSectionProps {
  formUrl: string;
}

const FormUrlSection: React.FC<FormUrlSectionProps> = ({ formUrl }) => (
  <div className="space-y-2">
    <div className="flex items-center gap-2 text-sm font-medium">
      <Link2 className="h-4 w-4" />
      Public Form URL
    </div>
    <code className="block text-xs break-all bg-background px-2 py-1 rounded">
      {formUrl}
    </code>
    <div className="flex gap-2">
      <Button
        variant="outline"
        size="sm"
        className="h-7 px-3 flex-1"
        onClick={() => copyToClipboard(formUrl, 'Form URL')}
      >
        <Copy className="mr-2 h-3.5 w-3.5" />
        Copy URL
      </Button>
      <Button
        variant="outline"
        size="sm"
        className="h-7 px-3 flex-1"
        onClick={() => window.open(formUrl, '_blank')}
      >
        <ExternalLink className="mr-2 h-3.5 w-3.5" />
        Preview
      </Button>
    </div>
  </div>
);

export const FormDetailsSection: React.FC<FormDetailsSectionProps> = ({ subscription, setDataSchema }) => {
  const workflowId = useActiveWorkflowId();
  const [expandedEvent, setExpandedEvent] = useState<number | null>(null);
  const [schemaApplied, setSchemaApplied] = useState(false);
  const schemaAppliedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerId = subscription?.id;
  const {
    isStarting,
    isListening,
    events,
    formUrl,
    handleStartListening,
    handleStopListening,
  } = useFormPolling(workflowId, triggerId, subscription);

  useEffect(() => {
    return () => {
      if (schemaAppliedTimerRef.current) clearTimeout(schemaAppliedTimerRef.current);
    };
  }, []);

  const handleUseSchema = (eventData: Record<string, unknown>) => {
    if (!setDataSchema) return;
    const schema = inferSchemaFromPayload(eventData);
    setDataSchema(schema);
    setSchemaApplied(true);
    if (schemaAppliedTimerRef.current) clearTimeout(schemaAppliedTimerRef.current);
    schemaAppliedTimerRef.current = setTimeout(() => setSchemaApplied(false), 2500);
  };

  if (!subscription) {
    return (
      <div className="rounded-md border border-dashed p-3 text-sm text-muted-foreground bg-muted/40">
        Save this trigger first to start listening for form submissions.
      </div>
    );
  }

  if (!formUrl) {
    return (
      <div className="rounded-md border border-dashed p-3 space-y-2 bg-muted/40">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Link2 className="h-4 w-4" />
          Public Form URL
        </div>
        <p className="text-xs text-muted-foreground">
          Click below to generate a form URL and start listening for submissions.
        </p>
        <Button
          size="sm"
          className="h-7 px-3"
          onClick={handleStartListening}
          disabled={isStarting || !workflowId}
        >
          <Play className="mr-2 h-3.5 w-3.5" />
          {isStarting ? 'Starting...' : 'Start Listening'}
        </Button>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-dashed p-3 space-y-3 bg-muted/40">
      <FormUrlSection formUrl={formUrl} />
      <ListeningSection
        isListening={isListening}
        isStarting={isStarting}
        events={events}
        expandedEvent={expandedEvent}
        setExpandedEvent={setExpandedEvent}
        onStartListening={handleStartListening}
        onStopListening={handleStopListening}
        setDataSchema={setDataSchema}
        schemaApplied={schemaApplied}
        onUseSchema={handleUseSchema}
      />
    </div>
  );
};
