import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { useActiveWorkflowId } from "@/hooks/useActiveWorkflowId";
import { Copy, Link, Play, Square, ChevronDown, ChevronRight } from "lucide-react";
import type { WorkflowNodeData } from "../../../types";
import { copyToClipboard } from "../components/handlers";
import { startListening, getPendingEvents } from "@/lib/api-client";
import type { PendingEventItem } from "@/lib/api-client";
import { useTriggersStore } from "@/stores/triggersStore";
import { useCanvasStore } from "@/stores/canvasStore";
import type { JsonObject } from "@/types/workflow-spec";
import { inferSchemaFromPayload } from "./utils";

export interface WebhookDetailsSectionProps {
  subscription: WorkflowNodeData["triggerMeta"]["subscription"];
  setDataSchema?: (schema: JsonObject) => void;
}

export interface EventCardProps {
  event: PendingEventItem;
  isExpanded: boolean;
  onToggle: () => void;
  setDataSchema?: (schema: JsonObject) => void;
  schemaApplied: boolean;
  onUseSchema: (data: Record<string, unknown>) => void;
}

export const EventCard: React.FC<EventCardProps> = ({
  event,
  isExpanded,
  onToggle,
  setDataSchema,
  schemaApplied,
  onUseSchema,
}) => (
  <div className="border border-border/60 rounded-md">
    <button
      className="w-full flex items-center justify-between px-2 py-1.5 text-xs hover:bg-muted/60 rounded-md transition-colors"
      onClick={onToggle}
    >
      <span className="text-muted-foreground">
        Event #{event.event_id}
        <span className="ml-2 text-muted-foreground/60">
          {new Date(event.received_at).toLocaleTimeString()}
        </span>
      </span>
      {isExpanded ? (
        <ChevronDown className="h-3 w-3 text-muted-foreground" />
      ) : (
        <ChevronRight className="h-3 w-3 text-muted-foreground" />
      )}
    </button>

    {isExpanded && (
      <div className="px-2 pb-2 space-y-2">
        <pre className="text-xs bg-muted/60 rounded p-2 overflow-auto max-h-32 whitespace-pre-wrap">
          {JSON.stringify(event.data, null, 2)}
        </pre>
        {setDataSchema &&
          (schemaApplied ? (
            <p className="text-xs text-center text-green-600 py-1">Schema applied — saving...</p>
          ) : (
            <Button size="sm" className="h-6 px-2 w-full" onClick={() => onUseSchema(event.data)}>
              Use this schema
            </Button>
          ))}
      </div>
    )}
  </div>
);

interface WebhookUrlSectionProps {
  webhookUrl: string;
  secretToken?: string;
}

const WebhookUrlSection: React.FC<WebhookUrlSectionProps> = ({ webhookUrl, secretToken }) => (
  <>
    <div className="space-y-1">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Link className="h-4 w-4" />
        Webhook endpoint
      </div>
      <code className="block text-xs break-all bg-background px-2 py-1 rounded">{webhookUrl}</code>
      <Button
        variant="outline"
        size="sm"
        className="h-7 px-3"
        onClick={() => copyToClipboard(webhookUrl, "Webhook URL")}
      >
        <Copy className="mr-2 h-3.5 w-3.5" />
        Copy URL
      </Button>
    </div>

    {secretToken && (
      <div className="space-y-1 pt-2 border-t border-dashed border-border/60">
        <p className="text-xs font-medium">Signing secret</p>
        <code className="text-xs break-all">{secretToken}</code>
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-3"
          onClick={() => copyToClipboard(secretToken, "Signing secret")}
        >
          <Copy className="mr-2 h-3.5 w-3.5" />
          Copy secret
        </Button>
      </div>
    )}
  </>
);

export interface ListeningSectionProps {
  isListening: boolean;
  isStarting: boolean;
  events: PendingEventItem[];
  expandedEvent: number | null;
  setExpandedEvent: (id: number | null) => void;
  onStartListening: () => void;
  onStopListening: () => void;
  setDataSchema?: (schema: JsonObject) => void;
  schemaApplied: boolean;
  onUseSchema: (data: Record<string, unknown>) => void;
}

export const ListeningSection: React.FC<ListeningSectionProps> = ({
  isListening,
  isStarting,
  events,
  expandedEvent,
  setExpandedEvent,
  onStartListening,
  onStopListening,
  setDataSchema,
  schemaApplied,
  onUseSchema,
}) => (
  <div className="pt-2 border-t border-dashed border-border/60 space-y-2">
    <div className="flex items-center justify-between">
      <p className="text-xs font-medium">
        {isListening ? "Listening for events..." : "Not listening"}
      </p>
      {isListening ? (
        <Button variant="outline" size="sm" className="h-6 px-2" onClick={onStopListening}>
          <Square className="mr-1 h-3 w-3" />
          Stop
        </Button>
      ) : (
        <Button size="sm" className="h-6 px-2" onClick={onStartListening} disabled={isStarting}>
          <Play className="mr-1 h-3 w-3" />
          {isStarting ? "Starting..." : "Listen"}
        </Button>
      )}
    </div>

    {events.length === 0 && isListening && (
      <p className="text-xs text-muted-foreground italic">
        Send a request to the webhook URL above to receive events here.
      </p>
    )}

    {events.length > 0 && (
      <div className="space-y-1.5">
        {events.map((event) => (
          <EventCard
            key={event.event_id}
            event={event}
            isExpanded={expandedEvent === event.event_id}
            onToggle={() =>
              setExpandedEvent(expandedEvent === event.event_id ? null : event.event_id)
            }
            setDataSchema={setDataSchema}
            schemaApplied={schemaApplied}
            onUseSchema={onUseSchema}
          />
        ))}
      </div>
    )}
  </div>
);

function useWebhookPolling(
  workflowId: string | null,
  triggerId: string | undefined,
  subscription: WebhookDetailsSectionProps["subscription"]
) {
  const [isStarting, setIsStarting] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [events, setEvents] = useState<PendingEventItem[]>([]);
  const [localWebhookUrl, setLocalWebhookUrl] = useState<string | null>(null);
  const [localSecretToken, setLocalSecretToken] = useState<string | null>(null);
  const latestEventIdRef = useRef<number | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startPolling = useCallback(() => {
    if (!workflowId || !triggerId) return;
    if (pollIntervalRef.current) return;

    const poll = async () => {
      try {
        const response = await getPendingEvents(
          workflowId,
          triggerId,
          latestEventIdRef.current ?? undefined
        );
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

  const absoluteWebhookUrl =
    localWebhookUrl ||
    ((subscription?.ui_meta as Record<string, unknown>)?.webhook_url as string | undefined) ||
    null;

  const secretToken =
    localSecretToken ||
    ((subscription?.ui_meta as Record<string, unknown>)?.secret_token as string | undefined) ||
    undefined;

  useEffect(() => {
    if (absoluteWebhookUrl) {
      setIsListening(true);
      startPolling();
    }
    return () => stopPolling();
  }, [absoluteWebhookUrl, startPolling, stopPolling]);

  const handleStartListening = async () => {
    if (!workflowId || !triggerId) return;
    setIsStarting(true);
    try {
      const response = await startListening(workflowId, triggerId);
      setLocalWebhookUrl(response.webhook_url);
      setLocalSecretToken(response.secret_token);
      useTriggersStore.getState().updateTrigger(workflowId, triggerId, {
        ui_meta: {
          ...((subscription?.ui_meta as Record<string, unknown>) || {}),
          webhook_url: response.webhook_url,
          secret_token: response.secret_token,
        },
      });
      useCanvasStore.getState().markDirty();
      setIsListening(true);
      startPolling();
    } catch (error) {
      console.error("Failed to start listening", error);
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
    absoluteWebhookUrl,
    secretToken,
    handleStartListening,
    handleStopListening,
  };
}

export const WebhookDetailsSection: React.FC<WebhookDetailsSectionProps> = ({
  subscription,
  setDataSchema,
}) => {
  const workflowId = useActiveWorkflowId();
  const [expandedEvent, setExpandedEvent] = useState<number | null>(null);
  const [schemaApplied, setSchemaApplied] = useState(false);
  const schemaAppliedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerId = subscription?.id;
  const {
    isStarting,
    isListening,
    events,
    absoluteWebhookUrl,
    secretToken,
    handleStartListening,
    handleStopListening,
  } = useWebhookPolling(workflowId, triggerId, subscription);

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
        Save this trigger to generate a trigger link and signing secret.
      </div>
    );
  }

  if (!absoluteWebhookUrl) {
    return (
      <div className="rounded-md border border-dashed p-3 space-y-2 bg-muted/40">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Link className="h-4 w-4" />
          Webhook endpoint
        </div>
        <p className="text-xs text-muted-foreground">
          Click below to start listening for webhook events and generate a URL.
        </p>
        <Button
          size="sm"
          className="h-7 px-3"
          onClick={handleStartListening}
          disabled={isStarting || !workflowId}
        >
          <Play className="mr-2 h-3.5 w-3.5" />
          {isStarting ? "Starting..." : "Start Listening"}
        </Button>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-dashed p-3 space-y-3 bg-muted/40">
      <WebhookUrlSection webhookUrl={absoluteWebhookUrl} secretToken={secretToken} />
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
