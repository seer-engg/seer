/**
 * Hook for managing WebSocket browser streaming.
 *
 * Handles connection lifecycle, frame receiving, and input dispatch.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { WebSocketConnection } from '@/lib/websocket';
import { useBrowserStore } from '@/stores/browserStore';
import type {
  ClientMessage,
  FrameMessage,
  MouseMessage,
  KeyMessage,
  ScrollMessage,
  ServerMessage,
  StreamStatus,
} from '@/types/browser';

export interface UseBrowserStreamOptions {
  /** Called when a new frame is received. */
  onFrame?: (frameData: string) => void;
  /** Called when stream status changes. */
  onStatusChange?: (status: StreamStatus) => void;
  /** Called on error. */
  onError?: (code: string, message: string) => void;
}

export interface UseBrowserStreamReturn {
  /** Current connection status. */
  status: StreamStatus;
  /** Last error message. */
  error: string | null;
  /** Latest frame data (base64 JPEG). */
  latestFrame: string | null;
  /** Connect to the stream. */
  connect: () => Promise<void>;
  /** Disconnect from the stream. */
  disconnect: () => void;
  /** Send a mouse event. */
  sendMouseEvent: (event: Omit<MouseMessage, 'type'>) => void;
  /** Send a keyboard event. */
  sendKeyEvent: (event: Omit<KeyMessage, 'type'>) => void;
  /** Send a scroll event. */
  sendScrollEvent: (event: Omit<ScrollMessage, 'type'>) => void;
  /** Navigate to a URL. */
  sendNavigate: (url: string) => void;
  /** Whether the stream is connected. */
  isConnected: boolean;
}

export function useBrowserStream(
  sessionId: string | null,
  options: UseBrowserStreamOptions = {}
): UseBrowserStreamReturn {
  const { setSessionStatus, setSessionError } = useBrowserStore();
  const [status, setStatus] = useState<StreamStatus>('closed');
  const [error, setError] = useState<string | null>(null);
  const [latestFrame, setLatestFrame] = useState<string | null>(null);

  const wsRef = useRef<WebSocketConnection<ServerMessage, ClientMessage> | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // Handle incoming messages
  const handleMessage = useCallback((data: ServerMessage) => {
    switch (data.type) {
      case 'frame': {
        const frame = data as FrameMessage;
        setLatestFrame(frame.data);
        optionsRef.current.onFrame?.(frame.data);
        break;
      }
      case 'status': {
        const newStatus = data.status as StreamStatus;
        setStatus(newStatus);
        setSessionStatus(newStatus);
        optionsRef.current.onStatusChange?.(newStatus);
        break;
      }
      case 'error': {
        setError(data.message);
        setSessionError(data.message);
        setStatus('error');
        setSessionStatus('error');
        optionsRef.current.onError?.(data.code, data.message);
        break;
      }
    }
  }, [setSessionStatus, setSessionError]);

  // Connect to WebSocket
  const connect = useCallback(async () => {
    if (!sessionId) {
      setError('No session ID');
      return;
    }

    setStatus('connecting');
    setSessionStatus('connecting');
    setError(null);
    setSessionError(null);

    const ws = new WebSocketConnection<ServerMessage, ClientMessage>(
      `/api/browser/sessions/${sessionId}/stream`,
      {
        onMessage: handleMessage,
        onOpen: () => {
          setStatus('connected');
          setSessionStatus('connected');
        },
        onClose: (code, reason) => {
          console.log(`[useBrowserStream] Closed: ${code} ${reason}`);
          setStatus('closed');
          setSessionStatus('closed');
        },
        onError: () => {
          setError('Connection failed');
          setSessionError('Connection failed');
          setStatus('error');
          setSessionStatus('error');
        },
      }
    );

    wsRef.current = ws;
    await ws.connect();
  }, [sessionId, handleMessage, setSessionStatus, setSessionError]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }
    setStatus('closed');
    setSessionStatus('closed');
    setLatestFrame(null);
  }, [setSessionStatus]);

  // Input dispatch functions
  const sendMouseEvent = useCallback((event: Omit<MouseMessage, 'type'>) => {
    const message = { type: 'mouse' as const, ...event };
    console.log('[useBrowserStream] sendMouseEvent', { hasWs: !!wsRef.current, message });
    wsRef.current?.send(message);
  }, []);

  const sendKeyEvent = useCallback((event: Omit<KeyMessage, 'type'>) => {
    wsRef.current?.send({ type: 'key', ...event });
  }, []);

  const sendScrollEvent = useCallback((event: Omit<ScrollMessage, 'type'>) => {
    wsRef.current?.send({ type: 'scroll', ...event });
  }, []);

  const sendNavigate = useCallback((url: string) => {
    wsRef.current?.send({ type: 'navigate', url });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.disconnect();
        wsRef.current = null;
      }
    };
  }, []);

  return {
    status,
    error,
    latestFrame,
    connect,
    disconnect,
    sendMouseEvent,
    sendKeyEvent,
    sendScrollEvent,
    sendNavigate,
    isConnected: status === 'connected',
  };
}
