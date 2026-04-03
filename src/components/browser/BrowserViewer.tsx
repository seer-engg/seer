/**
 * Live browser streaming viewer component.
 *
 * Renders video frames from WebSocket to a canvas and captures
 * mouse, keyboard, and scroll events to dispatch to the browser.
 */
import { useRef, useEffect, useCallback, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useBrowserStream } from '@/hooks/useBrowserStream';
import { BrowserViewerToolbar } from './BrowserViewerToolbar';
import type { StreamStatus } from '@/types/browser';

interface BrowserViewerProps {
  sessionId: string | null;
  className?: string;
  width?: number;
  height?: number;
  showToolbar?: boolean;
  onStatusChange?: (status: StreamStatus) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

interface ViewerOverlayProps {
  status: StreamStatus;
  error: string | null;
  sessionId: string | null;
  hasFrame: boolean;
}

function ViewerOverlay({ status, error, sessionId, hasFrame }: ViewerOverlayProps) {
  if ((status === 'connecting' || !hasFrame) && sessionId) {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
        <div className="flex flex-col items-center gap-2">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <span className="text-sm text-muted-foreground">
            {status === 'connecting' ? 'Connecting to browser...' : 'Waiting for stream...'}
          </span>
        </div>
      </div>
    );
  }
  if (status === 'error') {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-destructive/10 z-10">
        <div className="flex flex-col items-center gap-2 text-destructive">
          <span className="text-sm font-medium">Connection Error</span>
          <span className="text-xs">{error || 'Failed to connect to browser'}</span>
        </div>
      </div>
    );
  }
  return null;
}

/** Calculates scaled coordinates accounting for letterboxing/pillarboxing */
function getScaledCoordinates(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement | null,
  dimensions: { width: number; height: number }
): { x: number; y: number } | null {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const containerAspect = rect.width / rect.height;
  const contentAspect = dimensions.width / dimensions.height;

  let renderedWidth: number, renderedHeight: number;
  if (containerAspect > contentAspect) {
    renderedHeight = rect.height;
    renderedWidth = rect.height * contentAspect;
  } else {
    renderedWidth = rect.width;
    renderedHeight = rect.width / contentAspect;
  }

  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;
  const clickX = clientX - rect.left;
  const clickY = clientY - rect.top;

  if (clickX < offsetX || clickX > offsetX + renderedWidth || clickY < offsetY || clickY > offsetY + renderedHeight) {
    return null;
  }
  const scaleX = dimensions.width / renderedWidth;
  const scaleY = dimensions.height / renderedHeight;
  return { x: Math.round((clickX - offsetX) * scaleX), y: Math.round((clickY - offsetY) * scaleY) };
}

interface UseBrowserEventHandlersParams {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  canvasDimensions: { width: number; height: number };
  isConnected: boolean;
  sendMouseEvent: (event: { action: string; x?: number; y?: number; button?: string; clickCount?: number }) => void;
  sendKeyEvent: (event: { action: string; key: string; code: string; text?: string }) => void;
  sendScrollEvent: (event: { x: number; y: number; deltaX: number; deltaY: number }) => void;
}

function useBrowserEventHandlers({ canvasRef, canvasDimensions, isConnected, sendMouseEvent, sendKeyEvent, sendScrollEvent }: UseBrowserEventHandlersParams) {
  const isDraggingRef = useRef(false);
  const dragStartPosRef = useRef<{ x: number; y: number } | null>(null);

  const getCoords = useCallback((clientX: number, clientY: number) => getScaledCoordinates(clientX, clientY, canvasRef.current, canvasDimensions), [canvasRef, canvasDimensions]);

  const handleMouseClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isConnected || isDraggingRef.current) { isDraggingRef.current = false; return; }
    const coords = getCoords(e.clientX, e.clientY);
    if (!coords) return;
    const button = e.button === 0 ? 'left' : e.button === 1 ? 'middle' : 'right';
    sendMouseEvent({ action: 'click', x: coords.x, y: coords.y, button, clickCount: 1 });
    canvasRef.current?.focus();
  }, [isConnected, getCoords, sendMouseEvent, canvasRef]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isConnected || e.buttons === 0) return;
    const coords = getCoords(e.clientX, e.clientY);
    if (!coords) return;
    if (dragStartPosRef.current && !isDraggingRef.current) {
      const dx = Math.abs(coords.x - dragStartPosRef.current.x);
      const dy = Math.abs(coords.y - dragStartPosRef.current.y);
      if (dx > 5 || dy > 5) {
        isDraggingRef.current = true;
        const button = e.buttons === 1 ? 'left' : e.buttons === 4 ? 'middle' : 'right';
        sendMouseEvent({ action: 'down', ...dragStartPosRef.current, button });
      }
    }
    if (isDraggingRef.current) sendMouseEvent({ action: 'move', x: coords.x, y: coords.y });
  }, [isConnected, getCoords, sendMouseEvent]);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (!isConnected) return;
    const coords = getCoords(e.clientX, e.clientY);
    if (!coords) return;
    dragStartPosRef.current = coords;
    isDraggingRef.current = false;
    canvasRef.current?.focus();
  }, [isConnected, getCoords, canvasRef]);

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (!isConnected) return;
    const coords = getCoords(e.clientX, e.clientY);
    if (!coords) { dragStartPosRef.current = null; isDraggingRef.current = false; return; }
    const button = e.button === 0 ? 'left' : e.button === 1 ? 'middle' : 'right';
    if (isDraggingRef.current) sendMouseEvent({ action: 'up', x: coords.x, y: coords.y, button });
    dragStartPosRef.current = null;
  }, [isConnected, getCoords, sendMouseEvent]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (!isConnected) return;
    e.preventDefault();
    sendKeyEvent({ action: 'down', key: e.key, code: e.code, text: e.key.length === 1 ? e.key : '' });
  }, [isConnected, sendKeyEvent]);

  const handleKeyUp = useCallback((e: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (!isConnected) return;
    e.preventDefault();
    sendKeyEvent({ action: 'up', key: e.key, code: e.code });
  }, [isConnected, sendKeyEvent]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => { e.preventDefault(); }, []);

  // Scroll handler effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (e: WheelEvent) => {
      if (!isConnected) return;
      e.preventDefault();
      e.stopPropagation();
      const coords = getCoords(e.clientX, e.clientY);
      if (!coords) return;
      sendScrollEvent({ x: coords.x, y: coords.y, deltaX: e.deltaX, deltaY: e.deltaY });
    };
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [canvasRef, isConnected, getCoords, sendScrollEvent]);

  return { handleMouseClick, handleMouseMove, handleMouseDown, handleMouseUp, handleKeyDown, handleKeyUp, handleContextMenu };
}

export function BrowserViewer({ sessionId, className, width = 1280, height = 800, showToolbar = true, onStatusChange, onConnect, onDisconnect }: BrowserViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasDimensions, setCanvasDimensions] = useState({ width, height });
  const [currentUrl, setCurrentUrl] = useState('');

  const { status, error, latestFrame, connect, disconnect, sendMouseEvent, sendKeyEvent, sendScrollEvent, sendNavigate, isConnected } = useBrowserStream(sessionId, {
    onStatusChange: (newStatus) => {
      onStatusChange?.(newStatus);
      if (newStatus === 'connected') onConnect?.();
      else if (newStatus === 'closed') onDisconnect?.();
    },
  });

  useEffect(() => { if (sessionId) connect(); return () => { disconnect(); }; }, [sessionId, connect, disconnect]);

  useEffect(() => {
    if (!latestFrame || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      if (img.width !== canvasDimensions.width || img.height !== canvasDimensions.height) {
        setCanvasDimensions({ width: img.width, height: img.height });
        canvas.width = img.width;
        canvas.height = img.height;
      }
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${latestFrame}`;
  }, [latestFrame, canvasDimensions.width, canvasDimensions.height]);

  const handlers = useBrowserEventHandlers({ canvasRef, canvasDimensions, isConnected, sendMouseEvent, sendKeyEvent, sendScrollEvent });

  const handleNavigate = useCallback((url: string) => { if (!isConnected) return; setCurrentUrl(url); sendNavigate(url); }, [isConnected, sendNavigate]);

  return (
    <div className={cn('flex flex-col rounded-lg border overflow-hidden', className)}>
      {showToolbar && <BrowserViewerToolbar currentUrl={currentUrl} status={status} error={error} onNavigate={handleNavigate} disabled={!isConnected} />}
      <div className="relative bg-black flex-1 min-h-[400px]">
        <ViewerOverlay status={status} error={error} sessionId={sessionId} hasFrame={!!latestFrame} />
        <canvas
          ref={canvasRef}
          width={canvasDimensions.width}
          height={canvasDimensions.height}
          tabIndex={0}
          onClick={handlers.handleMouseClick}
          onMouseMove={handlers.handleMouseMove}
          onMouseDown={handlers.handleMouseDown}
          onMouseUp={handlers.handleMouseUp}
          onKeyDown={handlers.handleKeyDown}
          onKeyUp={handlers.handleKeyUp}
          onContextMenu={handlers.handleContextMenu}
          className={cn('w-full h-full object-contain cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring', !isConnected && 'opacity-50 cursor-not-allowed')}
          style={{ aspectRatio: `${canvasDimensions.width} / ${canvasDimensions.height}` }}
        />
      </div>
    </div>
  );
}
