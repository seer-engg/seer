/**
 * WebSocket connection utility for live browser streaming.
 *
 * Provides a typed WebSocket wrapper with JWT authentication via query parameter,
 * auto-reconnection with exponential backoff, and JSON message handling.
 */
import { backendTokenProvider, getBackendBaseUrl } from './api-client';

export interface WebSocketOptions<TReceive> {
  /** Called when the connection opens. */
  onOpen?: () => void;
  /** Called when a message is received. */
  onMessage?: (data: TReceive) => void;
  /** Called when the connection closes. */
  onClose?: (code: number, reason: string) => void;
  /** Called on connection error. */
  onError?: (error: Event) => void;
  /** Enable auto-reconnection (default: false). */
  autoReconnect?: boolean;
  /** Max reconnection attempts (default: 5). */
  maxReconnectAttempts?: number;
  /** Base delay for exponential backoff in ms (default: 1000). */
  reconnectDelay?: number;
}

export class WebSocketConnection<TReceive, TSend = unknown> {
  private ws: WebSocket | null = null;
  private endpoint: string;
  private options: WebSocketOptions<TReceive>;
  private reconnectAttempts = 0;
  private isIntentionallyClosed = false;
  private reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;

  constructor(endpoint: string, options: WebSocketOptions<TReceive> = {}) {
    this.endpoint = endpoint;
    this.options = {
      autoReconnect: false,
      maxReconnectAttempts: 5,
      reconnectDelay: 1000,
      ...options,
    };
  }

  /**
   * Build WebSocket URL with JWT token.
   * Converts http(s):// to ws(s):// and appends token as query param.
   */
  private async buildUrl(): Promise<string> {
    const baseUrl = getBackendBaseUrl();
    const wsBase = baseUrl.replace(/^http/, 'ws');
    const normalizedEndpoint = this.endpoint.startsWith('/') ? this.endpoint : `/${this.endpoint}`;

    const token = await backendTokenProvider();
    const url = new URL(`${wsBase}${normalizedEndpoint}`);
    if (token) {
      url.searchParams.set('token', token);
    }
    return url.toString();
  }

  /**
   * Establish WebSocket connection.
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.isIntentionallyClosed = false;
    const url = await this.buildUrl();

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.options.onOpen?.();
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TReceive;
          this.options.onMessage?.(data);
        } catch (e) {
          console.warn('[WebSocket] Failed to parse message:', e);
        }
      };

      this.ws.onclose = (event) => {
        this.options.onClose?.(event.code, event.reason);

        if (
          !this.isIntentionallyClosed &&
          this.options.autoReconnect &&
          this.reconnectAttempts < (this.options.maxReconnectAttempts ?? 5)
        ) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (event) => {
        this.options.onError?.(event);
        reject(new Error('WebSocket connection failed'));
      };
    });
  }

  /**
   * Send a JSON message through the WebSocket.
   */
  send(data: TSend): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Cannot send: connection not open', {
        readyState: this.ws?.readyState,
        readyStateText: this.ws ? ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][this.ws.readyState] : 'no socket'
      });
      return;
    }
    this.ws.send(JSON.stringify(data));
  }

  /**
   * Close the WebSocket connection.
   */
  disconnect(): void {
    this.isIntentionallyClosed = true;
    if (this.reconnectTimeoutId !== null) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  /**
   * Check if the connection is open.
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get the current WebSocket ready state.
   */
  get readyState(): number | null {
    return this.ws?.readyState ?? null;
  }

  /**
   * Schedule a reconnection attempt with exponential backoff.
   */
  private scheduleReconnect(): void {
    const delay = (this.options.reconnectDelay ?? 1000) * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    this.reconnectTimeoutId = setTimeout(() => {
      this.reconnectTimeoutId = null;
      this.connect().catch((e) => {
        console.warn('[WebSocket] Reconnect failed:', e);
      });
    }, delay);
  }
}
