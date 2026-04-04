/**
 * SSE (Server-Sent Events) stream parser utility.
 *
 * Parses a fetch Response body carrying an SSE stream into an async generator
 * of typed events. Each event follows the wire format:
 *
 *   id: {redis_msg_id}
 *   data: {json}
 *
 * Multiple events are separated by a blank line (\n\n).
 */

export interface SSEEvent<TData = unknown> {
  id: string;
  event?: string;
  data: TData;
}

/**
 * Reads an SSE Response body and yields parsed SSEEvent objects.
 *
 * Usage:
 *   const response = await backendApiClient.stream('/api/nexus/.../chat', { method: 'POST', ... });
 *   for await (const event of readSSEStream(response)) {
 *     // handle event
 *   }
 */
export async function* readSSEStream<TData = unknown>(
  response: Response,
): AsyncGenerator<SSEEvent<TData>> {
  if (!response.body) {
    throw new Error('Response body is null — SSE stream unavailable');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on double-newline which separates SSE events
      const parts = buffer.split('\n\n');

      // The last element may be an incomplete event — keep it in the buffer
      buffer = parts.pop() ?? '';

      for (const part of parts) {
        const event = parseSSEBlock(part.trim());
        if (event) yield event;
      }
    }

    // Flush any remaining content in the buffer
    if (buffer.trim()) {
      const event = parseSSEBlock(buffer.trim());
      if (event) yield event;
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Parses a single SSE event block (lines between blank lines) into an SSEEvent.
 * Returns null if the block cannot be parsed as a valid event.
 */
function parseSSEBlock<TData = unknown>(block: string): SSEEvent<TData> | null {
  if (!block) return null;

  let id = '';
  let eventName = '';
  const dataLines: string[] = [];

  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) {
      continue;
    }

    if (line.startsWith('id:')) {
      id = line.slice(3).trim();
    } else if (line.startsWith('event:')) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) return null;

  const dataLine = dataLines.join('\n');

  try {
    const parsed = JSON.parse(dataLine) as TData;
    return { id, event: eventName || undefined, data: parsed };
  } catch {
    console.warn('[sse-utils] Failed to parse SSE data payload:', dataLine);
    return null;
  }
}
