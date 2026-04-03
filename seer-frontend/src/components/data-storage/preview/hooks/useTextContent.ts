import { useState, useEffect } from "react";

export interface UseTextContentResult {
  content: string | null;
  isLoading: boolean;
  error: string | null;
  isTruncated: boolean;
}

/**
 * Hook for fetching and streaming text content from a URL.
 * Handles large files by truncating content beyond maxSize.
 */
export function useTextContent(url: string, maxSize: number): UseTextContentResult {
  const [content, setContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isTruncated, setIsTruncated] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function fetchContent() {
      setIsLoading(true);
      setError(null);
      setIsTruncated(false);

      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch: ${response.status}`);
        }

        const contentLength = response.headers.get("content-length");
        const size = contentLength ? parseInt(contentLength, 10) : 0;

        if (size > maxSize) {
          setIsTruncated(true);
          // For large files, read partial content
          const reader = response.body?.getReader();
          if (!reader) throw new Error("Failed to read response");

          const chunks: Uint8Array[] = [];
          let totalRead = 0;

          while (totalRead < maxSize) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            totalRead += value.length;
          }

          reader.cancel();

          const decoder = new TextDecoder("utf-8");
          const text = chunks.map((chunk) => decoder.decode(chunk, { stream: true })).join("");
          if (!cancelled) {
            setContent(text);
          }
        } else {
          const text = await response.text();
          if (!cancelled) {
            setContent(text);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load content");
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    fetchContent();

    return () => {
      cancelled = true;
    };
  }, [url, maxSize]);

  return { content, isLoading, error, isTruncated };
}
